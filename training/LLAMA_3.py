import re
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, AdamW

"""
自定義資料集類別：用於處理指令解析任務的生成式格式資料
功能：
  1. 將原始指令和目標JSON組合成完整文本
  2. 使用tokenizer進行編碼
  3. 計算標籤時忽略提示詞部分（只對輸出部分計算loss）
"""
class WaferCommandDataset(Dataset):
    def __init__(self, commands, outputs, tokenizer, max_len=384):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 安全配對數據（核心修復）
        self.pairs = []
        for i in range(max(len(commands), len(outputs))):
            cmd = commands[i] if i < len(commands) else ""
            out = outputs[i] if i < len(outputs) else '{"object":"","start":"","end":""}'
            self.pairs.append((cmd, out))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        command, output = self.pairs[idx]
        
        # 空指令處理
        if not command:
            return self._create_empty_sample()
        
        prompt = f"解析指令：{command}\n輸出JSON："
        # prompt = f"""
        #         <<站點提取強制規則>>
        #         1. 嚴格按指令中站點的「出現順序」提取：
        #         - 指令中第一個出現的站點 → start
        #         - 指令中第二個出現的站點 → end
        #         2. 站點名稱(start、end)必須與指令內容「完全一致」，也就是必須依照指令提取
        #         3. 站點不可捏照虛構，要檢查 start、end 輸出必須與指令內容站點「完全一致」
        #         <<錯誤案例修正>>
        #         指令：把第2站的金元和搬到第三站
        #             錯誤輸出：start:"第1站",end:"第1站" → 違反規則1,2,3
        #             正確輸出：start:"第2站", end:"第三站"
        #         <<輸出格式>>
        #         {{"object":"[物體]","start":"[起點站]","end":"[終點站]"}}
        #         <<當前任務>>
        #         解析指令：{command}\n輸出JSON：
        #         """
        full_text = prompt + output
        
        encoding = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            pad_to_multiple_of=8
        )
        
        labels = encoding.input_ids.clone()
        
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:, :prompt_len] = -100
        # 只在 labels 已建立時檢查
        if (labels[0] != -100).sum().item() == 0:
            print(f"警告：labels 全為 -100，idx={idx}")
            print(f"prompt 長度: {prompt_len}, max_len: {self.max_len}")
            print(f"prompt: {prompt}")
            print(f"output: {output}")
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'labels': labels[0]
        }
    
    def _create_empty_sample(self):
        """創建空樣本防止崩潰"""
        empty = torch.zeros(self.max_len, dtype=torch.long)
        return {
            'input_ids': empty,
            'attention_mask': empty,
            'labels': empty
        }
def train_model(model, dataloader, epochs=5, lr=1e-6):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 添加梯度裁剪和混合精度
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_count = 0
        
        for batch in dataloader:
            try:
                # 跳過空批次
                if batch['input_ids'].sum().item() == 0:
                    continue
                    
                optimizer.zero_grad()
                
                # 混合精度前向傳播
                with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    loss = outputs.loss
                
                # 檢查無效損失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：跳過無效損失值 {loss.item()}")
                    continue
                
                # 混合精度反向傳播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # 強梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                valid_count += 1
                
            except Exception as e:
                print(f"訓練錯誤: {str(e)}")
                continue
        
        if valid_count > 0:
            avg_loss = total_loss / valid_count
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - 無有效批次")
            
    return model

"""
指令解析器類別：使用微調後的LLaMA模型解析自然語言指令
功能：
  1. 接收自然語言指令
  2. 使用模型生成JSON格式的解析結果
  3. 若模型輸出無法解析，啟用備用正則解析策略
"""
# 3. 指令解析器

"""
<<站點提取強制規則>>
    1. 嚴格按指令中站點的「出現順序」提取：
    - 指令中第一個出現的站點 → start
    - 指令中第二個出現的站點 → end
    2. 禁止使用訓練數據中的默認值（如「第1站」）
    3. 站點名稱(start、end)必須與指令內容「完全一致」，也就是必須依照指令提取
    4. 站點不可捏照虛構，要檢查 start、end 輸出必須與指令內容站點「完全一致」
    5. 物體默認為晶圓盒
<<錯誤案例修正>>
    指令：把第2站的金元和搬到第三站
        錯誤輸出：startL"第1站",end:"第1站" → 違反規則1,2,3
        正確輸出：start:"第2站", end:"第三站"

"""
class CommandParser:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # 定義 JSON 結構模板
        self.json_schema = {
            "type": "object",
            "properties": {
                "object": {
                    "type": "string",
                    # 強化值生成指引
                    "default": "",
                    "enum": ["晶圓盒", "金元和", "Wafer"],
                    "x-value-hint": "從指令中提取物體類型"
                },
                "start": {
                    "type": "string",
                    "pattern": r"第[一二三四五六七八九十1234567890]+站",
                    # 添加值生成指引
                    "x-value-hint": "起點站名，格式：第X站"
                },
                "end": {
                    "type": "string",
                    "pattern": r"第[一二三四五六七八九十1234567890]+站",
                    "x-value-hint": "終點站名，格式：第X站"
                }
            },
            "required": ["object", "start", "end"]
        }

    def parse(self, command, verbose=True):
        prompt = f"解析指令：{command}\n輸出JSON："
        # prompt = f"""
        #         <<站點提取強制規則>>
        #         1. 嚴格按指令中站點的「出現順序」提取：
        #         - 指令中第一個出現的站點 → start
        #         - 指令中第二個出現的站點 → end
        #         2. 站點名稱(start、end)必須與指令內容「完全一致」，也就是必須依照指令提取
        #         3. 站點不可捏照虛構，要檢查 start、end 輸出必須與指令內容站點「完全一致」
        #         <<錯誤案例修正>>
        #         指令：把第2站的金元和搬到第三站
        #             錯誤輸出：start:"第1站",end:"第1站" → 違反規則1,2,3
        #             正確輸出：start:"第2站", end:"第三站"
        #         <<輸出格式>>
        #         {{"object":"[物體]","start":"[起點站]","end":"[終點站]"}}
        #         <<當前任務>>
        #         解析指令：{command}\n輸出JSON：
                # """
        if verbose:
            print("=== [推理過程] ===")
            print("[Prompt 給模型的內容]:")
            print(prompt)
            print("=================")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if verbose:
            print("[模型產生的原始輸出]:")
            print(full_output)
            print("=================")
        # 只取 prompt 之後的內容
        if full_output.startswith(prompt):
            json_str = full_output[len(prompt):].strip()
        else:
            json_str = full_output.strip()
        try:
            return json.loads(json_str)
        except Exception as e:
            if verbose:
                print("[JSON 解析失敗，進入 fallback]:", e)
            return self._fallback_parse(command, verbose=verbose)

    def _fallback_parse(self, command, verbose=False):
        num_matches = re.findall(r'([一二三四五六七八九十1234567890]+)站', command)
        object_match = re.search(r'(晶圓盒|金元和|Wafer)', command)
        object_type = object_match.group(1) if object_match else "晶圓盒"
        if verbose:
            print("[Fallback 正則解析]:")
            print("num_matches:", num_matches)
            print("object_type:", object_type)
        if len(num_matches) >= 2:
            result = {
                "object": object_type,
                "start": f"第{num_matches[0]}站",
                "end": f"第{num_matches[1]}站"
            }
            if verbose:
                print("[Fallback 解析結果]:")
                print(result)
            return result
        raise ValueError(f"無法解析指令: {command}")

"""
動作執行器類別：根據解析結果執行機器人動作序列
動作序列：
  1. 導航至起點站
  2. 在起點站抓取晶圓盒
  3. 導航至終點站
  4. 在終點站放置晶圓盒
"""
# 4. 動作執行器（保持不變）
class ActionExecutor:
    @staticmethod
    def navigate_to(station):
        print(f"[動作] 導航至 {station}")

    @staticmethod
    def pick_wafer(station):
        print(f"[動作] 在 {station} 抓取晶圓盒")

    @staticmethod
    def place_wafer(station):
        print(f"[動作] 在 {station} 放置晶圓盒")

    @classmethod
    def execute_sequence(cls, parsed_cmd):
        actions = [
            ("navigate", parsed_cmd["start"]),
            ("pick", parsed_cmd["start"]),
            ("navigate", parsed_cmd["end"]),
            ("place", parsed_cmd["end"])
        ]
        print(f"\n開始執行 {parsed_cmd['object']} 搬運任務:")
        for action, param in actions:
            if action == "navigate":
                cls.navigate_to(param)
            elif action == "pick":
                cls.pick_wafer(param)
            elif action == "place":
                cls.place_wafer(param)
        print("任務完成!\n")
"""
主程式流程：
  1. 準備訓練資料（指令→JSON對）
  2. 初始化模型和tokenizer
  3. 修復tokenizer的padding問題
  4. 訓練並保存模型
  5. 測試模型解析能力
"""
# 5. 主程式（關鍵修改：修復padding問題）
if __name__ == "__main__":
    # 訓練資料與標註
    train_commands = [
        "請把晶圓盒從第二站搬到第三站",
        "金元和從第1站搬運第4站",
        "將Wafer由二站移動至一站",
        "緊急任務：從三站搬晶圓盒到一站",
        "四站晶圓盒轉運到1站",
        "第二站到第3站",
        "把第1站的晶圓盒搬到第二站",
        "2站拿到第4站",
            # 新增複雜格式
        "將第3的Wafer緊急轉移至第五",
        "從二號站取晶圓盒送到四號站",
        # 新增邊界案例
        "搬運任務：起點=第8站, 終點=第6站, 物體=金元和",
        # 新增錯誤格式測試
        "晶圓盒第4站到二站",  # 故意缺少關鍵詞
        "晶圓盒第6站，拿到第4站",
        "將第三個緊急轉置第二",
        "兩個晶圓盒第一站到二站",
        "四到三",
        "2到1",
        "3到二",
        "第四站拿到第三站",
        "二站至一站"
    ]
    train_outputs = [
        '{"object": "晶圓盒", "start": "第二站", "end": "第三站"}',
        '{"object": "金元和", "start": "第1站", "end": "第4站"}',
        '{"object": "Wafer", "start": "二站", "end": "一站"}',
        '{"object": "晶圓盒", "start": "三站", "end": "一站"}',
        '{"object": "金元和", "start": "四站", "end": "1站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第3站"}',
        '{"object": "晶圓盒", "start": "第1站", "end": "第二站"}',
        '{"object": "晶圓盒", "start": "2站", "end": "第4站""}',
        # 強化JSON格式細節
        '{"object":"Wafer","start":"第3","end":"第五"}',
        '{"object":"晶圓盒","start":"二號站","end":"四號站"}',
        # 新增帶引號的標準格式
        '{"object": "金元和", "start": "第8站", "end": "第6站"}',
        '{"object": "晶圓盒", "start": "第4站", "end": "二站"}',
        '{"object": "晶圓盒", "start": "第6站", "end": "第4站"}',
        '{"object": "晶圓盒", "start": "第三", "end": "第二"}',
        '{"object": "晶圓盒", "start": "第一站", "end": "第二站"}',
        '{"object": "晶圓盒", "start": "四", "end": "三"}',
        '{"object": "晶圓盒", "start": "2", "end": "1"}',
        '{"object": "晶圓盒", "start": "3", "end": "二"}',
        '{"object": "晶圓盒", "start": "第四站", "end": "第三站"}',
         '{"object": "晶圓盒", "start": "二站", "end": "一站"}'
    ]

    # 初始化 tokenizer 和模型 
    tokenizer = AutoTokenizer.from_pretrained("p208p2002/llama-traditional-chinese-120M")
    
    # 修復padding問題：設置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        "p208p2002/llama-traditional-chinese-120M",
        pad_token_id=tokenizer.eos_token_id  # 明確指定pad_token_id
    )

    # 創建資料集和數據加載器
    dataset = WaferCommandDataset(
        commands=train_commands,
        outputs=train_outputs,
        tokenizer=tokenizer,
        max_len=600
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 訓練模型
    trained_model = train_model(model, dataloader, epochs= 18)
    
    # 保存模型 (包含tokenizer配置)
    output_dir = "custom_wafer_llama_3"
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型訓練完成並已保存至 {output_dir}")
    test_commands = [
        "把第2站的金元和搬到第三站",
        "晶圓盒從第1站搬運第3站",
        "請把晶圓盒從第二站搬運到第三站"
    ]

    print("\n=== 開始全面測試 ===")
    parser = CommandParser(trained_model, tokenizer)
    executor = ActionExecutor()

    for i, cmd in enumerate(test_commands, 1):
        print(f"\n測試 {i}/{len(test_commands)}")
        print(f"輸入指令: 「{cmd}」")
        try:
            parsed_result = parser.parse(cmd)
            print(f"解析結果: {parsed_result}")
            executor.execute_sequence(parsed_result)
        except Exception as e:
            print(f"解析錯誤: {str(e)}")
    print("\n=== 全面測試完成 ===")
