import re
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, AdamW
import random
"""
自定義資料集類別：用於處理指令解析任務的生成式格式資料
功能：
  1. 將原始指令和目標JSON組合成完整文本
  2. 使用tokenizer進行編碼
  3. 計算標籤時忽略提示詞部分（只對輸出部分計算loss）
"""
class WaferCommandDataset(Dataset):
    def __init__(self, commands, outputs, tokenizer, max_len=128):
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
        
       
        
        prompt = f"依據語序，第一個地點為起點（start），第二個地點為終點（end）。指令：{command}\n請輸出對應 JSON（不可加解釋）："
        # prompt = f"解析指令：{command}\n輸出JSON："


        full_text = prompt + output
        
        encoding = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            #pad_to_multiple_of=8
        )
        
        labels = encoding.input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:, :prompt_len] = -100
        
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
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, train_loss):
        if train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True    
# def train_model(model, dataloader, epochs=5, lr=1e-6):  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     # 優化器
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
#     # 設定混合精度訓練（Mixed Precision Training）
#     scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda') # 只有在你用 GPU 訓練時才會啟用這個功能，CPU 則自動關閉。
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         valid_count = 0
        
#         for batch in dataloader:
#             try:
#                 # 跳過空批次
#                 if batch['input_ids'].sum().item() == 0:
#                     continue
                    
#                 optimizer.zero_grad() # 歸零所有參數的梯度，為新一輪反向傳播做準備
                
#                 # 混合精度前向傳播，在 CPU 上這個區塊等同於普通的前向傳播， autocast 其實沒啟用
#                 with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
#                     outputs = model(
#                         input_ids=batch['input_ids'].to(device),
#                         attention_mask=batch['attention_mask'].to(device),
#                         labels=batch['labels'].to(device)
#                     )
#                     loss = outputs.loss
                
#                 # 檢查無效損失
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     print(f"警告：跳過無效損失值 {loss.item()}")
#                     continue
                
#                 # 混合精度反向傳播
#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
                
#                 # 強梯度裁剪 所有參數的「梯度」總長度（L2 norm），如果超過 0.5，就把它縮小到 0.5 以內防止「梯度爆炸」
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
#                 scaler.step(optimizer)
#                 scaler.update()
                
#                 total_loss += loss.item()
#                 valid_count += 1
                
#             except Exception as e:
#                 print(f"訓練錯誤: {str(e)}")
#                 continue
        
#         if valid_count > 0:
#             avg_loss = total_loss / valid_count
#             print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
#         else:
#             print(f"Epoch {epoch+1}/{epochs} - 無有效批次")
            
#     return model

# def train_model(model, dataloader, epochs=20, lr=1e-6, patience=3):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
#     scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
#     early_stopper = EarlyStopping(patience=patience)

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         valid_count = 0
#         for batch in dataloader:
#             try:
#                 if batch['input_ids'].sum().item() == 0:
#                     continue
#                 optimizer.zero_grad()
#                 with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
#                     outputs = model(
#                         input_ids=batch['input_ids'].to(device),
#                         attention_mask=batch['attention_mask'].to(device),
#                         labels=batch['labels'].to(device)
#                     )
#                     loss = outputs.loss
#                     if torch.isnan(loss) or torch.isinf(loss):
#                         print(f"警告：跳過無效損失值 {loss.item()}")
#                         continue
#                     scaler.scale(loss).backward()
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#                     scaler.step(optimizer)
#                     scaler.update()
#                     total_loss += loss.item()
#                     valid_count += 1
#             except Exception as e:
#                 print(f"訓練錯誤: {str(e)}")
#                 continue
#         if valid_count > 0:
#             avg_loss = total_loss / valid_count
#             print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
#         else:
#             avg_loss = float('inf')
#             print(f"Epoch {epoch+1}/{epochs} - 無有效批次")
#         # Early Stopping 判斷
#         early_stopper(avg_loss)
#         if early_stopper.early_stop:
#             print(f"訓練 loss 未改善，提前於第 {epoch+1} epoch 結束訓練。")
#             break
#     return model

def train_model2(model, dataloader, epochs=20, lr=2e-5, patience=3, eval_prompts=None, tokenizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_count = 0
        for batch in dataloader:
            try:
                if batch['input_ids'].sum().item() == 0:
                    continue
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
                    loss = outputs.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告：跳過無效損失值 {loss.item()}")
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
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
            avg_loss = float('inf')
            print(f"Epoch {epoch+1}/{epochs} - 無有效批次")

        # 🔍 測試推論輸出效果
        if eval_prompts and tokenizer:
            print("\n[推論測試範例]")
            model.eval()
            parser = CommandParser(model, tokenizer)
            for cmd in eval_prompts[:4]:  # 顯示前4句即可
                try:
                    result = parser.parse(cmd)
                    print(f"  指令：{cmd}")
                    print(f"  模型解析：{result}")
                except Exception as e:
                    print(f"  ❌解析失敗：{e}")
            print("[測試結束]\n")

        # Early Stopping 判斷
        early_stopper(avg_loss)
        if early_stopper.early_stop:
            print(f"訓練 loss 未改善，提前於第 {epoch+1} epoch 結束訓練。")
            break

    return model


"""
指令解析器類別：使用微調後的LLaMA模型解析自然語言指令
功能：
  1. 接收自然語言指令
  2. 使用模型生成JSON格式的解析結果
  3. 若模型輸出無法解析，啟用備用正則解析策略
"""
# 3. 指令解析器
class CommandParser:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def parse(self, command):
        
        prompt = f"依據語序，第一個地點為起點（start），第二個地點為終點（end）。指令：{command}\n請輸出對應 JSON（不可加解釋）："

        # prompt = f" 解析指令：{command}\n輸出JSON："
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
         # 標準生成
        output_ids = self.model.generate(
            input_ids=inputs.input_ids, # 輸入 prompt 的 token ID，模型根據這些內容生成新文字。
            attention_mask=inputs.attention_mask,  # 注意力遮罩:因為句子長度不一，要告訴模型哪些 token 是有效的，哪些只是「補齊用的空白（padding）」
            max_new_tokens= 160,  # 最多生成 160 個新 token，避免生成太長
            num_beams=3, # 使用 beam search，保留 3 條候選路徑，能探索更多可能性，避免只走單一路徑。生成的句子更自然、多樣。(每一步都只選擇機率最大的詞，結果常常單調、重複)
            do_sample=False, # 不隨機取樣，保證生成結果可重現   
            pad_token_id=self.tokenizer.eos_token_id, # 補齊讓一個 batch 裡所有輸入的長度一致，方便並行運算
            forced_bos_token_id=self.tokenizer.encode("{")[1]  # 強制JSON開頭，強制生成的第一個 token 必須是 {
        )
        
        # 提取JSON部分
        # full_output = self.tokenizer.decode(output_ids[0]) # 把模型產生的 token 轉回完整的文字
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 把字串用 "輸出JSON：" 這個關鍵詞分割，取出後面段落（[-1]），這樣可以跳過前面不是 JSON 的部分。 再用 } 分割，取第一段（），然後補上一個 }(避免重複輸出)
        # json_str = full_output.split("輸出JSON：")[-1].split("}")[0] + "}"  
        # 嘗試從 full_output 中找出第一個 {...} 結構
        match = re.search(r'\{.*?\}', full_output)
        print("full output:\n")
        print(full_output)
        if match:
            json_str = match.group()
            print(f"\njson_str\n {json_str}")

            return json.loads(json_str)
        else:
            raise ValueError(f"無法解析指令: {command}")


        # json_str = full_output.split("輸出：")[0].split("}")[0] + "}"
        # print("full output:\n")
        # print(full_output)
        # print(f"json_str\n {json_str}")


        # try:
        #     return json.loads(json_str) # JSON 字串轉回 Python 字典
        # except:
        #     raise ValueError(f"無法解析指令: {command}")
            # return self._fallback_parse(command) #備用方案

    def _fallback_parse(self, command):
        print("啟用備用方案~")
        num_matches = re.findall(r'([一二三四五六七八九十1234567890]+)站', command)
        object_match = re.search(r'(晶圓盒|金元和|Wafer)', command)
        object_type = object_match.group(1) if object_match else "晶圓盒"
        
        if len(num_matches) >= 2:
            return {
                "object": object_type,
                "start": f"第{num_matches[0]}站",
                "end": f"第{num_matches[1]}站"
            }
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
def data_pre():
    train_commands = [
        "請把晶圓盒從第二站搬到第三站",
        "金元和從第一站搬運第四站",
        "將Wafer由二站移動至一站",
        "緊急任務：從三站搬晶圓盒到一站",

        "四站晶圓盒轉運到1站",
        "第二站到第3站",
        "把第1站的晶圓盒搬到第二站",
        "第二站拿到第四站",

        "將第一的Wafer緊急轉移至第五",            # 新增複雜格式
        "從二號站取晶圓盒送到四號站",
        "搬運任務：起點=第二站, 終點=第六站, 物體=金元和", # 新增邊界案例
        "晶圓盒第4站到二站",  # 故意缺少關鍵詞

        "晶圓盒第ㄧ站，拿到第四站",
        "將第三個緊急轉置第二",
        "兩個晶圓盒第一站到二站",
        "四到三",

        "2到1",
        "3到二",
        "第四站拿到第三站",
        "二站至一站",

        "請將晶圓盒從第五站搬到第六站",
        "金元和從第七站搬運第二站",
        "將Wafer由三站移動至四站",
        "緊急任務：從六站搬晶圓盒到五站",

        "八站晶圓盒轉運到二站",
        "第五站到第八站",
        "把第三站的晶圓盒搬到第一站",
        "四站拿到第一站",

        "將第四的Wafer緊急轉移至第二",
        "從七號站取晶圓盒送到五號站",
        "搬運任務：起點=第三站, 終點=第9站, 物體=金元和",
        "晶圓盒第三站到二站",

        "晶圓盒第四站，拿到第三站",
        "將第五個緊急轉置第六",
        "兩個晶圓盒第九站到第八站",
        "一到六",

        "五到二",
        "一到三",
        "第七站拿到第六站",
        "四站至五站",

        "請把Wafer從八站轉運到三站",
        "搬運任務：起點=二站, 終點=四站, 物體=晶圓盒",
        "金元和從五站送到一站",
        "Wafer從一站移動到二站",

        "晶圓盒從三站搬到七站",
        "緊急：從四站搬晶圓盒到八站",
        "二站晶圓盒轉運到一站",
        "第一站到第三站",

        "第四站拿到第七站",
        "一站到二站",
        "請把晶圓盒從第二站搬運到第一站",
        "第一站拿到第二站",

        "請把晶圓盒從第二站拿到第一站",
        "請幫我把它從第二站，搬到第三站",
        "第三站搬到第四站",
        "第二站，搬到第一站",
    ]
    train_outputs = [
        '{"object": "晶圓盒", "start": "第二站", "end": "第三站"}',
        '{"object": "金元和", "start": "第一站", "end": "第四站"}',
        '{"object": "Wafer", "start": "二站", "end": "一站"}',
        '{"object": "晶圓盒", "start": "三站", "end": "一站"}',

        '{"object": "金元和", "start": "四站", "end": "1站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第3站"}',
        '{"object": "晶圓盒", "start": "第1站", "end": "第二站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第四站"}',
        # 強化JSON格式細節
        '{"object":"Wafer", "start":"第一", "end":"第五"}',
        '{"object":"晶圓盒", "start":"二號站", "end":"四號站"}',
        '{"object": "金元和", "start": "第二站", "end": "第六站"}',# 新增帶引號的標準格式
        '{"object": "晶圓盒", "start": "第4站", "end": "二站"}',

        '{"object": "晶圓盒", "start": "第一站", "end": "第四站"}',
        '{"object": "晶圓盒", "start": "第三", "end": "第二"}',
        '{"object": "晶圓盒", "start": "第一站", "end": "第二站"}',
        '{"object": "晶圓盒", "start": "四", "end": "三"}',

        '{"object": "晶圓盒", "start": "2", "end": "1"}',
        '{"object": "晶圓盒", "start": "3", "end": "二"}',
        '{"object": "晶圓盒", "start": "第四站", "end": "第三站"}',
         '{"object": "晶圓盒", "start": "二站", "end": "一站"}',

         '{"object": "晶圓盒", "start": "第五站", "end": "第六站"}',
        '{"object": "金元和", "start": "第七站", "end": "第二站"}',
        '{"object": "Wafer", "start": "三站", "end": "四站"}',
        '{"object": "晶圓盒", "start": "六站", "end": "五站"}',

        '{"object": "晶圓盒", "start": "八站", "end": "二站"}',
        '{"object": "晶圓盒", "start": "第五站", "end": "第八站"}',
        '{"object": "晶圓盒", "start": "第三站", "end": "第一站"}',
        '{"object": "晶圓盒", "start": "四站", "end": "第一站"}',

        '{"object":"Wafer", "start":"第四", "end":"第二"}',
        '{"object":"晶圓盒", "start":"七號站", "end":"五號站"}',
        '{"object": "金元和", "start": "第三站", "end": "第9站"}',
        '{"object": "晶圓盒", "start": "第三站", "end": "二站"}',

        '{"object": "晶圓盒", "start": "第四站", "end": "第三站"}',
        '{"object": "晶圓盒", "start": "第五", "end": "第六"}',
        '{"object": "晶圓盒", "start": "第九站", "end": "第八站"}',
        '{"object": "晶圓盒", "start": "一", "end": "六"}',

        '{"object": "晶圓盒", "start": "五", "end": "二"}',
        '{"object": "晶圓盒", "start": "一", "end": "三"}',
        '{"object": "晶圓盒", "start": "第七站", "end": "第六站"}',
        '{"object": "晶圓盒", "start": "四站", "end": "五站"}',

        '{"object": "Wafer", "start": "八站", "end": "三站"}',
        '{"object": "晶圓盒", "start": "二站", "end": "四站"}',
        '{"object": "金元和", "start": "五站", "end": "一站"}',
        '{"object": "Wafer", "start": "一站", "end": "二站"}',

        '{"object": "晶圓盒", "start": "三站", "end": "七站"}',
        '{"object": "晶圓盒", "start": "四站", "end": "八站"}',
        '{"object": "晶圓盒", "start": "二站", "end": "一站"}',
        '{"object": "晶圓盒", "start": "第一站", "end": "第三站"}',

        '{"object": "晶圓盒", "start": "第四站", "end": "第七站"}',
        '{"object": "晶圓盒", "start": "一站", "end": "二站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第一站"}',
        '{"object": "晶圓盒", "start": "第一站", "end": "第二站"}',

        '{"object": "晶圓盒", "start": "第二站", "end": "第一站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第三站"}',
        '{"object": "晶圓盒", "start": "第三站", "end": "第四站"}',
        '{"object": "晶圓盒", "start": "第二站", "end": "第一站"}',

    ]
    # 定義物件和站點
    objects = ["晶圓盒"]
    stations = [f"第{i}站" for i in ["一","二","三","四","五"]]

    # 自動擴增資料至 1000 筆
    while len(train_commands) < 1000:
        obj = random.choice(objects)
        start = random.choice(stations)
        end = random.choice(stations)
        while end == start:
            end = random.choice(stations)

        phrasing = [
            f"請將{obj}從{start}搬到{end}",
            f"{obj}從{start}移動至{end}",
            f"搬運任務：從{start}取{obj}送到{end}",
            f"請把{start}的{obj}搬到{end}",
            f"{start}拿到{end}",
            f"{start}到{end}",
            f"請將{obj}從{start}拿到{end}",
            f"從{start}搬到{end}",
            
        ]
        cmd = random.choice(phrasing)
        train_commands.append(cmd)
        train_outputs.append(json.dumps({"object": obj, "start": start, "end": end}, ensure_ascii=False))

    len(train_commands), len(train_outputs)
    # 將擴增後的資料儲存為 JSON 檔案
    output_data = [{"command": cmd, "output": out} for cmd, out in zip(train_commands, train_outputs)]

    output_path = "augmented_wafer_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

# 5. 主程式（關鍵修改：修復padding問題）
if __name__ == "__main__":
   
    # data_pre()
    with open("augmented_wafer_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # 拆分為兩個 list
    train_commands = [item["command"] for item in data]
    train_outputs = [item["output"] for item in data]
    print(train_commands[0])
    print(train_outputs[0])
    for i, out in enumerate(train_outputs):
        try:
            js = json.loads(out)
            if js["start"] == js["end"]:
                print(f"第{i}筆起點與終點相同：{train_commands[i]}")
        except:
            print(f"JSON 錯誤格式：{out}")
    val_commands = [
        "把第三站的金元和搬到第四站",
        "晶圓盒從第一站搬運第二站",
        "第二站搬運到第三站",
        "請把晶圓盒從第一站搬運到第四站"
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
        max_len=320 #96
    )
    dataloader = DataLoader(dataset, batch_size = 8, shuffle=True)

    # 訓練模型
    # trained_model = train_model(model, dataloader, epochs= 15)
    trained_model = train_model2(
        model, 
        dataloader, 
        epochs = 18, 
        patience=3, 
        eval_prompts=val_commands, 
        tokenizer=tokenizer
    )
    # 保存模型 (包含tokenizer配置)
    output_dir = "custom_wafer_llama"
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型訓練完成並已保存至 {output_dir}")

    # 測試指令集
    # test_commands = [
    #     "把第2站的金元和搬到第三站",
    #     "把第三站的金元和運到第1站",
    #     "Wafer由五站拿到第2站",
    #     "緊急任務：從三站搬晶圓盒到九站",
    #     "將二站晶圓盒轉運到三站",
    #     "第5站到第一站",
    #     "晶圓盒從第1站搬運第3站",
    #     "請把晶圓盒從第二站搬運到第三站"
    # ]
    test_commands = [
        "把第二站的金元和搬到第三站",
        "晶圓盒從第1站搬運第3站",
        "請把晶圓盒從第四站搬運到第三站",
        "第二站搬到第一站"
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
