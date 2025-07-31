import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from torch.optim import AdamW
import numpy as np
import os
# from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import AutoTokenizer, LlamaForCausalLM
import torch
print(torch.__version__)
# 1. 自定義資料集 (修正標籤對齊問題)
class WaferCommandDataset(Dataset):
    def __init__(self, commands, labels, tokenizer, label2id, max_len=128):
        self.tokenizer = tokenizer
        self.commands = commands
        self.labels = labels
        self.label2id = label2id
        self.max_len = max_len
        
    def __len__(self):
        return len(self.commands)
    
    def __getitem__(self, idx):
        command = self.commands[idx]
        raw_labels = self.labels[idx]
        
        encoding = self.tokenizer(
            command,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        labels_encoded = torch.full((self.max_len,), -100, dtype=torch.long)
        
        char_pos = 0
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if token.startswith("##"):
                token = token[2:]
            if char_pos < len(raw_labels):
                label = raw_labels[char_pos]
                labels_encoded[i] = self.label2id.get(label, 0)
            char_pos += len(token)
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': labels_encoded
        }

# 2. 訓練函數
def train_model(model, dataloader, epochs=3, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return model

# 3. 解析器類別
class CommandParser:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.num_map = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10
        }
    
    def parse(self, command):
        # 使用模型預測
        inputs = self.tokenizer(command, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 解析預測結果
        predictions = torch.argmax(outputs.logits, dim=-1)[0].numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # 提取關鍵資訊
        object_text = ""
        collecting_object = False
        current_entity = ""
        current_text = ""
        start_station = None
        end_station = None
        
        for token, pred_id in zip(tokens, predictions):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            label = self.id2label.get(pred_id, 'O')
            
            # 處理子詞
            if token.startswith("##"):
                token = token[2:]
            
            # 合併連續OBJECT標籤 (關鍵修改)
            if label == 'OBJECT':
                if not collecting_object:
                    object_text = token
                    collecting_object = True
                else:
                    object_text += token
            else:
                # 遇到非OBJECT標籤時，如果之前正在收集OBJECT，則停止收集
                collecting_object = False
            
            # 收集站點/動詞資訊
            if label in ['START', 'END']:
                current_entity = label
                current_text = token
            elif current_entity and label == current_entity:
                current_text += token
            elif current_entity:
                # 實體結束，保存結果
                if current_entity == 'START':
                    start_station = self._convert_num(current_text)
                elif current_entity == 'END':
                    end_station = self._convert_num(current_text)
                current_entity = ""
                current_text = ""
        
        # 備用解析策略
        if not start_station or not end_station:
            num_matches = re.findall(r'([一二三四五六七八九十1234567890]+)站', command)
            if len(num_matches) >= 2:
                start_station = self._convert_num(num_matches[0])
                end_station = self._convert_num(num_matches[1])
        
        # 設置默認物體名稱
        object_type = object_text if object_text else "晶圓盒"
        
        # 驗證結果
        if not start_station or not end_station:
            raise ValueError(f"無法解析指令: {command}")
        
        return {
            "object": object_type,
            "start": f"第{start_station}站",
            "end": f"第{end_station}站"
        }
    
    def _convert_num(self, num_str):
        return self.num_map.get(num_str, num_str)

# 4. 動作執行器
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

# 5. 主程式
if __name__ == "__main__":
    # 定義標籤映射
    LABEL_MAP = {'O': 0, 'OBJECT': 1, 'START': 2, 'END': 3, 'VERB': 4}
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
    
    # # 訓練資料
    # train_commands = [
    #     "請把晶圓盒從第二站搬到第三站",
    #     "金元和從第1站搬運第3站",
    #     "將Wafer由五站移動至七站",
    #     "緊急任務：從三站搬晶圓盒到九站",
    #     "十站金源和轉運到四站",
    #     "第6站到第8站",
    #     "把第三站的晶圓盒搬到第1站",
    #     "五站拿到第1站"
    # ]
    
    train_commands = [
        "請把晶圓盒從第二站搬到第三站",
        "第6站到第8站",
        "把第三站的晶圓盒搬到第1站"
    ]
    train_labels = [
        # "請把晶圓盒從第二站搬到第三站"
        ['O', 'O', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'START', 'START', 'START', 'VERB', 'END', 'END', 'END'],
        # "第6站到第8站"
        ['START', 'START', 'START', 'VERB', 'END', 'END', 'END'],
        # "把第三站的晶圓盒搬到第1站"
        ['O', 'START', 'START', 'START', 'O', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'VERB', 'END', 'END', 'END']
    ]

    # # 字符級標註 (每個字符對應一個標籤)
    # train_labels = [
    #     # "請把晶圓盒從第二站搬到第三站"
    #     ['O', 'O', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'START', 'START', 'START', 'VERB', 'END', 'END', 'END'],
        
    #     # "晶圓盒從第1站搬運第3站"
    #     ['OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'START', 'START', 'START', 'VERB', 'VERB', 'END', 'END', 'END'],
        
    #     # "將Wafer由五站移動至七站"
    #     ['O', 'OBJECT', 'OBJECT', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'START', 'START', 'VERB', 'VERB', 'VERB', 'END', 'END'],
        
    #     # "緊急任務：從三站搬晶圓盒到九站"
    #     ['O', 'O', 'O', 'O', 'O', 'O', 'VERB', 'START', 'START', 'VERB', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'END', 'END'],
        
    #     # "十站晶圓盒轉運到四站"
    #     ['START', 'START', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'VERB', 'VERB', 'END', 'END'],
        
    #     # "第6站到第8站"
    #     ['START', 'START', 'START', 'VERB', 'END', 'END', 'END'],
        
    #     # "把第三站的晶圓盒搬到第1站"
    #     ['O', 'START', 'START', 'START', 'O', 'OBJECT', 'OBJECT', 'OBJECT', 'VERB', 'VERB', 'END', 'END', 'END'],
    #     # "五站拿到第1站"
    #     ['START', 'START','VERB', 'VERB','END', 'END', 'END']
    # ]
    # 原始標註資料範例
    train_commands = ["請把晶圓盒從第二站搬到第三站"]
    train_labels = [['O','O','OBJECT','OBJECT','OBJECT','VERB','START','START','START','VERB','END','END','END']]

    # 轉換為生成任務格式
    train_outputs = [
        '{"object": "晶圓盒", "start": "第二站", "end": "第三站"}'
    ]
    # 以 Qwen2.5-7B-Chat 為例（可根據硬體資源選擇 1.8B, 4B, 7B 等不同規模）
    # MODEL_NAME = "Qwen/Qwen2.5-7B-Chat"

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForTokenClassification.from_pretrained(
    #     MODEL_NAME,
    #     num_labels=len(LABEL_MAP),
    #     id2label=ID2LABEL,
    #     label2id=LABEL_MAP
    # )
  
    # 替換為 LLaMA 模型 (新增)
    tokenizer = AutoTokenizer.from_pretrained("p208p2002/llama-traditional-chinese-120M")
    model = LlamaForCausalLM.from_pretrained(
        "p208p2002/llama-traditional-chinese-120M",
        pad_token_id=tokenizer.eos_token_id  # 解決填充問題
    )

    # # 初始化tokenizer和模型
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # model = BertForTokenClassification.from_pretrained(
    #     'bert-base-chinese',
    #     num_labels=len(LABEL_MAP),
    #     id2label=ID2LABEL,
    #     label2id=LABEL_MAP
    # )
    
    # 創建資料集和數據加載器
    dataset = WaferCommandDataset(
        commands=train_commands,
        labels=train_labels,
        tokenizer=tokenizer,
        label2id=LABEL_MAP,
        max_len=128
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 訓練模型
    trained_model = train_model(model, dataloader, epochs=5)
    trained_model.save_pretrained("./custom_wafer_bert")
    tokenizer.save_pretrained("./custom_wafer_bert")
    print("模型訓練完成並已保存！")
    
    # 設定模型儲存路徑（建議不要加 './' 或 '/' 開頭，直接用資料夾名稱）
    MODEL_PATH = "custom_wafer_bert"

    # 確認路徑存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路徑 {MODEL_PATH} 不存在")

    # 載入 tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)



    # 測試指令集
    test_commands = [
        "把第2站的金源合搬到第三站",  # 基本指令
        "把第三站的金元合運到第1站",  # 複雜指令
        "Wafer由五站拿到第2站",  # 英文物體名稱
        "緊急任務：從三站搬晶圓盒到九站",  # 包含標點符號
        "將二站晶圓盒轉運到三站",  # 語序變化
        "第5站到第一站",  # 簡化指令
        "晶圓盒從第1站搬運第3站",  # 數字格式混合
        "請把晶圓盒從第二站搬運到第三站",  # 禮貌用語
    ]
    
    print("\n=== 開始全面測試 ===")
    parser = CommandParser(trained_model, tokenizer, ID2LABEL)
    executor = ActionExecutor()
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n測試 {i}/{len(test_commands)}")
        print(f"輸入指令: 「{cmd}」")
        
        try:
            # 使用模型解析指令
            parsed_result = parser.parse(cmd)
            print(f"解析結果: {parsed_result}")
            
            # 執行動作序列
            
            executor.execute_sequence(parsed_result)
            
        except ValueError as e:
            print(f"解析錯誤: {str(e)}")
            print("嘗試使用備用解析策略...")
            
            # 備用解析策略
            try:
                # 提取所有站點
                num_matches = re.findall(r'([\w\d]+站)', cmd)
                if len(num_matches) >= 2:
                    start_station = num_matches[-2]  # 倒數第二個站點
                    end_station = num_matches[-1]    # 最後一個站點
                    
                    # 提取物體名稱
                    obj_match = re.search(r'(晶圓盒|晶圓載具|Wafer)', cmd)
                    object_type = obj_match.group(1) if obj_match else "晶圓盒"
                    
                    parsed_result = {
                        "object": object_type,
                        "start": start_station,
                        "end": end_station
                    }
                    
                    print(f"備用解析結果: {parsed_result}")
                    executor.execute_sequence(parsed_result)
                else:
                    print("備用解析失敗：站點數量不足")
                    
            except Exception as e:
                print(f"備用解析也失敗: {str(e)}")
    
    print("\n=== 全面測試完成 ===")

    # # 測試指令
    # test_command = "把第三站的晶圓盒搬到第1站"
    
    # print(f"\n輸入指令: 「{test_command}」")
    
    # # 使用模型解析指令
    # parser = CommandParser(trained_model, tokenizer, ID2LABEL)
    # parsed_result = parser.parse(test_command)
    # print(f"解析結果: {parsed_result}")
    
    # # 執行動作序列
    # executor = ActionExecutor()
    # executor.execute_sequence(parsed_result)
