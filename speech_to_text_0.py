import speech_recognition as sr
import pyaudio
import re
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, AdamW
import pyttsx3

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
        
        prompt = f"解析指令：{command}\n輸出JSON："
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
        
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'labels': labels[0]
        }
    
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

    def parse(self, command):

        prompt = f"""
               <<站點提取強制規則>>
                1. 嚴格按指令中站點的「出現順序」提取：
                - 指令中第一個出現的站點 → start
                - 指令中第二個出現的站點 → end
                2. 禁止使用訓練數據中的默認值（如「第1站」）
                3. 站點名稱(start、end)必須與指令內容「完全一致」，也就是必須依照指令提取
                4. 站點不可捏照虛構，要檢查 start、end 輸出必須與指令內容站點「完全一致」
                <<錯誤案例修正>>
                指令：把第2站的金元和搬到第三站
                ❌ 錯誤輸出：start="第1站" → 違反規則1,2,3
                ✅ 正確輸出：start="第2站", end="第三站"

                <<當前任務>>
                指令：{command}
                <<輸出格式>>
                {{"object":"[物體]","start":"[起點站]","end":"[終點站]"}}
                """
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
         # 標準生成
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # 正確傳遞注意力掩碼
            max_new_tokens=50,
            num_beams=3,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            forced_bos_token_id=self.tokenizer.encode("{")[1]  # 強制JSON開頭
        )
        
        # 提取JSON部分
        full_output = self.tokenizer.decode(output_ids[0])
        json_str = full_output.split("輸出JSON：")[-1].split("}")[0] + "}"
        
        try:
            return json.loads(json_str)
        except:
            return self._fallback_parse(command)

    def _fallback_parse(self, command):
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
def speaker(string):
    engine = pyttsx3.init()
    engine.say(string)
    engine.runAndWait()
def chinese_to_int(chinese_num):
    chinese_num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }
    if chinese_num.isdigit():
        return int(chinese_num)
    if chinese_num in chinese_num_map:
        return chinese_num_map[chinese_num]
    if chinese_num.startswith('十'):
        if len(chinese_num) == 1:
            return 10
        else:
            return 10 + chinese_num_map.get(chinese_num[1], 0)
    if '十' in chinese_num:
        parts = chinese_num.split('十')
        left = parts[0] if parts[0] else '一'
        right = parts[1] if len(parts) > 1 else ''
        return chinese_num_map[left] * 10 + (chinese_num_map[right] if right else 0)
    return None
# 4. 動作執行器（保持不變）
class ActionExecutor:
    @staticmethod
    def navigate_to(station):
        robot_command=f"導航至 {station}" 
        print(robot_command)
        speaker(robot_command)
        num_matches = re.findall(r'([一二三四五六七八九十1234567890]+)', station)
        if num_matches:
            num_str = num_matches[0]
            num_int = chinese_to_int(num_str)
            print(f"第{num_int}站 (型別: {type(num_int)})")
        else:
            print("未找到數字")
    @staticmethod
    def pick_wafer(station):
        robot_command=f"在 {station} 抓取晶圓盒"
        print(robot_command)
        speaker(robot_command)

    @staticmethod
    def place_wafer(station):
        robot_command=f"在 {station} 放置晶圓盒"
        print(robot_command)
        speaker(robot_command)
        num_matches = re.findall(r'([一二三四五六七八九十1234567890])', station)
        if num_matches:
            num_str = num_matches[0]
            num_int = chinese_to_int(num_str)
            print(f"第{num_int}站 (型別: {type(num_int)})")
        else:
            print("未找到數字")
    @classmethod
    def execute_sequence(cls, parsed_cmd):
        actions = [
            ("navigate", parsed_cmd["start"]),
            ("pick", parsed_cmd["start"]),
            ("navigate", parsed_cmd["end"]),
            ("place", parsed_cmd["end"])
        ]
        response=f"好的，目前將進行 從 {parsed_cmd['start']} 到 {parsed_cmd['end']} 的晶圓盒搬運任務安排"
        speaker(response)
        print(f"\n開始執行 {parsed_cmd['object']} 搬運任務:")
        for action, param in actions:
            if action == "navigate":
                cls.navigate_to(param)
            elif action == "pick":
                cls.pick_wafer(param)
            elif action == "place":
                cls.place_wafer(param)
        print("任務完成!\n")

def speech_to_text():
    # 建立辨識器
    r = sr.Recognizer()
    
    # 設定麥克風
    mic = sr.Microphone()
    
    # 調整環境噪音
    with mic as source:# 打開麥克風
        print("正在調整環境噪音...")
        r.adjust_for_ambient_noise(source, duration=1) # 麥克風錄音
        print("請說話...")
        
        try:
            # 錄製語音
            # 最多等待 5 秒來偵測你開始說話。如果 6 秒內沒有任何語音輸入，會丟出 WaitTimeoutError 
            # 最多錄音 10 秒
            audio = r.listen(source, timeout=6, phrase_time_limit=10) 
            print("語音錄製完成，處理中...")
            
            # 使用Google語音辨識
            # 設定語言設定語音辨識使用繁體中文（台灣）
            text = r.recognize_google(audio, language="zh-TW")
            return text
        except sr.WaitTimeoutError:
            print("等待語音輸入超時")
            return None
        except sr.UnknownValueError:
            print("無法辨識語音內容")
            return None
        except sr.RequestError as e:
            print(f"語音辨識服務錯誤：{e}")
            return None

if __name__ == "__main__":
    # 檢查麥克風是否可用
    # print("可用的麥克風設備：")
    # print(sr.Microphone.list_microphone_names())
    
    # 執行語音轉文字
    result = speech_to_text()
    
    if result:
        print("辨識結果：", result)
        # 設定模型儲存路徑（建議不要加 './' 或 '/' 開頭，直接用資料夾名稱）
        MODEL_PATH = "custom_wafer_llama_0"

        # 確認路徑存在
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型路徑 {MODEL_PATH} 不存在")

        # 載入 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            pad_token_id=tokenizer.eos_token_id  # 明確指定pad_token_id
        )
        parser = CommandParser(model, tokenizer)
        executor = ActionExecutor()
        test_commands=result
        print(f"\n測試 {(test_commands)}") 
        try:
            # 使用模型解析指令
            parsed_result = parser.parse(test_commands)
            print(f"解析結果: {parsed_result}")
            # speaker(f"解析結果: {parsed_result}")
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
    else:
        print("未獲取到有效的語音輸入")
