import speech_recognition as sr
import pyaudio
import re
import os
import time
import json
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, LlamaForCausalLM, AdamW
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from gtts import gTTS
import os
import pyttsx3
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk 
import customtkinter as ctk
import sys


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
            if (json.loads(json_str) == self._fallback_parse(command)):
                return json.loads(json_str)
            else:
                print("啟用備用方案~")
                return self._fallback_parse(command) #備用方案
        else:
            print("啟用備用方案~")
            return self._fallback_parse(command) #備用方案
            # raise ValueError(f"無法解析指令: {command}")

    def _fallback_parse(self, command):
        num_matches = re.findall(r'([一二三四五六七八九十1234567890]+)站', command)
        object_match = re.search(r'(晶圓盒|金元和|Wafer| 金圓盒)', command)
        object_type = object_match.group(1) if object_match else "晶圓盒"
        
        if len(num_matches) >= 2:
            return {
                "object": object_type,
                "start": f"第{num_matches[0]}站",
                "end": f"第{num_matches[1]}站"
            }
        raise ValueError(f"無法解析指令: {command}")
# def speaker(string):
#     engine = pyttsx3.init()
#     engine.say(string)
#     engine.runAndWait()


def speaker(text):
    tts = gTTS(text=text, lang='zh-tw')  # 設定中文繁體
    filename = "temp_speech.mp3"
    tts.save(filename)
    os.system('ffplay -nodisp -autoexit -af "atempo=1.5" temp_speech.mp3')


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
        add_message(f"導航至 {station}", is_user=False)
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
        add_message(f"在 {station} 抓取晶圓盒", is_user=False)
    @staticmethod
    def place_wafer(station):
        robot_command=f"在 {station} 放置晶圓盒"
        add_message(f"在 {station} 放置晶圓盒", is_user=False)
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
        add_message(response, is_user=False)
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
def record_to_wav(wav_path="temp.wav"):
    
    r = sr.Recognizer()
    # mic = sr.Microphone()
    mic = sr.Microphone(device_index=8)  # 換成對應的 index

    with mic as source:
        print("正在調整環境噪音...")
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            # 錄製語音
            # 最多等待 5 秒來偵測你開始說話。如果 6 秒內沒有任何語音輸入，會丟出 WaitTimeoutError 
            # 最多錄音 10 秒
            speaker("正在調整環境噪音，請給予指令")
            print("請說話...")
            # time.sleep(2)  # 等1秒讓播放穩定後再開始錄音
            audio = r.listen(source, timeout=15, phrase_time_limit=20) 
            print("語音錄製完成，處理中...")
            with open(wav_path, "wb") as f:
                f.write(audio.get_wav_data())
        except sr.WaitTimeoutError  :
            print("等待語音輸入超時")
            return None
        except sr.UnknownValueError:
            print("無法辨識語音內容")
            speaker("無法辨識語音內容")
            return None
        except sr.RequestError as e:
            print(f"語音辨識服務錯誤：{e}")
            return None
    return wav_path
def transcribe_with_faster_whisper(wav_path, model_size="small"):
    # device="cuda" 代表用 GPU，如果沒有 GPU 可以改為 "cpu"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, language="zh")
    print(f"偵測語言：{info.language}，信心度：{info.language_probability}")
    result = ""
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        result += segment.text
    return result
def is_wafer_command(text):
    # 可根據實際需求擴充關鍵詞
    keywords = ["晶圓盒", "金元和", "Wafer", "搬", "運", "站","到"]
    return any(kw in text for kw in keywords) and "站" in text
# def is_Finish_command(text):
#     keywords = ["結束","掰掰","停止","停"]
#     return any(kw in text for kw in keywords) 

from datetime import datetime
from PIL import Image, ImageTk
def add_message(text, is_user=True):
    timestamp = datetime.now().strftime("%p %I:%M").replace("AM", "上午").replace("PM", "下午")
    bubble_color = "#FFF2C2" if is_user else "#D9EAFD"
    icon_path = "user.png" if is_user else "robot.png"
    icon_image = ctk.CTkImage(Image.open(icon_path), size=(30, 30))

    # 外層對齊容器（控制整體靠左或靠右）
    msg_container = ctk.CTkFrame(chat_frame, fg_color="transparent")
    msg_container.pack(anchor="e" if is_user else "w", pady=4, padx=10, fill="x")

    # 內層訊息框（放 icon 和 bubble）
    msg_frame = ctk.CTkFrame(msg_container, fg_color="transparent")
    msg_frame.pack(side="right" if is_user else "left")

    # 氣泡
    bubble = ctk.CTkLabel(
        msg_frame,
        text=text,
        font=("Microsoft JhengHei", 14),
        fg_color=bubble_color,
        text_color="#2c3e50",
        corner_radius=16,
        anchor="w",
        justify="left",
        width=460,
        wraplength=440,
    )

    # icon
    icon = ctk.CTkLabel(msg_frame, image=icon_image, text="", width=30, height=30)
    icon.image = icon_image

    # 時間戳
    time_label = ctk.CTkLabel(
        msg_container,
        text=timestamp,
        font=("Microsoft JhengHei", 10),
        fg_color="transparent",
        text_color="#95a5a6"
    )

    # 元素排列順序
    if is_user:
        bubble.pack(side="right", padx=(10, 6))
        icon.pack(side="right")
        time_label.pack(anchor="e", padx=40)
    else:
        icon.pack(side="left")
        bubble.pack(side="left", padx=(6, 10))
        time_label.pack(anchor="w", padx=40)

def on_record():
    def task():
        record_btn.configure(state="disabled", text="錄音中...")

        wav_file = record_to_wav()
        if wav_file:
            result = transcribe_with_faster_whisper(wav_file)
            add_message("你說了：" + result, is_user=True)

            if is_wafer_command(result):
                try:
                    parsed = parser.parse(result)
                    add_message("解析結果：" + str(parsed), is_user=False)
                    executor.execute_sequence(parsed)
                except Exception as e:
                    add_message("解析失敗：" + str(e), is_user=False)
                    speaker("不太清楚您的指令，請再說明一次")
            else:
                add_message("非晶圓盒搬運指令", is_user=False)
                speaker("未獲得有效的晶圓盒搬運指令，請說明搬運任務的請點站與終點站")
        else:
            speaker("未獲取到有效的語音輸入，請再說明一次")
            add_message("未獲取到有效語音", is_user=False)

        record_btn.configure(state="normal", text="開始錄音")

    # Thread(target=task).start()
    # daemon=True 主程式結束時，子執行緒會自動中止。
    Thread(target=task, daemon=True).start()
def on_closing():
    root.destroy()  # 關閉視窗
    sys.exit()      # 強制結束所有執行緒與主程式
if __name__ == "__main__":
    mic_list = sr.Microphone.list_microphone_names()

#   列出所有裝置及其 index
    for index, name in enumerate(mic_list):
        print(f"Index {index}: {name}")
    MODEL_PATH = "custom_wafer_llama"

    # 確認路徑存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路徑 {MODEL_PATH} 不存在")

    # 載入 tokenizer 輸入的句子切分成一個個「token」（通常是字、詞或子詞），再轉換成對應的數字 ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        pad_token_id=tokenizer.eos_token_id  # 讓所有輸入長度一樣，補上<eos>，代表句子的結束
    )

    parser = CommandParser(model, tokenizer)
    executor = ActionExecutor()
    
   # 主視窗設計
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.geometry("700x600")
    root.title("語音指令智能助手")
    # 標題
  
    # 載入圖片
    # robot_img = Image.open("robot_img.png")

    # 建立 CTkImage 物件，支援高 DPI
    # robot_ctk_image = ctk.CTkImage(light_image=robot_img, size=(50, 50))
    title = ctk.CTkLabel(root, text="語音指令智能助手", font=("Microsoft JhengHei", 24, "bold"))
    title.pack(pady=(16, 6))
    
    # 分隔線
    separator = ctk.CTkFrame(root, height=2, fg_color="#d0d7e5", corner_radius=1)
    separator.pack(fill="x", padx=30, pady=4)

    # 麥克風圖示
    mic_img = Image.open("mic_icon.png")
    mic_photo = ctk.CTkImage(light_image=mic_img, size=(35, 35))
    # 錄音按鈕（圓角）
    record_btn = ctk.CTkButton(
        root, image=mic_photo, text="開始錄音", command=on_record,
        font=("Microsoft JhengHei", 20, "bold"),
        fg_color="#6cace4", hover_color="#4a90e2", text_color="white",
        corner_radius=24, width=220, height=50, compound="left"
    )
    record_btn.image = mic_photo
    record_btn.pack(pady=30)
    # 新增一個滾動框架作為聊天區域
    chat_frame = ctk.CTkScrollableFrame(root, width=600, height=300, fg_color="transparent")
    chat_frame.pack(padx=30, pady=10)

    # 儲存訊息數量
    messages = []
    # cancel_img = Image.open("cancel_icon.png")
    # cancel_photo = ctk.CTkImage(light_image=cancel_img, size=(50, 50))
    # # 結束按鈕（圓角）
    # exit_btn = ctk.CTkButton(
    #     root, image=cancel_photo, text="結束", command=root.destroy,
    #     font=("Microsoft JhengHei", 14, "bold"),
    #     fg_color="#e57373", hover_color="#c0392b", text_color="white",
    #     corner_radius=18, width=120, height=40, compound="left"
    # )
    # exit_btn.pack(pady=28)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    

    



