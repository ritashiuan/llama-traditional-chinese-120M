# import pyttsx3

# def speaker(string):
#     engine = pyttsx3.init()
#     engine.say(string)
#     engine.runAndWait()
# import speech_recognition as sr

# def speech_to_text_whisper():
#     r = sr.Recognizer()
#     mic = sr.Microphone()
#     with mic as source:
#         print("正在調整環境噪音...")
#         r.adjust_for_ambient_noise(source, duration=1)
#         print("請說話...")
#         try:
#             audio = r.listen(source, timeout=5, phrase_time_limit=10)
#             print("語音錄製完成，處理中...")
#             # 使用 Whisper 辨識
#             text = r.recognize_whisper(audio, language="zh")
#             return text
#         except sr.WaitTimeoutError:
#             print("等待語音輸入超時")
#             return None
#         except sr.UnknownValueError:
#             print("無法辨識語音內容")
#             return None
#         except sr.RequestError as e:
#             print(f"語音辨識服務錯誤：{e}")
#             return None
# if __name__ == "__main__":

#     # 執行語音轉文字
#     result = speech_to_text_whisper()
    
#     if result:
#         print("辨識結果：", result)
import speech_recognition as sr

def record_to_wav(wav_path="temp.wav"):
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("正在調整環境噪音...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("請說話...")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
        print("語音錄製完成，儲存音檔...")
        with open(wav_path, "wb") as f:
            f.write(audio.get_wav_data())
    return wav_path
from faster_whisper import WhisperModel

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
if __name__ == "__main__":
    wav_file = record_to_wav()
    result = transcribe_with_faster_whisper(wav_file)
    if result:
        print("辨識結果：", result)
