from openai import OpenAI
import os

client = OpenAI(api_key="dummy_api_key",base_url="http://localhost:8000/v1")

# 需要识别的音频文件路径（支持 wav / mp3 / m4a 等常见格式）
audio_file_path = "data-bin/huaqiang/403369728_nb2-1-30280_left_16k.wav"

with open(audio_file_path, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-transcribe",
        # 可选参数
        # language="zh",        # 指定语言（如中文）
        # prompt="这是一次会议录音",  # 给模型的上下文提示
        # response_format="text" # 返回纯文本
    )

print("识别结果：")
print(transcription.text)
