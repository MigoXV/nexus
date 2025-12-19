from openai import OpenAI
import httpx

client = OpenAI(
    api_key="dummy_api_key",
    base_url="https://localhost:8000/v1",
    http_client=httpx.Client(
        verify=False  # 🔴 关键：关闭证书校验
    ),
)

audio_file_path = "data-bin/huaqiang/403369728_nb2-1-30280_left_16k.wav"

with open(audio_file_path, "rb") as audio_file:
    stream = client.audio.transcriptions.create(
        file=audio_file,
        model="gpt-4o-transcribe",
        stream=True,          # 👈 关键
        language="zh",
    )

    print("流式识别结果：")
    for event in stream:
        # 兼容 OpenAI / vLLM / FastAPI 实现
        if hasattr(event, "text") and event.text:
            print(event.text, end="", flush=True)
