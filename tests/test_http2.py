# # 使用OpenAI SDK测试兼容性
# import openai

# client = openai.OpenAI(
#     base_url="http://127.0.0.1:8000/v1", 
#     api_key="dummy-key"  
# )

# transcription = client.audio.transcriptions.create(
#     file=open("speaker1_a_cn_16k.wav", "rb"),
#     model="ux_speech_grpc_proxy"
# )
# print(transcription.text)

import requests

url = "http://127.0.0.1:8000/v1/audio/transcriptions"

files = { "file": ("src\grpc_to_http2\speaker1_a_cn_16k.wav", open("src\grpc_to_http2\speaker1_a_cn_16k.wav", "rb")) }
payload = { "model": "ux_speech_grpc_proxy" }
headers = {"Authorization": "Bearer <token>"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.text)


#我是否可以在运行它的时候加入命令行，或许可以起到覆盖的作用？