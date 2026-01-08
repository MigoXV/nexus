import os

import dotenv
import httpx
import openai

dotenv.load_dotenv()
client = openai.OpenAI(
    base_url=os.getenv("TEST_BASR_URL", "http://localhost:10002/v1"),
    api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
    http_client=httpx.Client(verify=False),  # 🔴 关键：关闭证书校验
)


stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下你自己。"},
    ],
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    content = getattr(delta, "content", None)
    if content:
        print(content, end="", flush=True)
print()
