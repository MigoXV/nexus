import os

import dotenv
import httpx
import openai

dotenv.load_dotenv()
client = openai.OpenAI(
    base_url=os.getenv("TEST_BASR_URL", "http://localhost:10002/v1"),
    api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
    http_client=httpx.Client(verify=False), 
)


respons = client.chat.completions.create(
    model="qwen3:8B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下你自己和你的公司。 /no_think"},
    ],
)

print(respons.choices[0].message.content)
