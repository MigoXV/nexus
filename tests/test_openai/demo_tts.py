"""TTS Demo - 使用 OpenAI SDK 调用 TTS 并保存为 WAV"""

from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:10003/v1",
    api_key="no-key",
)

output_path = Path("data-bin/speech.wav")

with client.audio.speech.with_streaming_response.create(
    model="fnlp/MOSS-TTSD-v0.5",
    voice="fnlp/MOSS-TTSD-v0.5:anna",
    input="你好，这是一段测试语音。",
    response_format="wav",
) as response:
    response.stream_to_file(output_path)

print(f"已保存到 {output_path}")
