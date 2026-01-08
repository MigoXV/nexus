from __future__ import annotations

import asyncio
import base64
import io
import os
import ssl

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from pydub import AudioSegment
from tqdm import trange

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 3200  # 每个块的字节大小

AUDIO_FILE_PATH = "data-bin/huaqiang/403369728_nb2-1-30280_left_16k.wav"


def load_audio_as_pcm16(file_path: str) -> bytes:
    """读取音频文件并转换为 PCM16 格式"""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    print(
        f"Loaded audio: {audio.frame_rate=} {audio.channels=} {audio.sample_width=} {audio.frame_width=}"
    )
    print(f"Duration: {len(audio) / 1000:.2f}s")

    # 转换为 16kHz 单声道 PCM16
    pcm_audio = (
        audio.set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(2)
        .raw_data
    )
    return pcm_audio


async def send_audio_from_file(connection: AsyncRealtimeConnection, audio_data: bytes):
    """将音频数据切分成块并发送"""
    total_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Total audio size: {len(audio_data)} bytes, chunks: {total_chunks}")

    # 🔴 先发送 response.create 启动流式转录
    await connection.send({"type": "response.create", "response": {}})

    for i in trange(0, len(audio_data), CHUNK_SIZE):
        chunk = audio_data[i : i + CHUNK_SIZE]
        await connection.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
        )
        # 小延迟模拟实时音频流
        await asyncio.sleep(0.01)

    # 发送结束标记
    print("\nAudio sent, committing buffer...")
    await connection.send({"type": "input_audio_buffer.commit"})


async def receive_responses(
    connection: AsyncRealtimeConnection, done_event: asyncio.Event
):
    """接收并实时打印服务器返回的结果"""
    async for event in connection:
        event_type = event.type

        if event_type == "session.created":
            print(f"[Session] Created: {event.session.id}")
        elif event_type == "response.created":
            print(f"[Response] Created: {event.response.id}")
        elif event_type == "response.audio_transcript.delta":
            # 实时打印增量文本，不换行
            print(f"{event.delta}")
        elif event_type == "response.audio_transcript.done":
            print()  # 换行
            print(f"[Transcript Done] {event.transcript}")
        elif event_type == "response.text.delta":
            print(f"{event.delta}")
        elif event_type == "response.text.done":
            print()
            print(f"[Text Done] {event.text}")
        elif event_type == "response.done":
            print(f"[Response] Done!")
            done_event.set()
            break
        elif event_type == "error":
            print(f"[Error] {event}")
            done_event.set()
            break
        else:
            # 打印其他事件
            print(f"[Event] {event_type}")


async def main():
    # 读取音频文件
    print(f"Loading audio file: {AUDIO_FILE_PATH}")
    audio_data = load_audio_as_pcm16(AUDIO_FILE_PATH)

    # 创建禁用证书验证的 SSL 上下文
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # 创建 OpenAI 客户端
    client = AsyncOpenAI(
        base_url=os.getenv("TEST_BASR_URL", "http://localhost:10002/v1"),
        api_key=os.getenv("TEST_API_KEY", "dummy_api_key"),
    )

    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview",
        websocket_connection_options={
            "ssl": ssl_context
        },  # 🔴 关键：传递 SSL 上下文禁用证书校验
    ) as connection:
        print("Connected to realtime API")

        # 完成事件
        done_event = asyncio.Event()

        # 启动接收任务（在后台持续运行，实时打印结果）
        receive_task = asyncio.create_task(receive_responses(connection, done_event))

        # 发送音频（会实时收到服务端返回的流式结果）
        await send_audio_from_file(connection, audio_data)

        # 等待接收完成
        await done_event.wait()
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
