"""双流式 TTS gRPC 客户端 Demo（多协程并发）"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import grpc
import grpc.aio
import numpy as np
import soundfile as sf
from tqdm import tqdm

# from openchat_service.protos.tts import tts_pb2, tts_pb2_grpc
from nexus.protos.tts import tts_pb2, tts_pb2_grpc

# ---------------------------------------------------------------------------
# 配置（按需修改）
# ---------------------------------------------------------------------------

# HOST = "localhost"
# PORT = 50004
HOST = "39.106.1.132"
PORT = 30036
CONCURRENCY = 1  # 并发协程数

# 预置音色：服务端约定好的 preset_voice_id；设为 "" 则走参考音频克隆模式。
PRESET_VOICE_ID = ""

REF_AUDIO_PATH = "model-bin/voices/speaker1_b_cn_16k.wav"
REF_TEXT = "充满了营养的一个一个大的作品，使得我们能够也跟着成长。"

TEXT_FILE = "data-bin/demo_text.txt"  # 待合成文本文件
LANGUAGE = "auto"  # 语言
DECODER_CHUNK_SIZE = 50  # 声码器解码块大小（token 数）
TEXT_CHUNK_SIZE = 1  # 每帧发送字符数（模拟流式输入粒度）
OUTPUT_DIR = Path("data-bin/outputs/duplex")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def pcm_s16le_to_float32(data: bytes) -> np.ndarray:
    """将 PCM S16LE 字节流转换为 float32 波形 [-1, 1]。"""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def pcm_s16le_duration_seconds(data: bytes, sample_rate: int) -> float:
    """根据 PCM S16LE 字节数估算单声道音频时长（秒）。"""
    if sample_rate <= 0:
        return 0.0
    return len(data) / 2 / sample_rate


def load_ref_audio_as_pcm(path: str) -> tuple[bytes, int]:
    """加载参考音频文件，返回 (pcm_s16le_bytes, sample_rate)。"""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 转单声道
    # float32 -> int16
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes(), sr


# ---------------------------------------------------------------------------
# 构建 gRPC 请求帧
# ---------------------------------------------------------------------------


def make_config_frame(
    preset_voice_id: str | None,
    ref_audio_bytes: bytes | None,
    ref_sample_rate: int | None,
) -> tts_pb2.DuplexSynthesizeRequest:
    req = tts_pb2.DuplexSynthesizeRequest()
    cfg = req.config
    if preset_voice_id:
        cfg.voice.preset_voice_id = preset_voice_id
    else:
        cfg.voice.reference_voice.pcm_s16le = ref_audio_bytes
        cfg.voice.reference_voice.sample_rate = ref_sample_rate
        cfg.voice.reference_voice.ref_text = REF_TEXT
    cfg.language = LANGUAGE
    cfg.decoder_chunk_size = DECODER_CHUNK_SIZE
    return req


def make_text_frame(chunk: str) -> tts_pb2.DuplexSynthesizeRequest:
    """构造文本块帧。"""
    return tts_pb2.DuplexSynthesizeRequest(text_chunk=chunk)


# ---------------------------------------------------------------------------
# 单协程任务
# ---------------------------------------------------------------------------


async def duplex_synthesize(
    stub: tts_pb2_grpc.TtsServiceStub,
    coroutine_id: int,
    text: str,
    config_frame: tts_pb2.DuplexSynthesizeRequest,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"duplex_coroutine_{coroutine_id:02d}.wav"

    print(f"[{coroutine_id}] 开始合成，文本长度={len(text)}，输出={output_path}")
    t0 = time.monotonic()
    first_packet_latency: float | None = None

    async def request_iter():
        yield config_frame
        for i in range(0, len(text), TEXT_CHUNK_SIZE):
            yield make_text_frame(text[i : i + TEXT_CHUNK_SIZE])
            await asyncio.sleep(0)  # 让出事件循环，使多协程可以交替发送

    pcm_chunks: list[bytes] = []
    sample_rate = 24000

    with tqdm(
        desc=f"coroutine {coroutine_id}",
        unit="s",
        unit_scale=True,
        unit_divisor=1,
    ) as pbar:
        async for audio_chunk in stub.DuplexSynthesize(request_iter()):
            if first_packet_latency is None:
                first_packet_latency = time.monotonic() - t0
                print(f"[{coroutine_id}] 首包延迟={first_packet_latency:.3f}s")
            pcm_chunks.append(audio_chunk.pcm_s16le)
            sample_rate = audio_chunk.sample_rate
            pbar.update(
                pcm_s16le_duration_seconds(
                    audio_chunk.pcm_s16le, audio_chunk.sample_rate
                )
            )

    if pcm_chunks:
        waveform = pcm_s16le_to_float32(b"".join(pcm_chunks))
        sf.write(str(output_path), waveform, samplerate=sample_rate)
        elapsed = time.monotonic() - t0
        print(
            f"[{coroutine_id}] 完成：{len(pcm_chunks)} 块，"
            f"首包延迟={first_packet_latency:.3f}s，"
            f"时长={len(waveform)/sample_rate:.2f}s，"
            f"耗时={elapsed:.2f}s，已保存至 {output_path}"
        )
    else:
        print(f"[{coroutine_id}] 警告：未收到任何音频数据")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


async def main() -> None:
    preset_voice_id = PRESET_VOICE_ID or None

    # 加载参考音频（仅音色克隆时需要）
    ref_audio_bytes: bytes | None = None
    ref_sample_rate: int | None = None
    if not preset_voice_id:
        print(f"加载参考音频: {REF_AUDIO_PATH}")
        ref_audio_bytes, ref_sample_rate = load_ref_audio_as_pcm(REF_AUDIO_PATH)
    else:
        print(f"使用预置音色: {preset_voice_id}")

    # 读取待合成文本
    text = Path(TEXT_FILE).read_text(encoding="utf-8").strip()
    print(f"合成文本（前80字）: {text[:80]}")
    print(f"并发协程数: {CONCURRENCY}")

    async with grpc.aio.insecure_channel(f"{HOST}:{PORT}") as channel:
        stub = tts_pb2_grpc.TtsServiceStub(channel)
        await asyncio.gather(
            *[
                duplex_synthesize(
                    stub=stub,
                    coroutine_id=i,
                    text=text,
                    # 每个协程独立构造配置帧（避免共享同一个 protobuf 对象）
                    config_frame=make_config_frame(
                        preset_voice_id, ref_audio_bytes, ref_sample_rate
                    ),
                )
                for i in range(CONCURRENCY)
            ]
        )

    print("全部完成。")


if __name__ == "__main__":
    asyncio.run(main())
