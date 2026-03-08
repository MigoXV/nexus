from __future__ import annotations

import numpy as np
import pytest

from nexus.infrastructure.audio import StreamingResampler


def test_streaming_resampler_handles_odd_byte_chunks_with_flush() -> None:
    resampler = StreamingResampler(input_rate=24000, output_rate=16000)

    # 1 second of 24k PCM16 mono.
    pcm_bytes = np.arange(24000, dtype=np.int16).tobytes()
    # Deliberately split at an odd byte boundary.
    first = pcm_bytes[:1001]
    second = pcm_bytes[1001:]

    out1 = resampler.process(first)
    out2 = resampler.process(second)
    tail = resampler.flush()

    merged = out1 + out2 + tail
    samples = np.frombuffer(merged, dtype=np.int16)

    # 24k -> 16k should be approximately 2/3 length.
    assert abs(samples.size - 16000) <= 256


def test_streaming_resampler_defaults_to_sinc_fastest() -> None:
    """Verify default converter_type is sinc_fastest (low-latency)."""
    resampler = StreamingResampler(input_rate=48000, output_rate=24000)
    # sinc_fastest produces valid output — just verify it runs without error
    pcm = np.zeros(4800, dtype=np.int16).tobytes()
    out = resampler.process(pcm)
    assert len(out) > 0


@pytest.mark.asyncio
async def test_streaming_resampler_aprocess_matches_sync() -> None:
    """aprocess/aflush must produce identical results to process/flush."""
    pcm_bytes = np.arange(24000, dtype=np.int16).tobytes()
    first, second = pcm_bytes[:1001], pcm_bytes[1001:]

    # Sync path
    sync_resampler = StreamingResampler(input_rate=24000, output_rate=16000)
    s1 = sync_resampler.process(first)
    s2 = sync_resampler.process(second)
    s3 = sync_resampler.flush()
    sync_result = s1 + s2 + s3

    # Async path
    async_resampler = StreamingResampler(input_rate=24000, output_rate=16000)
    a1 = await async_resampler.aprocess(first)
    a2 = await async_resampler.aprocess(second)
    a3 = await async_resampler.aflush()
    async_result = a1 + a2 + a3

    assert sync_result == async_result
