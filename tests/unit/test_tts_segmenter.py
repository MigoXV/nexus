from __future__ import annotations

import pytest

from nexus.application.realtime.orchestrators.tts_orchestrator import (
    split_text_to_tts_segments,
    stream_tts_audio_for_text,
)


class FakeTTSInferencer:
    def __init__(self, delays: dict[str, float] | None = None):
        self.delays = delays or {}

    async def speech_stream(
        self,
        *,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "pcm",
        speed: float = 1.0,
        **kwargs,
    ):
        del model, voice, response_format, speed, kwargs
        delay = self.delays.get(text, 0.0)
        if delay > 0:
            import asyncio

            await asyncio.sleep(delay)

        content = text.encode("utf-8")
        mid = len(content) // 2
        if mid > 0:
            yield content[:mid]
            yield content[mid:]
        else:
            yield content


def test_split_text_to_tts_segments_enforces_min_chars() -> None:
    text = (
        "这是第一句很短。"
        "这是第二句也很短。"
        "这是第三句会让聚合后的文本超过三十个字。"
        "最后一句收尾。"
    )
    segments = split_text_to_tts_segments(text, min_segment_chars=30)

    assert segments
    if len(segments) > 1:
        for segment in segments[:-1]:
            assert len(segment) >= 30
    assert "".join(segments).replace(" ", "") == text.replace(" ", "")


def test_split_text_to_tts_segments_keeps_short_single_utterance() -> None:
    text = "你好世界。"
    segments = split_text_to_tts_segments(text, min_segment_chars=30)
    assert segments == [text]


def test_split_text_to_tts_segments_caps_each_segment_at_150_chars() -> None:
    sentence_a = "甲" * 80 + "。"
    sentence_b = "乙" * 80 + "。"
    segments = split_text_to_tts_segments(
        sentence_a + sentence_b,
        min_segment_chars=30,
        max_segment_chars=150,
    )

    assert segments
    assert all(len(segment) <= 150 for segment in segments)
    assert "".join(segments) == sentence_a + sentence_b


def test_split_text_to_tts_segments_hard_splits_single_long_sentence() -> None:
    text = "长" * 320
    segments = split_text_to_tts_segments(
        text,
        min_segment_chars=30,
        max_segment_chars=150,
    )

    assert [len(segment) for segment in segments] == [150, 150, 20]
    assert "".join(segments) == text


@pytest.mark.asyncio
async def test_stream_tts_audio_for_text_keeps_segment_order() -> None:
    segment_a = "a" * 30 + "。"
    segment_b = "b" * 30 + "。"
    text = segment_a + segment_b

    backend = FakeTTSInferencer(
        delays={
            segment_a: 0.2,
            segment_b: 0.01,
        }
    )

    emitted: list[bytes] = []

    async def send_chunk(chunk: bytes) -> None:
        emitted.append(chunk)

    await stream_tts_audio_for_text(
        backend=backend,
        text=text,
        voice="alloy",
        speed=1.0,
        format_type="audio/pcm",
        send_chunk=send_chunk,
        min_segment_chars=30,
        concurrency=3,
    )

    joined = b"".join(emitted).decode("utf-8")
    assert joined == f"{segment_a}{segment_b}"
