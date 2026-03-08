from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import List

from nexus.infrastructure.tts import TTSBackend
from nexus.infrastructure.tts.text_normalizer import normalize_for_tts, split_text_by_punctuation

MIN_TTS_SEGMENT_CHARS = 30
MAX_TTS_SEGMENT_CHARS = 150
DEFAULT_TTS_SEGMENT_CONCURRENCY = 3
AUDIO_CHUNK_SIZE = 4096

_SENTINEL = object()  # 标记段内音频流结束


def split_text_to_tts_segments(
    text: str,
    *,
    min_segment_chars: int = MIN_TTS_SEGMENT_CHARS,
    max_segment_chars: int = MAX_TTS_SEGMENT_CHARS,
) -> List[str]:
    """Split text into TTS-ready segments while enforcing a minimum length."""
    normalized = normalize_for_tts(text)
    if not normalized:
        return []

    sentences = split_text_by_punctuation(normalized) or [normalized]

    segments: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        sentence_chunks = _split_sentence_by_max_chars(sentence, max_segment_chars)
        for chunk in sentence_chunks:
            if current and len(current) + len(chunk) > max_segment_chars:
                segments.append(current)
                current = ""
            current = f"{current}{chunk}" if current else chunk
            if len(current) >= min_segment_chars:
                segments.append(current)
                current = ""

    if current:
        if segments and len(segments[-1]) < min_segment_chars:
            merged = f"{segments[-1]}{current}"
            if len(merged) <= max_segment_chars:
                segments[-1] = merged
            else:
                segments.append(current)
        else:
            segments.append(current)

    return [segment for segment in segments if segment]


def _split_sentence_by_max_chars(sentence: str, max_segment_chars: int) -> List[str]:
    if max_segment_chars <= 0 or len(sentence) <= max_segment_chars:
        return [sentence]
    return [
        sentence[index : index + max_segment_chars]
        for index in range(0, len(sentence), max_segment_chars)
    ]


def realtime_audio_format_to_tts_response_format(format_type: str) -> str:
    """Map Realtime audio format type to TTS API response_format."""
    if format_type == "audio/pcm":
        return "pcm"
    raise ValueError(f"Unsupported realtime audio output format: {format_type}")


async def _produce_segment_to_queue(
    backend: TTSBackend,
    segment_text: str,
    voice: str,
    response_format: str,
    speed: float,
    queue: asyncio.Queue,
) -> None:
    """Stream TTS chunks for one segment into *queue*, then push sentinel."""
    try:
        async for chunk in backend.speech_stream(
            text=segment_text,
            model="tts-1",
            voice=voice,
            response_format=response_format,
            speed=speed,
        ):
            if chunk:
                await queue.put(chunk)
    finally:
        await queue.put(_SENTINEL)


async def stream_tts_audio_for_text(
    *,
    backend: TTSBackend,
    text: str,
    voice: str,
    speed: float,
    format_type: str,
    send_chunk: Callable[[bytes], Awaitable[None]],
    min_segment_chars: int = MIN_TTS_SEGMENT_CHARS,
    max_segment_chars: int = MAX_TTS_SEGMENT_CHARS,
    concurrency: int = DEFAULT_TTS_SEGMENT_CONCURRENCY,
) -> None:
    """Synthesize text in parallel segments and stream chunks in order.

    Each segment is synthesized concurrently (bounded by *concurrency*),
    but chunks are forwarded to *send_chunk* in strict segment order.
    Within each segment, chunks are streamed as soon as they arrive from
    the TTS backend — no buffering of the entire segment.
    """
    segments = split_text_to_tts_segments(
        text,
        min_segment_chars=min_segment_chars,
        max_segment_chars=max_segment_chars,
    )
    if not segments:
        return

    response_format = realtime_audio_format_to_tts_response_format(format_type)
    semaphore = asyncio.Semaphore(max(concurrency, 1))

    # 为每个段创建一个独立的 queue 和对应的生产者 task
    queues: list[asyncio.Queue] = []
    tasks: list[asyncio.Task] = []

    for seg in segments:
        q: asyncio.Queue = asyncio.Queue()
        queues.append(q)

        async def _bounded_produce(
            _seg: str = seg, _q: asyncio.Queue = q
        ) -> None:
            async with semaphore:
                await _produce_segment_to_queue(
                    backend, _seg, voice, response_format, speed, _q
                )

        tasks.append(asyncio.create_task(_bounded_produce()))

    try:
        # 按段顺序消费，段内流式
        for q in queues:
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    break
                await send_chunk(item)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
