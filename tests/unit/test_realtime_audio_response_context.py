from __future__ import annotations

import asyncio
import numpy as np
import pytest

from nexus.application.realtime.emitters.response_contexts import AudioResponseContext
from nexus.infrastructure.tts.backend import DuplexAudioChunk


class CollectingSession:
    def __init__(self):
        self.events = []

    async def send_event(self, event):
        self.events.append(event)

    def get_cancel_reason(self) -> str:
        return "turn_detected"


class FakeTTSInferencer:
    """Fake TTS that yields valid PCM16 audio data large enough for the
    48 kHz→24 kHz resampler to produce output."""

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
        del text, model, voice, response_format, speed, kwargs
        # Generate ~0.1 s of 48 kHz PCM16 silence-ish data (4800 samples = 9600 bytes).
        # This is large enough for even sinc_best to produce resampled output.
        n_samples = 4800
        pcm = np.zeros(n_samples, dtype=np.int16).tobytes()
        yield pcm


class FakeDuplexSession:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[DuplexAudioChunk | None] = asyncio.Queue()
        self.sent_text: list[str] = []
        self.end_called = False
        self.closed = False

    async def send_text(self, delta: str) -> None:
        self.sent_text.append(delta)
        await self.queue.put(DuplexAudioChunk(data=delta.encode("utf-8"), sample_rate=24000))

    async def end_input(self) -> None:
        self.end_called = True
        await self.queue.put(None)

    async def iter_audio(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item

    async def aclose(self) -> None:
        self.closed = True
        if not self.end_called:
            await self.queue.put(None)


class FakeDuplexBackend:
    supports_duplex = True

    def __init__(self) -> None:
        self.session = FakeDuplexSession()

    async def open_duplex_session(self, **kwargs):
        del kwargs
        return self.session


@pytest.mark.asyncio
async def test_audio_context_emits_audio_and_transcript_events_without_text_events() -> None:
    session = CollectingSession()
    ctx = AudioResponseContext(
        session=session,
        tts_backend=FakeTTSInferencer(),
        modalities=["audio"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("你好，音频模式。")
    event_types_before_audio = [event.type for event in session.events]
    assert "response.output_audio_transcript.delta" not in event_types_before_audio
    await ctx.synthesize_audio()
    await ctx.finish()

    event_types = [event.type for event in session.events]
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio.done" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert "response.output_audio_transcript.done" in event_types
    assert "response.output_text.delta" not in event_types
    assert "response.output_text.done" not in event_types
    assert all(not isinstance(event, dict) for event in session.events)
    assert event_types.index("response.output_audio.delta") < event_types.index(
        "response.output_audio_transcript.delta"
    )


@pytest.mark.asyncio
async def test_audio_context_with_text_modalities_keeps_audio_transcript_events() -> None:
    session = CollectingSession()
    ctx = AudioResponseContext(
        session=session,
        tts_backend=FakeTTSInferencer(),
        modalities=["audio", "text"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("第一段。")
    await ctx.add_model_text_delta("第二段。")
    event_types_before_audio = [event.type for event in session.events]
    assert "response.output_audio_transcript.delta" not in event_types_before_audio
    await ctx.synthesize_audio()
    await ctx.finish()

    event_types = [event.type for event in session.events]
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio.done" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert "response.output_audio_transcript.done" in event_types
    assert "response.output_text.delta" not in event_types
    assert "response.output_text.done" not in event_types
    assert event_types.index("response.output_audio.delta") < event_types.index(
        "response.output_audio_transcript.delta"
    )


@pytest.mark.asyncio
async def test_audio_context_duplex_backend_emits_audio_before_finish() -> None:
    session = CollectingSession()
    backend = FakeDuplexBackend()
    ctx = AudioResponseContext(
        session=session,
        tts_backend=backend,
        modalities=["audio"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
        tts_sample_rate=24000,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("hello")
    event_types_before_audio = [event.type for event in session.events]
    assert "response.output_audio_transcript.delta" not in event_types_before_audio
    await asyncio.sleep(0)

    event_types = [event.type for event in session.events]
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio_transcript.delta" in event_types
    assert event_types.index("response.output_audio.delta") < event_types.index(
        "response.output_audio_transcript.delta"
    )
    assert backend.session.sent_text == ["hello"]

    await ctx.synthesize_audio()
    await ctx.finish()

    assert backend.session.end_called is True
    assert backend.session.closed is True


@pytest.mark.asyncio
async def test_audio_context_uses_clean_transcript_and_tts_text_separately() -> None:
    session = CollectingSession()
    backend = FakeDuplexBackend()
    ctx = AudioResponseContext(
        session=session,
        tts_backend=backend,
        modalities=["audio"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
        tts_sample_rate=24000,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("你好世界", tts_delta="你好世界。")
    await asyncio.sleep(0)
    await ctx.synthesize_audio()
    await ctx.finish()

    transcript_deltas = [
        event.delta
        for event in session.events
        if getattr(event, "type", None) == "response.output_audio_transcript.delta"
    ]

    assert transcript_deltas == ["你好世界"]
    assert backend.session.sent_text == ["你好世界。"]


@pytest.mark.asyncio
async def test_audio_context_duplex_backend_closes_on_cancel() -> None:
    session = CollectingSession()
    backend = FakeDuplexBackend()
    ctx = AudioResponseContext(
        session=session,
        tts_backend=backend,
        modalities=["audio"],
        format_type="audio/pcm",
        voice="alloy",
        speed=1.0,
        tts_sample_rate=24000,
    )

    await ctx.__aenter__()
    await ctx.add_model_text_delta("partial")
    await ctx.finish(cancelled=True)

    assert backend.session.closed is True
    event_types = [event.type for event in session.events]
    assert "response.output_audio_transcript.delta" not in event_types
    assert "response.output_audio_transcript.done" not in event_types
