from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from nexus.application.realtime.orchestrators.transcription_worker import run_transcription_worker
from nexus.infrastructure.asr import TranscriptionResult


@dataclass
class _FakeSession:
    chunks: list[np.ndarray]
    audio_input_sample_rate: int = 24000
    asr_sample_rate: int = 16000
    _current_chat_task: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def audio_iter(self) -> AsyncIterator[np.ndarray]:
        for chunk in self.chunks:
            yield chunk

    def get_current_chat_task(self):
        return self._current_chat_task

    def set_current_chat_task(self, task):
        self._current_chat_task = task

    def request_cancel(self, reason: str = "turn_detected") -> None:
        del reason

    def reset_cancel(self) -> None:
        return


class _FakeInferencer:
    def __init__(self) -> None:
        self.sample_rate: int | None = None
        self.interim_results: bool | None = None
        self.collected: list[np.ndarray] = []

    async def transcribe(
        self,
        audio: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        interim_results: bool = True,
        **kwargs,
    ):
        del kwargs
        self.sample_rate = sample_rate
        self.interim_results = interim_results
        async for chunk in audio:
            self.collected.append(chunk)

        yield TranscriptionResult(transcript="ok", is_final=True)


@pytest.mark.asyncio
async def test_transcription_worker_resamples_24k_to_16k_before_asr() -> None:
    # 24kHz 1 second PCM16 ramp split into uneven chunks.
    samples_24k = np.arange(24000, dtype=np.int16)
    chunks = [samples_24k[:5000], samples_24k[5000:13333], samples_24k[13333:]]
    session = _FakeSession(chunks=chunks)
    inferencer = _FakeInferencer()

    with (
        patch(
            "nexus.application.realtime.orchestrators.transcription_worker.send_transcribe_interim",
            new=AsyncMock(),
        ),
        patch(
            "nexus.application.realtime.orchestrators.transcription_worker.send_transcribe_response",
            new=AsyncMock(),
        ) as mocked_final,
    ):
        await run_transcription_worker(
            inferencer=inferencer,
            session=session,
            interim_results=True,
            is_chat_model=False,
            chat_worker=AsyncMock(),
        )

    assert inferencer.sample_rate == 16000
    assert inferencer.interim_results is True
    assert mocked_final.await_count == 1

    total_resampled_samples = int(sum(chunk.size for chunk in inferencer.collected))
    # 24k -> 16k should be approximately 2/3 length (allow filter edge effects).
    assert abs(total_resampled_samples - 16000) <= 256
