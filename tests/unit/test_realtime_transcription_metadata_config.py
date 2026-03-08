from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from nexus.application.realtime.orchestrators.transcription_worker import run_transcription_worker
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.infrastructure.asr import TranscriptionResult


@dataclass
class _FakeSession:
    events: list[Any] = field(default_factory=list)
    audio_input_sample_rate: int = 16000
    asr_sample_rate: int = 16000
    audio_queue: asyncio.Queue[np.ndarray | None] = field(default_factory=asyncio.Queue)
    _current_chat_task: asyncio.Task | None = None

    async def send_event(self, event: Any) -> None:
        self.events.append(event)

    async def audio_iter(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def get_current_chat_task(self):
        return self._current_chat_task

    def set_current_chat_task(self, task):
        self._current_chat_task = task

    def request_cancel(self, reason: str = "turn_detected") -> None:
        del reason

    def reset_cancel(self) -> None:
        return None


class _FakeInferencer:
    def __init__(self, results: list[TranscriptionResult]) -> None:
        self.results = results

    async def transcribe(self, audio, **kwargs):
        del kwargs
        async for _ in audio:
            pass
        for result in self.results:
            yield result


@pytest.mark.asyncio
async def test_transcription_worker_preserves_metadata_for_client_when_disabled() -> None:
    session = _FakeSession()
    await session.audio_queue.put(np.zeros(160, dtype=np.int16))
    await session.audio_queue.put(None)

    captured_turns: list[PreparedRealtimeUserTurn] = []

    async def _chat_worker(session_arg, turn: PreparedRealtimeUserTurn) -> None:
        del session_arg
        captured_turns.append(turn)

    inferencer = _FakeInferencer(
        [
            TranscriptionResult(
                transcript="<sid migo 0.56> 来给我讲个故事",
                is_final=True,
                words=[("来给我讲个故事", 0.0, 0.3)],
            )
        ]
    )

    await run_transcription_worker(
        inferencer=inferencer,
        session=session,
        interim_results=False,
        hide_metadata=False,
        is_chat_model=True,
        chat_worker=_chat_worker,
    )

    completed_events = [
        event for event in session.events
        if getattr(event, "type", None) == "conversation.item.input_audio_transcription.completed"
    ]
    if session.get_current_chat_task() is not None:
        await session.get_current_chat_task()

    assert completed_events[0].transcript == "<sid migo 0.56> 来给我讲个故事"
    assert captured_turns[0].display_transcript == "来给我讲个故事"
    assert captured_turns[0].speaker_name == "migo"
    assert "当前说话人是migo" in captured_turns[0].model_text
