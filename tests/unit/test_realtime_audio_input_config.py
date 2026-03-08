from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.application.realtime.service import RealtimeApplicationService


def _service_without_init() -> RealtimeApplicationService:
    return RealtimeApplicationService.__new__(RealtimeApplicationService)


def test_validate_audio_input_update_accepts_pcm24k() -> None:
    service = _service_without_init()
    update = {
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 24000,
                }
            }
        }
    }
    service._validate_audio_input_update(update)


def test_validate_audio_input_update_rejects_non_24k() -> None:
    service = _service_without_init()
    update = {
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 16000,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="sample rate"):
        service._validate_audio_input_update(update)


def test_validate_audio_input_update_rejects_non_pcm() -> None:
    service = _service_without_init()
    update = {
        "audio": {
            "input": {
                "format": {
                    "type": "audio/pcmu",
                    "rate": 24000,
                }
            }
        }
    }

    with pytest.raises(ValueError, match="input format"):
        service._validate_audio_input_update(update)


@dataclass
class _FakeTool:
    payload: dict

    def model_dump(self, exclude_none: bool = True) -> dict:
        del exclude_none
        return self.payload


@dataclass
class _FakeSession:
    session_id: str = "sess_test"

    def get_output_modalities(self) -> list[str]:
        return ["audio"]

    def get_audio_input_config(self) -> dict:
        return {"format_type": "audio/pcm", "sample_rate": 24000}

    def get_audio_output_config(self) -> dict:
        return {"format_type": "audio/pcm", "voice": "alloy", "speed": 1.0}

    def get_all_tools(self) -> list[_FakeTool]:
        return [_FakeTool({"type": "function", "name": "demo", "parameters": {"type": "object"}})]


def test_session_payload_contains_input_and_output_audio_rates() -> None:
    service = _service_without_init()
    payload = service._session_payload(session=_FakeSession(), model="gpt-4o-realtime-preview")

    assert payload["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert payload["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}


@pytest.mark.asyncio
async def test_apply_session_update_rejects_invalid_input_rate_without_session_updated() -> None:
    service = _service_without_init()
    service.tts_backend = None

    writer = SimpleNamespace(send_error=AsyncMock())
    session = SimpleNamespace(
        chat_model="gpt-4o-realtime-preview",
        writer=writer,
        update_output_modalities=lambda modalities: modalities,
        update_audio_output_config=lambda **kwargs: kwargs,
        mcp_registry=SimpleNamespace(server_labels=[]),
        get_output_modalities=lambda: ["text"],
        get_audio_input_config=lambda: {"format_type": "audio/pcm", "sample_rate": 24000},
        get_audio_output_config=lambda: {"format_type": "audio/pcm", "voice": "alloy", "speed": 1.0},
        get_all_tools=lambda: [],
        send_event=AsyncMock(),
    )
    update = SimpleNamespace(
        model="gpt-4o-realtime-preview",
        output_modalities=None,
        tools=None,
        audio={
            "input": {
                "format": {
                    "type": "audio/pcm",
                    "rate": 16000,
                }
            }
        },
    )

    await service.apply_session_update(session, update, model="gpt-4o-realtime-preview")

    assert writer.send_error.await_count == 1
    assert session.send_event.await_count == 0
