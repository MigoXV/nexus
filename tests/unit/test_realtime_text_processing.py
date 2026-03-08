from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.application.realtime.orchestrators.response_orchestrator import process_chat_stream
from nexus.application.realtime.text_processing import (
    SanitizedModelOutputAccumulator,
    parse_asr_speaker_prefix,
    prepare_realtime_user_turn,
)
from nexus.domain.realtime.session_state import RealtimeSessionState


class _FakeChatSession:
    def __init__(self) -> None:
        self.last_user_message: str | None = None

    def chat(self, *, user_message: str, **kwargs):
        del kwargs
        self.last_user_message = user_message

        async def _stream():
            if False:
                yield None

        return _stream()

    def replace_last_assistant_message_content(self, content: str) -> None:
        del content


@dataclass
class _CollectingSession:
    events: list[Any] = field(default_factory=list)
    replacements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat_session = SimpleNamespace(
            replace_last_assistant_message_content=self.replacements.append,
            chat_history=[],
        )
        self.writer = SimpleNamespace(send_error=_noop_async)

    async def send_event(self, event: Any) -> None:
        self.events.append(event)

    def is_cancel_requested(self) -> bool:
        return False

    def is_mcp_tool(self, tool_name: str) -> bool:
        del tool_name
        return False

    def get_mcp_server_for_tool(self, tool_name: str) -> None:
        del tool_name
        return None

    def get_cancel_reason(self) -> str:
        return "turn_detected"


async def _noop_async(**kwargs) -> None:
    del kwargs


def _make_chunk(content: str, *, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ]
    )


def test_parse_asr_speaker_prefix_extracts_voiceprint_name() -> None:
    speaker_name, display_transcript = parse_asr_speaker_prefix(
        "<sid migo 0.56> 来给我讲个有趣的故事吧"
    )

    assert speaker_name == "migo"
    assert display_transcript == "来给我讲个有趣的故事吧"


def test_parse_asr_speaker_prefix_ignores_others() -> None:
    speaker_name, display_transcript = parse_asr_speaker_prefix("<sid <others>> 你好")

    assert speaker_name is None
    assert display_transcript == "你好"


def test_prepare_realtime_user_turn_keeps_invalid_prefix_unchanged() -> None:
    turn = prepare_realtime_user_turn("<sid broken> 你好")

    assert turn.speaker_name is None
    assert turn.display_transcript == "<sid broken> 你好"
    assert turn.model_text == "<sid broken> 你好"


def test_prepare_realtime_user_turn_injects_hidden_speaker_context() -> None:
    turn = prepare_realtime_user_turn("<sid migo 0.56> 来给我讲个有趣的故事吧")

    assert turn.display_transcript == "来给我讲个有趣的故事吧"
    assert "当前说话人是migo" in turn.model_text
    assert "用户说：来给我讲个有趣的故事吧" in turn.model_text


def test_sanitized_output_accumulator_strips_markdown_symbols_and_emoji() -> None:
    accumulator = SanitizedModelOutputAccumulator()

    delta_1 = accumulator.push("### ")
    delta_2 = accumulator.push("**你好")
    delta_3 = accumulator.push("🙂世界**\n- ok")

    assert delta_1 == ("", "")
    assert delta_2 == ("你好", "你好")
    assert delta_3[0] == "世界 ok"
    assert accumulator.display_text == "你好世界 ok"
    assert accumulator.tts_text == "你好世界。ok"


@pytest.mark.asyncio
async def test_process_chat_stream_emits_clean_text_deltas() -> None:
    session = _CollectingSession()

    async def _chat_stream():
        yield _make_chunk("## 你好")
        yield _make_chunk("🙂世界\n")
        yield _make_chunk("**朋友**", finish_reason="stop")

    result = await process_chat_stream(
        session=session,
        chat_stream=_chat_stream(),
        modalities=["text"],
    )

    text_deltas = [
        event.delta
        for event in session.events
        if getattr(event, "type", None) == "response.output_text.delta"
    ]

    assert text_deltas == ["你好", "世界", "朋友"]
    assert result.content == "你好世界朋友"
    assert session.replacements == ["你好世界朋友"]


@pytest.mark.asyncio
async def test_realtime_session_chat_uses_prepared_turn_model_text() -> None:
    chat_session = _FakeChatSession()
    session = RealtimeSessionState(
        chat_session=chat_session,
        chat_model="gpt-4o-realtime-preview",
        writer=SimpleNamespace(send_event=_noop_async),
    )
    turn = prepare_realtime_user_turn("<sid migo 0.56> 给我讲个故事")

    async for _ in session.chat(turn):
        pass

    assert chat_session.last_user_message == turn.model_text
