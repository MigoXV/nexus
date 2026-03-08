"""Unit tests for TranscriptionStreamTracker and streaming transcription logic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import AsyncMock, patch

import pytest

from nexus.infrastructure.asr import TranscriptionResult
from nexus.application.realtime.orchestrators.response_orchestrator import (
    TranscriptionStreamTracker,
    send_transcribe_interim,
    send_transcribe_response,
)


# ---------------------------------------------------------------------------
# TranscriptionStreamTracker – pure logic tests
# ---------------------------------------------------------------------------


class TestTranscriptionStreamTracker:
    """Test delta computation from cumulative ASR transcripts."""

    def test_first_result_returns_full_text(self):
        tracker = TranscriptionStreamTracker()
        assert tracker.compute_delta("今天的") == "今天的"

    def test_incremental_prefix_extension(self):
        tracker = TranscriptionStreamTracker()
        assert tracker.compute_delta("今天的") == "今天的"
        assert tracker.compute_delta("今天的天气真") == "天气真"
        assert tracker.compute_delta("今天的天气真好") == "好"

    def test_identical_transcript_returns_empty(self):
        tracker = TranscriptionStreamTracker()
        tracker.compute_delta("hello")
        assert tracker.compute_delta("hello") == ""

    def test_non_prefix_fallback(self):
        """When ASR corrects earlier text, the full new transcript is returned."""
        tracker = TranscriptionStreamTracker()
        tracker.compute_delta("今天的天汽")
        # ASR corrected "天汽" -> "天气" – no longer a prefix
        delta = tracker.compute_delta("今天的天气")
        assert delta == "今天的天气"

    def test_reset_clears_state(self):
        tracker = TranscriptionStreamTracker()
        tracker.compute_delta("hello world")
        _ = tracker.item_id  # force allocation
        tracker.mark_speech_started()

        tracker.reset()

        assert tracker.compute_delta("new") == "new"
        assert not tracker.speech_started_sent
        # item_id should be newly allocated after reset
        old_id = tracker.item_id
        tracker.reset()
        assert tracker.item_id != old_id

    def test_item_id_stable_within_utterance(self):
        tracker = TranscriptionStreamTracker()
        id1 = tracker.item_id
        id2 = tracker.item_id
        assert id1 == id2

    def test_item_id_changes_after_reset(self):
        tracker = TranscriptionStreamTracker()
        id1 = tracker.item_id
        tracker.reset()
        id2 = tracker.item_id
        assert id1 != id2

    def test_speech_started_flag(self):
        tracker = TranscriptionStreamTracker()
        assert not tracker.speech_started_sent
        tracker.mark_speech_started()
        assert tracker.speech_started_sent


# ---------------------------------------------------------------------------
# Helpers for integration-style tests
# ---------------------------------------------------------------------------


@dataclass
class FakeSession:
    """Minimal stand-in for RealtimeSessionState."""

    events: List[Any] = field(default_factory=list)

    async def send_event(self, event: Any) -> None:
        self.events.append(event)


def _make_result(transcript: str, is_final: bool, words=None) -> TranscriptionResult:
    return TranscriptionResult(transcript=transcript, is_final=is_final, words=words)


# ---------------------------------------------------------------------------
# send_transcribe_interim tests
# ---------------------------------------------------------------------------


class TestSendTranscribeInterim:
    @pytest.mark.asyncio
    async def test_first_interim_sends_speech_started_and_delta(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()
        result = _make_result("今天的", is_final=False, words=[("今天的", 0.1, 0.5)])

        await send_transcribe_interim(session, result, tracker)

        types = [e.type for e in session.events]
        assert types == [
            "input_audio_buffer.speech_started",
            "conversation.item.input_audio_transcription.delta",
        ]
        assert tracker.speech_started_sent
        # Check delta content
        delta_event = session.events[1]
        assert delta_event.delta == "今天的"

    @pytest.mark.asyncio
    async def test_subsequent_interim_sends_only_delta(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_interim(
            session, _make_result("今天的", False, [("今天的", 0.1, 0.5)]), tracker
        )
        session.events.clear()

        await send_transcribe_interim(
            session, _make_result("今天的天气真", False), tracker
        )

        types = [e.type for e in session.events]
        assert types == ["conversation.item.input_audio_transcription.delta"]
        assert session.events[0].delta == "天气真"

    @pytest.mark.asyncio
    async def test_empty_delta_sends_nothing(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_interim(
            session, _make_result("hello", False, [("hello", 0.0, 0.5)]), tracker
        )
        session.events.clear()

        # Same transcript – no delta
        await send_transcribe_interim(session, _make_result("hello", False), tracker)
        assert session.events == []

    @pytest.mark.asyncio
    async def test_item_id_consistent_across_interims(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_interim(
            session, _make_result("a", False, [("a", 0.0, 0.1)]), tracker
        )
        await send_transcribe_interim(session, _make_result("ab", False), tracker)

        item_ids = {e.item_id for e in session.events}
        assert len(item_ids) == 1  # all events share the same item_id


# ---------------------------------------------------------------------------
# send_transcribe_response (final) tests
# ---------------------------------------------------------------------------


class TestSendTranscribeResponseWithTracker:
    @pytest.mark.asyncio
    async def test_final_after_interims_skips_speech_started(self):
        """speech_started was already sent during interim, should not repeat."""
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        # Simulate interim
        await send_transcribe_interim(
            session, _make_result("今天的", False, [("今天的", 0.1, 0.5)]), tracker
        )
        await send_transcribe_interim(
            session, _make_result("今天的天气真", False), tracker
        )
        session.events.clear()

        # Final
        await send_transcribe_response(
            session,
            _make_result("今天的天气真好", True, [("今天的天气真好", 0.1, 1.0)]),
            tracker,
        )

        types = [e.type for e in session.events]
        # Should NOT contain speech_started (already sent)
        assert "input_audio_buffer.speech_started" not in types
        # Should contain the rest
        assert "input_audio_buffer.speech_stopped" in types
        assert "input_audio_buffer.committed" in types
        assert "conversation.item.input_audio_transcription.delta" in types
        assert "conversation.item.input_audio_transcription.completed" in types
        assert "conversation.item.added" in types
        assert "conversation.item.done" in types

        # The final delta should be just "好"
        delta_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.delta"
        ]
        assert len(delta_events) == 1
        assert delta_events[0].delta == "好"

        # Completed transcript should be the full text
        completed = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert completed[0].transcript == "今天的天气真好"

    @pytest.mark.asyncio
    async def test_final_without_interims_sends_speech_started(self):
        """When no interim results came, final should send speech_started."""
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_response(
            session,
            _make_result("你好", True, [("你好", 0.0, 0.5)]),
            tracker,
        )

        types = [e.type for e in session.events]
        assert types[0] == "input_audio_buffer.speech_started"
        assert "conversation.item.input_audio_transcription.delta" in types
        # Delta should be full text since no prior interim
        delta_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.delta"
        ]
        assert delta_events[0].delta == "你好"

    @pytest.mark.asyncio
    async def test_tracker_reset_after_final(self):
        """After final, tracker should be clean for the next utterance."""
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_response(
            session, _make_result("first", True), tracker
        )
        # Tracker should be reset
        assert not tracker.speech_started_sent
        assert tracker.compute_delta("second") == "second"

    @pytest.mark.asyncio
    async def test_backward_compat_no_tracker(self):
        """When tracker is None, function should work like the original."""
        session = FakeSession()

        await send_transcribe_response(
            session, _make_result("hello", True, [("hello", 0.0, 0.3)])
        )

        types = [e.type for e in session.events]
        assert "input_audio_buffer.speech_started" in types
        assert "conversation.item.input_audio_transcription.delta" in types
        assert "conversation.item.input_audio_transcription.completed" in types

    @pytest.mark.asyncio
    async def test_final_transcript_strips_sid_prefix_from_client_events(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_response(
            session,
            _make_result("<sid migo 0.56> 来给我讲个故事", True, [("来给我讲个故事", 0.0, 0.5)]),
            tracker,
        )

        delta_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.delta"
        ]
        completed_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.completed"
        ]

        assert delta_events[0].delta == "来给我讲个故事"
        assert completed_events[0].transcript == "来给我讲个故事"

    @pytest.mark.asyncio
    async def test_others_sid_prefix_is_ignored_for_client_transcript(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_response(
            session,
            _make_result("<sid <others>> 你好", True, [("你好", 0.0, 0.5)]),
            tracker,
        )

        completed_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert completed_events[0].transcript == "你好"

    @pytest.mark.asyncio
    async def test_hide_metadata_false_keeps_sid_prefix_for_client_transcript(self):
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_response(
            session,
            _make_result("<sid migo 0.56> 来给我讲个故事", True, [("来给我讲个故事", 0.0, 0.5)]),
            tracker,
            hide_metadata=False,
        )

        completed_events = [
            e for e in session.events
            if e.type == "conversation.item.input_audio_transcription.completed"
        ]
        assert completed_events[0].transcript == "<sid migo 0.56> 来给我讲个故事"

    @pytest.mark.asyncio
    async def test_item_id_consistent_interim_to_final(self):
        """item_id should be the same across interim deltas and final events."""
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        await send_transcribe_interim(
            session, _make_result("a", False, [("a", 0.0, 0.1)]), tracker
        )
        await send_transcribe_response(
            session, _make_result("ab", True, [("ab", 0.0, 0.2)]), tracker
        )

        item_ids = {e.item_id for e in session.events if hasattr(e, "item_id")}
        assert len(item_ids) == 1


# ---------------------------------------------------------------------------
# Full sequence integration test
# ---------------------------------------------------------------------------


class TestFullStreamingSequence:
    @pytest.mark.asyncio
    async def test_chinese_sentence_streaming(self):
        """Simulate a full ASR streaming sequence for a Chinese sentence."""
        session = FakeSession()
        tracker = TranscriptionStreamTracker()

        # Simulate ASR engine returning cumulative transcripts
        asr_results = [
            _make_result("今天", False, [("今天", 0.1, 0.3)]),
            _make_result("今天的天气", False),
            _make_result("今天的天气真好", False),
            _make_result("今天的天气真好啊", True, [("今天的天气真好啊", 0.1, 1.2)]),
        ]

        for result in asr_results:
            if not result.is_final:
                await send_transcribe_interim(session, result, tracker)
            else:
                await send_transcribe_response(session, result, tracker)

        # Collect all delta values
        deltas = [
            e.delta
            for e in session.events
            if hasattr(e, "type")
            and e.type == "conversation.item.input_audio_transcription.delta"
        ]
        assert deltas == ["今天", "的天气", "真好", "啊"]
        # Concatenated deltas should equal the final transcript
        assert "".join(deltas) == "今天的天气真好啊"

        # Verify event ordering
        types = [e.type for e in session.events]
        # speech_started should be the very first event
        assert types[0] == "input_audio_buffer.speech_started"
        # Only one speech_started
        assert types.count("input_audio_buffer.speech_started") == 1
        # completed should come after all deltas
        completed_idx = types.index(
            "conversation.item.input_audio_transcription.completed"
        )
        last_delta_idx = len(types) - 1 - types[::-1].index(
            "conversation.item.input_audio_transcription.delta"
        )
        assert completed_idx > last_delta_idx
