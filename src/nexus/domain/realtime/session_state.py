from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from openai.types.chat import ChatCompletionChunk
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from nexus.application.realtime.protocol.tools import to_chat_tools
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.infrastructure.mcp import McpToolRegistry
from nexus.sessions.chat_session import AsyncChatSession

if TYPE_CHECKING:
    from asyncio import Task
    from nexus.application.realtime.protocol.server_writer import RealtimeServerWriter

logger = logging.getLogger(__name__)


@dataclass
class RealtimeSessionState:
    """Domain session state for a realtime websocket connection."""

    chat_session: AsyncChatSession
    chat_model: str
    writer: "RealtimeServerWriter"

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tools: List[RealtimeFunctionTool] = field(default_factory=list)
    mcp_registry: McpToolRegistry = field(default_factory=McpToolRegistry)

    audio_input_format_type: str = "audio/pcm"
    audio_input_sample_rate: int = 24000
    asr_sample_rate: int = 16000
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    audio_output_format_type: str = "audio/pcm"
    audio_output_voice: str = "alloy"
    audio_output_speed: float = 1.0
    audio_queue: asyncio.Queue[np.ndarray] = field(default_factory=asyncio.Queue)

    _current_chat_task: Optional["Task"] = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _cancel_reason: str = field(default="turn_detected", repr=False)

    async def send_event(self, event) -> None:
        await self.writer.send_event(event)

    def request_cancel(self, reason: str = "turn_detected") -> None:
        self._cancel_event.set()
        self._cancel_reason = reason

    def reset_cancel(self) -> None:
        self._cancel_event.clear()
        self._cancel_reason = "turn_detected"

    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def get_cancel_reason(self) -> str:
        return self._cancel_reason

    def set_current_chat_task(self, task: Optional["Task"]) -> None:
        self._current_chat_task = task

    def get_current_chat_task(self) -> Optional["Task"]:
        return self._current_chat_task

    async def audio_iter(self) -> AsyncGenerator[np.ndarray, None]:
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def update_output_modalities(self, modalities: List[str]) -> None:
        self.output_modalities = modalities

    def get_output_modalities(self) -> List[str]:
        return self.output_modalities.copy()

    def update_audio_output_config(
        self,
        *,
        format_type: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> None:
        if format_type is not None:
            self.audio_output_format_type = format_type
        if voice is not None:
            self.audio_output_voice = voice
        if speed is not None:
            self.audio_output_speed = speed

    def get_audio_output_config(self) -> dict:
        return {
            "format_type": self.audio_output_format_type,
            "voice": self.audio_output_voice,
            "speed": self.audio_output_speed,
        }

    def get_audio_input_config(self) -> dict:
        return {
            "format_type": self.audio_input_format_type,
            "sample_rate": self.audio_input_sample_rate,
        }

    def get_all_tools(self) -> List[RealtimeFunctionTool]:
        all_tools = list(self.tools)
        all_tools.extend(self.mcp_registry.to_realtime_function_tools())
        return all_tools

    def is_mcp_tool(self, tool_name: str) -> bool:
        return self.mcp_registry.is_mcp_tool(tool_name)

    def get_mcp_server_for_tool(self, tool_name: str) -> Optional[str]:
        return self.mcp_registry.get_server_for_tool(tool_name)

    async def chat(self, user_turn: PreparedRealtimeUserTurn) -> AsyncGenerator[ChatCompletionChunk, None]:
        chat_stream_resp = self.chat_session.chat(
            user_message=user_turn.model_text,
            model=self.chat_model,
            stream=True,
            tools=to_chat_tools(self.get_all_tools()),
        )
        async for chunk in chat_stream_resp:
            yield chunk

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.chat_session.chat_history.append(tool_msg)
        logger.info("Tool result added to history: tool_call_id=%s", tool_call_id)

    async def continue_conversation(self) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream_resp = await self.chat_session.chat_inferencer.chat(
            messages=self.chat_session.chat_history,
            model=self.chat_model,
            stream=True,
            tools=to_chat_tools(self.get_all_tools()),
        )
        async for chunk in self.chat_session.get_result_record_itr(stream_resp):
            yield chunk
