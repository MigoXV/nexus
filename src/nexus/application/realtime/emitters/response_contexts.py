"""
响应上下文管理器模块。

提供用于管理响应生命周期的上下文管理器，自动处理前置/后置事件发送。
"""
import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Any, Dict, List

from openai.types.realtime import McpListToolsFailed, ResponseMcpCallFailed

from nexus.application.realtime.orchestrators.tts_orchestrator import stream_tts_audio_for_text
from nexus.infrastructure.tts.resampler import StreamingResampler
from nexus.application.realtime.protocol.ids import (
    event_id,
    item_id,
    response_id,
    conversation_id,
)

from .event_factory import (
    build_response_created_event,
    build_response_done_event,
    build_assistant_message_item,
    build_output_item_added_event,
    build_conversation_item_added_event,
    build_content_part_added_event,
    build_text_delta_event,
    build_text_done_event,
    build_content_part_done_event,
    build_audio_content_part_added_event,
    build_audio_content_part_done_event,
    build_audio_delta_event,
    build_audio_done_event,
    build_audio_transcript_delta_event,
    build_audio_transcript_done_event,
    build_conversation_item_done_event,
    build_output_item_done_event,
    build_function_call_item,
    build_function_call_output_item_added_event,
    build_function_call_conversation_item_added_event,
    build_function_call_conversation_item_done_event,
    build_function_call_arguments_delta,
    build_function_call_arguments_done,
    realtime_conversation_item_assistant_message,
    # MCP events
    build_mcp_list_tools_item,
    build_mcp_list_tools_added_event,
    build_mcp_list_tools_in_progress_event,
    build_mcp_list_tools_completed_event,
    build_mcp_list_tools_done_event,
    build_mcp_call_item,
    build_mcp_call_output_item_added_event,
    build_mcp_call_conversation_item_added_event,
    build_mcp_call_arguments_delta_event,
    build_mcp_call_arguments_done_event,
    build_mcp_call_in_progress_event,
    build_mcp_call_completed_event,
    build_mcp_call_output_item_done_event,
    build_mcp_call_conversation_item_done_event,
)

if TYPE_CHECKING:
    from nexus.domain.realtime import RealtimeSessionState
    from nexus.infrastructure.tts import DuplexTTSSession, TTSBackend

logger = logging.getLogger(__name__)


class TextResponseContext:
    """
    文本响应上下文管理器。
    
    管理响应的生命周期，在进入时发送前置事件，在退出时发送后置事件。
    使用方式：
        async with TextResponseContext(session) as ctx:
            await ctx.send_text_delta("Hello")
            await ctx.send_text_delta(" World")
    """
    
    def __init__(
        self,
        session: "RealtimeSessionState",
        modalities: List[str] = None,
    ):
        self.session = session
        self.modalities = modalities or ["text"]
        
        # 生成 IDs
        self.item_id = item_id()
        self.response_id = response_id()
        self.conversation_id = conversation_id()
        
        # 累积的内容
        self._content = ""
        # 助手消息 item
        self._item = None
    
    @property
    def content(self) -> str:
        """返回累积的内容"""
        return self._content
    
    async def __aenter__(self) -> "TextResponseContext":
        """发送前置事件"""
        # 1. Response created
        await self.session.send_event(
            build_response_created_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        
        # 2. 创建 assistant message item
        self._item = build_assistant_message_item(
            item_id=self.item_id,
            status="in_progress",
        )
        
        # 3. Output item added
        await self.session.send_event(
            build_output_item_added_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_conversation_item_added_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        # 5. Content part added
        await self.session.send_event(
            build_content_part_added_event(
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件（正常完成）"""
        await self.finish(cancelled=False)
        return False  # 不抑制异常
    
    async def finish(self, cancelled: bool = False):
        """
        完成响应并发送后置事件。
        
        Args:
            cancelled: 是否因为被打断而结束（新的转写事件到来）
        """
        # 6. Text done
        await self.session.send_event(
            build_text_done_event(
                text=self._content,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 7. Content part done
        await self.session.send_event(
            build_content_part_done_event(
                text=self._content,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 8. 更新 item 内容和状态
        # 如果被取消，状态为 incomplete，否则为 completed
        status = "incomplete" if cancelled else "completed"
        self._item.content.append(
            realtime_conversation_item_assistant_message.Content(
                audio=None, text=self._content, transcript=None, type="output_text"
            )
        )
        self._item.status = status
        
        # 9. Output item done (新增)
        await self.session.send_event(
            build_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 10. Conversation item done
        await self.session.send_event(
            build_conversation_item_done_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        # 11. Response done（取消时通过 response.status/status_details 表达）
        if cancelled:
            reason = (
                self.session.get_cancel_reason()
                if hasattr(self.session, "get_cancel_reason")
                else "turn_detected"
            )
            await self.session.send_event(
                build_response_done_event(
                    response_id=self.response_id,
                    conversation_id=self.conversation_id,
                    event_id=event_id(),
                    modalities=self.modalities,
                    status="cancelled",
                    reason=reason,
                )
            )
            logger.info(
                f"TextResponseContext cancelled: item_id={self.item_id}, "
                f"response_id={self.response_id}, partial_content='{self._content}'"
            )
        else:
            await self.session.send_event(
                build_response_done_event(
                    response_id=self.response_id,
                    conversation_id=self.conversation_id,
                    event_id=event_id(),
                    modalities=self.modalities,
                )
            )
            logger.info(
                f"TextResponseContext completed: item_id={self.item_id}, "
                f"response_id={self.response_id}, content='{self._content}'"
            )
    
    async def send_text_delta(self, delta: str):
        """发送文本增量"""
        if not delta:
            return
        self._content += delta
        await self.session.send_event(
            build_text_delta_event(
                delta=delta,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )


class AudioResponseContext:
    """音频响应上下文管理器（支持 output_audio_transcript 事件）。"""

    # 上游 TTS 服务的原始采样率
    TTS_INPUT_SAMPLE_RATE = 48000
    # Realtime 协议要求的输出采样率
    REALTIME_OUTPUT_SAMPLE_RATE = 24000

    def __init__(
        self,
        session: "RealtimeSessionState",
        *,
        tts_backend: "TTSBackend",
        modalities: List[str] | None = None,
        format_type: str = "audio/pcm",
        voice: str = "alloy",
        speed: float = 1.0,
        min_segment_chars: int = 30,
        segment_concurrency: int = 3,
        tts_sample_rate: int | None = None,
        output_sample_rate: int | None = None,
    ):
        self.session = session
        self.tts_backend = tts_backend
        self.modalities = modalities or ["audio"]
        self.format_type = format_type
        self.voice = voice
        self.speed = speed
        self.min_segment_chars = min_segment_chars
        self.segment_concurrency = segment_concurrency

        self.item_id = item_id()
        self.response_id = response_id()
        self.conversation_id = conversation_id()

        self._display_text = ""
        self._tts_text = ""
        self._item = None
        self._audio_delta_count = 0
        self._duplex_session: "DuplexTTSSession | None" = None
        self._duplex_consumer_task = None
        self._duplex_audio_failed = False
        self._audio_started = False
        self._pending_transcript_deltas: list[str] = []

        # 流式重采样：当上游 TTS 采样率与输出不一致时启用
        _in_rate = tts_sample_rate or self.TTS_INPUT_SAMPLE_RATE
        _out_rate = output_sample_rate or self.REALTIME_OUTPUT_SAMPLE_RATE
        self._resampler_input_rate = _in_rate
        if _in_rate != _out_rate:
            self._resampler: StreamingResampler | None = StreamingResampler(
                input_rate=_in_rate,
                output_rate=_out_rate,
            )
            logger.info(
                "AudioResponseContext: streaming resampler enabled %dHz -> %dHz",
                _in_rate,
                _out_rate,
            )
        else:
            self._resampler = None

    @property
    def include_transcript(self) -> bool:
        # audio modality always carries transcript events in this service.
        return True

    @property
    def content(self) -> str:
        return self._display_text

    @property
    def tts_content(self) -> str:
        return self._tts_text

    async def __aenter__(self) -> "AudioResponseContext":
        await self.session.send_event(
            build_response_created_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )

        self._item = build_assistant_message_item(item_id=self.item_id, status="in_progress")
        await self.session.send_event(
            build_output_item_added_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        await self.session.send_event(
            build_conversation_item_added_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        await self.session.send_event(
            build_audio_content_part_added_event(
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        if getattr(self.tts_backend, "supports_duplex", False):
            self._duplex_session = await self.tts_backend.open_duplex_session(
                model="tts-1",
                voice=self.voice,
                response_format="pcm",
                speed=self.speed,
            )
            self._duplex_consumer_task = asyncio.create_task(self._consume_duplex_audio())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.finish(cancelled=False)
        return False

    async def add_model_text_delta(self, delta: str, *, tts_delta: str | None = None) -> None:
        if not delta and not tts_delta:
            return
        tts_delta = delta if tts_delta is None else tts_delta
        if delta:
            self._display_text += delta
        if tts_delta:
            self._tts_text += tts_delta
        if self._duplex_session is not None and tts_delta:
            await self._duplex_session.send_text(tts_delta)
        if not self.include_transcript or not delta:
            return
        if self._audio_started:
            await self._emit_transcript_delta(delta)
            return
        self._pending_transcript_deltas.append(delta)

    async def _consume_duplex_audio(self) -> None:
        if self._duplex_session is None:
            return
        try:
            async for chunk in self._duplex_session.iter_audio():
                if chunk.sample_rate and self._resampler_input_rate != chunk.sample_rate:
                    self._resampler_input_rate = chunk.sample_rate
                    if chunk.sample_rate == self.REALTIME_OUTPUT_SAMPLE_RATE:
                        self._resampler = None
                    else:
                        self._resampler = StreamingResampler(
                            input_rate=chunk.sample_rate,
                            output_rate=self.REALTIME_OUTPUT_SAMPLE_RATE,
                        )
                await self._send_audio_bytes(chunk.data)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._duplex_audio_failed = True
            raise

    async def _send_audio_raw(self, audio_bytes: bytes) -> None:
        """直接将音频 bytes base64 编码后发送给客户端。"""
        if not audio_bytes:
            return
        is_first_audio_delta = not self._audio_started
        delta_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        await self.session.send_event(
            build_audio_delta_event(
                delta=delta_b64,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        self._audio_delta_count += 1
        if is_first_audio_delta:
            self._audio_started = True
            await self._flush_pending_transcript_deltas()

    async def _send_audio_bytes(self, audio_bytes: bytes) -> None:
        """接收上游 TTS 原始音频，经流式重采样后发送。"""
        if self._resampler is not None:
            resampled = await self._resampler.aprocess(audio_bytes)
            if resampled:
                await self._send_audio_raw(resampled)
        else:
            await self._send_audio_raw(audio_bytes)

    async def synthesize_audio(self) -> None:
        if not self._tts_text.strip():
            return
        if self._duplex_session is not None:
            await self._duplex_session.end_input()
            if self._duplex_consumer_task is not None:
                await self._duplex_consumer_task
            if self._resampler is not None:
                tail = await self._resampler.aflush()
                if tail:
                    await self._send_audio_raw(tail)
            return

        await stream_tts_audio_for_text(
            backend=self.tts_backend,
            text=self._tts_text,
            voice=self.voice,
            speed=self.speed,
            format_type=self.format_type,
            send_chunk=self._send_audio_bytes,
            min_segment_chars=self.min_segment_chars,
            concurrency=self.segment_concurrency,
        )
        # 冲刷重采样器内部滤波器缓冲区中的剩余数据
        if self._resampler is not None:
            tail = await self._resampler.aflush()
            if tail:
                await self._send_audio_raw(tail)

    async def finish(
        self,
        *,
        cancelled: bool = False,
        failed: bool = False,
        error_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if cancelled or failed:
            await self._abort_duplex()
        else:
            await self._close_duplex()

        await self.session.send_event(
            build_audio_done_event(
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )

        transcript = self._display_text if self.include_transcript and self._audio_started else None
        if transcript is not None:
            await self.session.send_event(
                build_audio_transcript_done_event(
                    transcript=transcript,
                    item_id=self.item_id,
                    response_id=self.response_id,
                    event_id=event_id(),
                )
            )

        await self.session.send_event(
            build_audio_content_part_done_event(
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
                transcript=transcript,
            )
        )

        item_status = "completed"
        if cancelled or failed:
            item_status = "incomplete"

        self._item.content.append(
            realtime_conversation_item_assistant_message.Content(
                audio=None,
                text=None,
                transcript=transcript,
                type="output_audio",
            )
        )
        self._item.status = item_status

        await self.session.send_event(
            build_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        await self.session.send_event(
            build_conversation_item_done_event(
                item=self._item,
                event_id=event_id(),
            )
        )

        if failed:
            await self.session.send_event(
                build_response_done_event(
                    response_id=self.response_id,
                    conversation_id=self.conversation_id,
                    event_id=event_id(),
                    modalities=self.modalities,
                    status="failed",
                    error_code=error_code or "audio_synthesis_failed",
                    error_type=error_type or "server_error",
                )
            )
            logger.warning(
                "AudioResponseContext failed: item_id=%s response_id=%s content_len=%s deltas=%s",
                self.item_id,
                self.response_id,
                len(self._display_text),
                self._audio_delta_count,
            )
            return

        if cancelled:
            reason = (
                self.session.get_cancel_reason()
                if hasattr(self.session, "get_cancel_reason")
                else "turn_detected"
            )
            await self.session.send_event(
                build_response_done_event(
                    response_id=self.response_id,
                    conversation_id=self.conversation_id,
                    event_id=event_id(),
                    modalities=self.modalities,
                    status="cancelled",
                    reason=reason,
                )
            )
            logger.info(
                "AudioResponseContext cancelled: item_id=%s response_id=%s content_len=%s deltas=%s",
                self.item_id,
                self.response_id,
                len(self._display_text),
                self._audio_delta_count,
            )
            return

        await self.session.send_event(
            build_response_done_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        logger.info(
            "AudioResponseContext completed: item_id=%s response_id=%s content_len=%s deltas=%s",
            self.item_id,
            self.response_id,
            len(self._display_text),
            self._audio_delta_count,
        )

    async def _abort_duplex(self) -> None:
        if self._duplex_consumer_task is not None and not self._duplex_consumer_task.done():
            self._duplex_consumer_task.cancel()
            try:
                await self._duplex_consumer_task
            except asyncio.CancelledError:
                pass
        await self._close_duplex()

    async def _close_duplex(self) -> None:
        if self._duplex_session is not None:
            await self._duplex_session.aclose()
            self._duplex_session = None

    async def _emit_transcript_delta(self, delta: str) -> None:
        await self.session.send_event(
            build_audio_transcript_delta_event(
                delta=delta,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )

    async def _flush_pending_transcript_deltas(self) -> None:
        if not self.include_transcript or not self._pending_transcript_deltas:
            return
        pending = self._pending_transcript_deltas
        self._pending_transcript_deltas = []
        for delta in pending:
            await self._emit_transcript_delta(delta)


class FunctionCallResponseContext:
    """
    工具调用响应上下文管理器。
    
    管理工具调用响应的生命周期，发送完整的事件序列。
    
    官方事件时序：
    1. ResponseCreatedEvent
    2. ResponseOutputItemAddedEvent  
    3. ConversationItemAdded
    4. ResponseFunctionCallArgumentsDeltaEvent (多个)
    5. ResponseFunctionCallArgumentsDoneEvent
    6. ConversationItemDone
    7. ResponseOutputItemDoneEvent
    8. ResponseDoneEvent
    
    使用方式：
        async with FunctionCallResponseContext(session, name, call_id) as ctx:
            await ctx.send_arguments_delta('{"')
            await ctx.send_arguments_delta('city')
            ...
    """
    
    def __init__(
        self,
        session: "RealtimeSessionState",
        name: str,
        call_id: str,
        modalities: List[str] = None,
    ):
        self.session = session
        self.name = name
        self.call_id = call_id
        self.modalities = modalities or ["text"]
        
        # 生成 IDs
        self.item_id = item_id()
        self.response_id = response_id()
        self.conversation_id = conversation_id()
        
        # 累积的参数
        self._arguments = ""
        # function call item
        self._item = None
    
    @property
    def arguments(self) -> str:
        """返回累积的参数"""
        return self._arguments
    
    async def __aenter__(self) -> "FunctionCallResponseContext":
        """发送前置事件"""
        # 1. Response created
        await self.session.send_event(
            build_response_created_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        
        # 2. 创建 function call item
        self._item = build_function_call_item(
            name=self.name,
            arguments="",
            call_id=self.call_id,
            item_id=self.item_id,
            status="in_progress",
        )
        
        # 3. Output item added
        await self.session.send_event(
            build_function_call_output_item_added_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_function_call_conversation_item_added_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件"""
        # 5. Function call arguments done
        await self.session.send_event(
            build_function_call_arguments_done(
                arguments=self._arguments,
                call_id=self.call_id,
                item_id=self.item_id,
                response_id=self.response_id,
                name=self.name,
            )
        )
        
        # 6. 更新 item 状态
        self._item = build_function_call_item(
            name=self.name,
            arguments=self._arguments,
            call_id=self.call_id,
            item_id=self.item_id,
            status="completed",
        )
        
        # 7. Conversation item done
        await self.session.send_event(
            build_function_call_conversation_item_done_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        # 8. Output item done
        await self.session.send_event(
            build_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 9. Response done
        await self.session.send_event(
            build_response_done_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        
        logger.info(
            f"FunctionCallResponseContext completed: item_id={self.item_id}, "
            f"response_id={self.response_id}, name={self.name}, "
            f"call_id={self.call_id}, arguments='{self._arguments}'"
        )
        
        return False  # 不抑制异常
    
    async def send_arguments_delta(self, delta: str):
        """发送参数增量"""
        if not delta:
            return
        self._arguments += delta
        await self.session.send_event(
            build_function_call_arguments_delta(
                arguments_delta=delta,
                call_id=self.call_id,
                item_id=self.item_id,
                response_id=self.response_id,
            )
        )


class McpListToolsContext:
    """
    MCP 工具列表获取上下文管理器。
    
    管理 mcp_list_tools 的事件生命周期。
    
    事件时序（与 OpenAI 官方对齐）：
    1. ConversationItemAdded (mcp_list_tools item, tools=[])
    2. mcp_list_tools.in_progress
    3. [执行工具列表获取]
    4. mcp_list_tools.completed
    5. ConversationItemDone (mcp_list_tools item, tools=[...])
    
    使用方式：
        async with McpListToolsContext(session, server_label) as ctx:
            tools = await mcp_client.list_tools()
            ctx.set_tools(tools)
    """
    
    def __init__(
        self,
        session: "RealtimeSessionState",
        server_label: str,
    ):
        self.session = session
        self.server_label = server_label
        
        # 生成 IDs
        self.item_id = item_id()
        
        # 工具列表
        self._tools: List[Dict[str, Any]] = []
        # item
        self._item = None
    
    @property
    def tools(self) -> List[Dict[str, Any]]:
        return self._tools
    
    def set_tools(self, tools: List[Dict[str, Any]]):
        """设置工具列表"""
        self._tools = tools
    
    async def __aenter__(self) -> "McpListToolsContext":
        """发送前置事件"""
        # 1. 创建 mcp_list_tools item (tools 为空)
        self._item = build_mcp_list_tools_item(
            item_id=self.item_id,
            server_label=self.server_label,
            tools=[],
        )
        
        # 2. Conversation item added
        await self.session.send_event(
            build_mcp_list_tools_added_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        # 3. mcp_list_tools.in_progress
        await self.session.send_event(
            build_mcp_list_tools_in_progress_event(
                item_id=self.item_id,
                event_id=event_id(),
            )
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件"""
        if exc_type is not None:
            await self.session.send_event(
                McpListToolsFailed(
                    type="mcp_list_tools.failed",
                    event_id=event_id(),
                    item_id=self.item_id,
                )
            )
        else:
            # 4. mcp_list_tools.completed
            await self.session.send_event(
                build_mcp_list_tools_completed_event(
                    item_id=self.item_id,
                    event_id=event_id(),
                )
            )

        # 5. 更新 item 工具列表（成功和失败都发送 done，便于客户端收敛状态）
        self._item = build_mcp_list_tools_item(
            item_id=self.item_id,
            server_label=self.server_label,
            tools=self._tools,
        )
        
        # 6. Conversation item done
        await self.session.send_event(
            build_mcp_list_tools_done_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        logger.info(
            f"McpListToolsContext completed: item_id={self.item_id}, "
            f"server_label={self.server_label}, tools_count={len(self._tools)}"
        )
        
        return False  # 不抑制异常


class McpCallResponseContext:
    """
    MCP 工具调用响应上下文管理器。
    
    管理 MCP 工具调用响应的生命周期，发送完整的事件序列。
    
    官方事件时序：
    1. ResponseCreatedEvent
    2. ResponseOutputItemAddedEvent (mcp_call item)
    3. ConversationItemAdded (mcp_call item)
    4. ResponseMcpCallArgumentsDelta (多个)
    5. ResponseMcpCallArgumentsDone
    6. ResponseDoneEvent (第一个 response，包含 mcp_call)
    7. response.mcp_call.in_progress
    8. [执行 MCP 调用]
    9. response.mcp_call.completed
    10. ConversationItemDone (mcp_call item with output)
    11. ResponseOutputItemDoneEvent
    
    使用方式：
        async with McpCallResponseContext(session, name, server_label) as ctx:
            await ctx.send_arguments_delta('{"')
            await ctx.send_arguments_delta('city')
            ...
            # 完成参数后
            await ctx.finish_arguments()
            # 执行 MCP 调用
            output = await mcp_client.call_tool(...)
            ctx.set_output(output)
    """
    
    def __init__(
        self,
        session: "RealtimeSessionState",
        name: str,
        server_label: str,
        modalities: List[str] = None,
    ):
        self.session = session
        self.name = name
        self.server_label = server_label
        self.modalities = modalities or ["text"]
        
        # 生成 IDs (MCP call ID 以 mcp_ 开头)
        self.item_id = f"mcp_{item_id().split('_', 1)[1]}"
        self.response_id = response_id()
        self.conversation_id = conversation_id()
        
        # 累积的参数
        self._arguments = ""
        # 调用结果
        self._output: str = None
        self._error: str = None
        # mcp_call item
        self._item = None
        # 是否已完成参数
        self._arguments_finished = False
    
    @property
    def arguments(self) -> str:
        return self._arguments
    
    @property
    def output(self) -> str:
        return self._output
    
    def set_output(self, output: str):
        """设置调用结果"""
        self._output = output
    
    def set_error(self, error: str):
        """设置错误信息"""
        self._error = error
    
    async def __aenter__(self) -> "McpCallResponseContext":
        """发送前置事件"""
        # 1. Response created
        await self.session.send_event(
            build_response_created_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        
        # 2. 创建 mcp_call item
        self._item = build_mcp_call_item(
            item_id=self.item_id,
            name=self.name,
            arguments="",
            server_label=self.server_label,
        )
        
        # 3. Output item added
        await self.session.send_event(
            build_mcp_call_output_item_added_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_mcp_call_conversation_item_added_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        return self
    
    async def send_arguments_delta(self, delta: str):
        """发送参数增量"""
        if not delta:
            return
        self._arguments += delta
        await self.session.send_event(
            build_mcp_call_arguments_delta_event(
                delta=delta,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
    
    async def finish_arguments(self):
        """完成参数发送，发送 arguments.done 和第一个 response.done"""
        if self._arguments_finished:
            return
        self._arguments_finished = True
        
        # 5. mcp_call_arguments.done
        await self.session.send_event(
            build_mcp_call_arguments_done_event(
                arguments=self._arguments,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        # 6. Response done (第一个 response，包含 mcp_call 但没有 output)
        await self.session.send_event(
            build_response_done_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=event_id(),
                modalities=self.modalities,
            )
        )
        
        # 7. response.mcp_call.in_progress
        await self.session.send_event(
            build_mcp_call_in_progress_event(
                item_id=self.item_id,
                event_id=event_id(),
            )
        )
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件"""
        # 确保参数已完成
        if not self._arguments_finished:
            await self.finish_arguments()
        
        if self._error or exc_type is not None:
            await self.session.send_event(
                ResponseMcpCallFailed(
                    type="response.mcp_call.failed",
                    event_id=event_id(),
                    item_id=self.item_id,
                    output_index=0,
                )
            )
        else:
            # 8. response.mcp_call.completed
            await self.session.send_event(
                build_mcp_call_completed_event(
                    item_id=self.item_id,
                    event_id=event_id(),
                )
            )
        
        # 9. 更新 item 状态和输出
        self._item = build_mcp_call_item(
            item_id=self.item_id,
            name=self.name,
            arguments=self._arguments,
            server_label=self.server_label,
            output=self._output,
            error=self._error,
        )
        
        # 10. Conversation item done
        await self.session.send_event(
            build_mcp_call_conversation_item_done_event(
                item=self._item,
                event_id=event_id(),
            )
        )
        
        # 11. Output item done
        await self.session.send_event(
            build_mcp_call_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=event_id(),
            )
        )
        
        logger.info(
            f"McpCallResponseContext completed: item_id={self.item_id}, "
            f"name={self.name}, server_label={self.server_label}, "
            f"arguments='{self._arguments}', output_length={len(self._output or '')}"
        )
        
        return False  # 不抑制异常
