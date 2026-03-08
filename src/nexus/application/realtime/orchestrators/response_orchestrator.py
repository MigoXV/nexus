from typing import Iterable, Optional
import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai.types import realtime
from openai.types.chat import ChatCompletionChunk

from nexus.infrastructure.asr import TranscriptionResult
from nexus.application.realtime.protocol.ids import event_id, item_id
from nexus.application.realtime.text_processing import (
    SanitizedModelOutputAccumulator,
    prepare_realtime_user_turn,
)
from nexus.application.realtime.emitters.response_contexts import (
    AudioResponseContext,
    FunctionCallResponseContext,
    McpCallResponseContext,
    TextResponseContext,
)

if TYPE_CHECKING:
    from nexus.domain.realtime import RealtimeSessionState
    from nexus.infrastructure.tts import TTSBackend

logger = logging.getLogger(__name__)


def get_usage_tokens(transcript: str):
    """计算转录文本的使用 token 数"""
    # 简单按空格分词计数，实际可根据具体模型的 tokenizer 实现更精确的计数
    tokens = len(transcript.strip().split())
    usage = realtime.conversation_item_input_audio_transcription_completed_event.UsageTranscriptTextUsageTokens(
        total_tokens=tokens,
        output_tokens=0,
        input_tokens=tokens,
        type="tokens",
    )
    return usage


# ---------------------------------------------------------------------------
# TranscriptionStreamTracker – 追踪流式转写状态，从累积字符串中提取增量 delta
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionStreamTracker:
    """Tracks incremental transcription state across interim ASR results.

    ASR 引擎每次返回累积后的完整字符串（如 "今天的" → "今天的天气真" → "今天的天气真好"），
    本类负责从中提取真正的增量 delta（"今天的" / "天气真" / "好"），
    以符合 OpenAI Realtime API 的 conversation.item.input_audio_transcription.delta 语义。
    """

    _previous_transcript: str = field(default="", init=False)
    _item_id: Optional[str] = field(default=None, init=False)
    _speech_started_sent: bool = field(default=False, init=False)

    @property
    def item_id(self) -> str:
        """当前语句的 item_id，首次访问时自动分配。"""
        if self._item_id is None:
            self._item_id = item_id()
        return self._item_id

    @property
    def speech_started_sent(self) -> bool:
        return self._speech_started_sent

    def mark_speech_started(self) -> None:
        self._speech_started_sent = True

    def compute_delta(self, current_transcript: str) -> str:
        """Compute the incremental delta from the previous transcript.

        If *current_transcript* starts with the previous accumulated string,
        return the new suffix.  Otherwise (ASR corrected earlier text) fall
        back to returning the full *current_transcript* and log a warning.
        """
        prev = self._previous_transcript
        if current_transcript.startswith(prev):
            delta = current_transcript[len(prev):]
        else:
            # ASR 纠正了之前的识别结果，回退到完整文本
            logger.warning(
                "ASR transcript not a prefix extension (prev=%r, cur=%r); "
                "sending full transcript as delta",
                prev,
                current_transcript,
            )
            delta = current_transcript
        self._previous_transcript = current_transcript
        return delta

    def reset(self) -> None:
        """Reset state after a final result, ready for the next utterance."""
        self._previous_transcript = ""
        self._item_id = None
        self._speech_started_sent = False


# ---------------------------------------------------------------------------
# send_transcribe_interim – 处理 is_final=False 的中间 ASR 结果
# ---------------------------------------------------------------------------

async def send_transcribe_interim(
    session: "RealtimeSessionState",
    transcription_result: TranscriptionResult,
    tracker: TranscriptionStreamTracker,
    *,
    hide_metadata: bool = True,
) -> None:
    """Send streaming delta events for an interim (non-final) ASR result."""

    # 首次收到 interim 结果时立即发送 speech_started（低延迟）
    if not tracker.speech_started_sent:
        if transcription_result.words:
            _, start_time, _ = transcription_result.words[0]
        else:
            start_time = 0.0
        vad_start_event = realtime.InputAudioBufferSpeechStartedEvent(
            audio_start_ms=int(start_time * 1000),
            type="input_audio_buffer.speech_started",
            event_id=event_id(),
            item_id=tracker.item_id,
        )
        await session.send_event(vad_start_event)
        tracker.mark_speech_started()

    # 计算增量 delta
    event_transcript = (
        prepare_realtime_user_turn(transcription_result.transcript).display_transcript
        if hide_metadata
        else transcription_result.transcript
    )
    delta = tracker.compute_delta(event_transcript)
    if not delta:
        return

    delta_event = realtime.ConversationItemInputAudioTranscriptionDeltaEvent(
        event_id=event_id(),
        item_id=tracker.item_id,
        type="conversation.item.input_audio_transcription.delta",
        content_index=0,
        delta=delta,
    )
    await session.send_event(delta_event)
    logger.debug("Sent interim delta: item_id=%s, delta=%r", tracker.item_id, delta)


# ---------------------------------------------------------------------------
# send_transcribe_response – 处理 is_final=True 的最终 ASR 结果（重构后）
# ---------------------------------------------------------------------------

async def send_transcribe_response(
    session: "RealtimeSessionState",
    transcription_result: TranscriptionResult,
    tracker: Optional[TranscriptionStreamTracker] = None,
    *,
    hide_metadata: bool = True,
):
    """Complete the transcription event sequence for a final ASR result.

    When *tracker* is provided the function cooperates with prior interim
    deltas: it reuses the same ``item_id``, skips ``speech_started`` if
    already sent, and only emits the remaining delta.

    When *tracker* is ``None`` (backward-compat / non-interim mode) the
    function behaves like the original – sends the full transcript in a
    single delta event.
    """
    is_final = transcription_result.is_final
    if not is_final:
        logger.warning(
            "send_transcribe_response called with non-final result",
        )
        return

    transcript = (
        prepare_realtime_user_turn(transcription_result.transcript).display_transcript
        if hide_metadata
        else transcription_result.transcript
    )

    # Determine item_id – reuse from tracker if available
    if tracker is not None:
        response_item_id = tracker.item_id
    else:
        response_item_id = item_id()

    if transcription_result.words:
        _, start_time, end_time = transcription_result.words[0]
    else:
        start_time = end_time = 0.0

    # speech_started – only send if not already sent by interim handler
    if tracker is None or not tracker.speech_started_sent:
        vad_start_event = realtime.InputAudioBufferSpeechStartedEvent(
            audio_start_ms=int(start_time * 1000),
            type="input_audio_buffer.speech_started",
            event_id=event_id(),
            item_id=response_item_id,
        )
        await session.send_event(vad_start_event)

    # speech_stopped
    vad_stop_event = realtime.InputAudioBufferSpeechStoppedEvent(
        audio_end_ms=int(end_time * 1000),
        type="input_audio_buffer.speech_stopped",
        event_id=event_id(),
        item_id=response_item_id,
    )
    await session.send_event(vad_stop_event)

    # committed
    committed_event = realtime.InputAudioBufferCommittedEvent(
        event_id=event_id(),
        item_id=response_item_id,
        type="input_audio_buffer.committed",
    )
    await session.send_event(committed_event)

    # Final delta – send remaining increment (or full transcript in legacy mode)
    if tracker is not None:
        delta = tracker.compute_delta(transcript)
    else:
        delta = transcript
    if delta:
        delta_event = realtime.ConversationItemInputAudioTranscriptionDeltaEvent(
            event_id=event_id(),
            item_id=response_item_id,
            type="conversation.item.input_audio_transcription.delta",
            content_index=0,
            delta=delta,
        )
        await session.send_event(delta_event)

    # completed
    completed_event = realtime.ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id=event_id(),
        item_id=response_item_id,
        transcript=transcript,
        type="conversation.item.input_audio_transcription.completed",
        usage=get_usage_tokens(transcript),
    )
    await session.send_event(completed_event)

    item = realtime.RealtimeConversationItemUserMessage(
        content=[
            realtime.realtime_conversation_item_user_message.Content(type="input_audio")
        ],
        role="user",
        type="message",
        id=response_item_id,
        object=None,
        status="completed",
    )
    conversation_add_event = realtime.ConversationItemAdded(
        event_id=event_id(), item=item, type="conversation.item.added"
    )
    await session.send_event(conversation_add_event)
    conversation_done_event = realtime.ConversationItemDone(
        event_id=event_id(), item=item, type="conversation.item.done"
    )
    await session.send_event(conversation_done_event)

    logger.info("Sent transcription response: item_id=%s, is_final=%s", response_item_id, is_final)

    # Reset tracker for the next utterance
    if tracker is not None:
        tracker.reset()


@dataclass
class ToolCallInfo:
    """工具调用信息"""
    call_id: str
    name: str
    arguments: str
    is_mcp: bool = False  # 是否为 MCP 工具调用
    server_label: Optional[str] = None  # MCP 服务器标签
    mcp_ctx: Optional["McpCallResponseContext"] = None  # MCP 上下文（用于后续事件发送）


@dataclass 
class ChatStreamResult:
    """聊天流式响应结果"""
    content: str = ""
    raw_content: str = ""
    tts_text: str = ""
    tool_call: Optional[ToolCallInfo] = None
    was_cancelled: bool = False  # 是否被打断
    
    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None
    
    @property
    def has_mcp_call(self) -> bool:
        return self.tool_call is not None and self.tool_call.is_mcp


def _modalities_or_default(modalities: Optional[list[str]]) -> list[str]:
    return list(modalities) if modalities else ["text"]


def _is_audio_mode(modalities: list[str]) -> bool:
    return "audio" in modalities


async def process_chat_stream(
    session: "RealtimeSessionState",
    chat_stream: Iterable[ChatCompletionChunk],
    *,
    modalities: Optional[list[str]] = None,
    tts_backend: Optional["TTSBackend"] = None,
    audio_output_format_type: str = "audio/pcm",
    audio_output_voice: str = "alloy",
    audio_output_speed: float = 1.0,
) -> ChatStreamResult:
    """
    处理 chat 流式响应，同时流式发送文本给客户端。
    
    此函数会立即将文本 delta 发送给客户端，
    实现真正的流式响应，降低首字延迟。
    
    返回 ChatStreamResult，包含完整文本内容或工具调用信息。
    
    事件时序（与 OpenAI 官方对齐）：
    
    文本响应：
    1. ResponseCreatedEvent
    2. ResponseOutputItemAddedEvent
    3. ConversationItemAdded
    4. ResponseContentPartAddedEvent
    5. ResponseOutputTextDeltaEvent (多个)
    6. ResponseOutputTextDoneEvent
    7. ResponseContentPartDoneEvent
    8. ResponseOutputItemDoneEvent
    9. ConversationItemDone
    10. ResponseDoneEvent
    
    工具调用：
    1. ResponseCreatedEvent
    2. ResponseOutputItemAddedEvent  
    3. ConversationItemAdded
    4. ResponseFunctionCallArgumentsDeltaEvent (多个)
    5. ResponseFunctionCallArgumentsDoneEvent
    6. ConversationItemDone
    7. ResponseOutputItemDoneEvent
    8. ResponseDoneEvent
    """
    active_modalities = _modalities_or_default(modalities)
    audio_mode = _is_audio_mode(active_modalities)

    result = ChatStreamResult()
    text_ctx: Optional[TextResponseContext] = None
    audio_ctx: Optional[AudioResponseContext] = None
    func_ctx: Optional[FunctionCallResponseContext] = None
    mcp_ctx: Optional[McpCallResponseContext] = None
    
    # 用于累积工具调用参数
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    is_mcp_tool: bool = False
    mcp_server_label: Optional[str] = None
    sanitizer = SanitizedModelOutputAccumulator()
    
    try:
        async for chunk in chat_stream:
            # 🔴 检查是否需要取消（新转写事件到来）
            if session.is_cancel_requested():
                logger.info("Chat stream cancelled due to new transcription")
                result.was_cancelled = True
                break
            
            delta = chunk.choices[0].delta
            
            # 处理工具调用
            if delta.tool_calls:
                tool_call = delta.tool_calls[0]
                function = tool_call.function
                
                # 首次出现工具调用名称，判断是否为 MCP 工具并创建对应上下文
                if function.name:
                    tool_name = function.name
                    tool_call_id = tool_call.id
                    
                    # 检查是否为 MCP 工具
                    is_mcp_tool = session.is_mcp_tool(tool_name)
                    
                    if is_mcp_tool:
                        mcp_server_label = session.get_mcp_server_for_tool(tool_name)
                        mcp_ctx = McpCallResponseContext(
                            session=session,
                            name=tool_name,
                            server_label=mcp_server_label,
                            modalities=active_modalities,
                        )
                        await mcp_ctx.__aenter__()
                        if function.arguments:
                            await mcp_ctx.send_arguments_delta(function.arguments)
                    else:
                        # 普通 function call
                        func_ctx = FunctionCallResponseContext(
                            session=session,
                            name=tool_name,
                            call_id=tool_call_id,
                            modalities=active_modalities,
                        )
                        await func_ctx.__aenter__()
                        if function.arguments:
                            await func_ctx.send_arguments_delta(function.arguments)
                elif function.arguments:
                    # 后续参数增量
                    if mcp_ctx:
                        await mcp_ctx.send_arguments_delta(function.arguments)
                    elif func_ctx:
                        await func_ctx.send_arguments_delta(function.arguments)
            
            # 🚀 流式发送文本内容
            if delta.content:
                result.raw_content += delta.content
                display_delta, tts_delta = sanitizer.push(delta.content)
                result.content = sanitizer.display_text
                result.tts_text = sanitizer.tts_text

                if audio_mode:
                    if not display_delta and not tts_delta:
                        continue
                    if audio_ctx is None:
                        if tts_backend is None:
                            raise RuntimeError("TTS backend is not configured for audio output")
                        audio_ctx = AudioResponseContext(
                            session,
                            tts_backend=tts_backend,
                            modalities=active_modalities,
                            format_type=audio_output_format_type,
                            voice=audio_output_voice,
                            speed=audio_output_speed,
                        )
                        await audio_ctx.__aenter__()
                    await audio_ctx.add_model_text_delta(display_delta, tts_delta=tts_delta)
                else:
                    if not display_delta:
                        continue
                    # 延迟创建上下文，在第一个文本到达时才发送前置事件
                    if text_ctx is None:
                        text_ctx = TextResponseContext(session, modalities=active_modalities)
                        await text_ctx.__aenter__()
                    await text_ctx.send_text_delta(display_delta)
        
        # 流结束后，如果有工具调用，记录结果
        if mcp_ctx and tool_call_id:
            # MCP 工具调用 - 完成参数发送阶段
            await mcp_ctx.finish_arguments()
            
            result.tool_call = ToolCallInfo(
                call_id=tool_call_id,
                name=tool_name or "",
                arguments=mcp_ctx.arguments,
                is_mcp=True,
                server_label=mcp_server_label,
                mcp_ctx=mcp_ctx,  # 传递上下文给 servicer
            )
            logger.info(
                f"MCP tool call detected: name={tool_name}, "
                f"server_label={mcp_server_label}, arguments={mcp_ctx.arguments}"
            )
            # 注意：mcp_ctx 不在这里关闭，由 servicer 在执行调用后关闭
        elif func_ctx and tool_call_id:
            # 普通 function call
            result.tool_call = ToolCallInfo(
                call_id=tool_call_id,
                name=tool_name or "",
                arguments=func_ctx.arguments,
                is_mcp=False,
            )
            logger.info(
                f"Function call detected: name={tool_name}, call_id={tool_call_id}, "
                f"arguments={func_ctx.arguments}"
            )
        elif result.content:
            logger.info(f"Chat stream response sent: content='{result.content}'")
    
    except asyncio.CancelledError:
        # 任务被真正取消（Task.cancel()）
        logger.info("Chat stream task was cancelled by CancelledError")
        result.was_cancelled = True
        # 显式关闭生成器，停止底层 HTTP 流
        if hasattr(chat_stream, 'aclose'):
            try:
                await chat_stream.aclose()
            except Exception as e:
                logger.debug(f"Error closing chat stream: {e}")
        raise  # 重新抛出让调用者知道任务被取消
    
    finally:
        # 确保上下文正确关闭，发送后置事件
        audio_synthesis_error: Optional[Exception] = None
        if audio_ctx is not None:
            should_synthesize = (
                not result.was_cancelled
                and not result.has_tool_call
                and bool(result.tts_text.strip())
            )
            if should_synthesize:
                try:
                    await audio_ctx.synthesize_audio()
                except Exception as exc:  # pragma: no cover - defensive boundary
                    audio_synthesis_error = exc
                    logger.error("Audio synthesis failed: %s", exc)

            await audio_ctx.finish(
                cancelled=result.was_cancelled,
                failed=audio_synthesis_error is not None,
                error_code="audio_synthesis_failed" if audio_synthesis_error else None,
                error_type="server_error" if audio_synthesis_error else None,
            )
            if audio_synthesis_error and hasattr(session, "writer"):
                await session.writer.send_error(
                    message=f"Audio synthesis failed: {audio_synthesis_error}",
                    error_type="server_error",
                    code="audio_synthesis_failed",
                )

        if text_ctx is not None:
            await text_ctx.finish(cancelled=result.was_cancelled)
        if func_ctx is not None:
            await func_ctx.__aexit__(None, None, None)
        # 注意：MCP 上下文需要在执行调用后关闭，这里不关闭
        
        # 🔴 如果被取消，手动将部分内容添加到历史记录
        # 正常结束时，chat_session.get_result_record_itr 会自动处理
        # 但被取消时流不会正常结束，需要手动添加
        if result.was_cancelled and result.content:
            from openai.types.chat import ChatCompletionMessage
            cancelled_message = ChatCompletionMessage(
                role="assistant",
                content=result.content,  # 保存已生成的部分内容
                tool_calls=[],
            )
            session.chat_session.chat_history.append(cancelled_message)
            logger.info(
                f"Cancelled chat partial content saved to history: '{result.content}'"
            )
        elif not result.was_cancelled and (result.raw_content or result.has_tool_call):
            session.chat_session.replace_last_assistant_message_content(result.content)
    
    return result


async def send_tool_result_response(
    session: "RealtimeSessionState",
    chat_stream: Iterable[ChatCompletionChunk],
    *,
    modalities: Optional[list[str]] = None,
    tts_backend: Optional["TTSBackend"] = None,
    audio_output_format_type: str = "audio/pcm",
    audio_output_voice: str = "alloy",
    audio_output_speed: float = 1.0,
):
    """
    发送工具调用结果后的响应流。
    使用与主对话一致的响应上下文发送完整事件序列。
    """
    result = await process_chat_stream(
        session=session,
        chat_stream=chat_stream,
        modalities=modalities,
        tts_backend=tts_backend,
        audio_output_format_type=audio_output_format_type,
        audio_output_voice=audio_output_voice,
        audio_output_speed=audio_output_speed,
    )

    logger.info("Tool result response sent: content='%s'", result.content)


async def send_text_response(session: "RealtimeSessionState", content: str):
    """发送纯文本响应（使用上下文管理器）"""
    async with TextResponseContext(session) as ctx:
        await ctx.send_text_delta(content)


async def send_chat_stream_response(
    session: "RealtimeSessionState",
    response_chunk: Iterable[str],
):
    """发送流式聊天响应（使用上下文管理器）"""
    async with TextResponseContext(session) as ctx:
        async for chunk in response_chunk:
            await ctx.send_text_delta(chunk)
