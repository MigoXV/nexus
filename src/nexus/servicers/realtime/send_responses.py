from typing import Iterable, Optional, List
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai.types import realtime
from openai.types.chat import ChatCompletionChunk

from nexus.inferencers.asr.inferencer import TranscriptionResult

from .utils import get_event_id, get_item_id, get_response_id, get_conversation_id
from .contexts import TextResponseContext, FunctionCallResponseContext, McpCallResponseContext
from .build_events import (
    build_response_text_delta,
    build_response_text_done,
)

if TYPE_CHECKING:
    from nexus.sessions import RealtimeSession

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


async def send_transcribe_response(
    session: "RealtimeSession",
    transcription_result: TranscriptionResult,
):
    item_id = get_item_id()
    is_final = transcription_result.is_final
    if not is_final:
        logger.warning(
            f"send_transcribe_response called with non-final result, item_id={item_id}"
        )
        return
    transcript = transcription_result.transcript
    _, start_time, end_time = transcription_result.words[0]
    vad_start_event = realtime.InputAudioBufferSpeechStartedEvent(
        audio_start_ms=int(start_time * 1000),
        type="input_audio_buffer.speech_started",
        event_id=get_event_id(),
        item_id=item_id,
    )
    await session.send_event(vad_start_event)
    vad_stop_event = realtime.InputAudioBufferSpeechStoppedEvent(
        audio_end_ms=int(end_time * 1000),
        type="input_audio_buffer.speech_stopped",
        event_id=get_event_id(),
        item_id=item_id,
    )
    await session.send_event(vad_stop_event)
    committed_event = realtime.InputAudioBufferCommittedEvent(
        event_id=get_event_id(),
        item_id=item_id,
        type="input_audio_buffer.committed",
    )
    await session.send_event(committed_event)
    delta_event = realtime.ConversationItemInputAudioTranscriptionDeltaEvent(
        event_id=get_event_id(),
        item_id=item_id,
        type="conversation.item.input_audio_transcription.delta",
        content_index=0,
        delta=transcript,
    )
    await session.send_event(delta_event)
    completed_event = realtime.ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id=get_event_id(),
        item_id=item_id,
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
        id=item_id,
        object=None,
        status="completed",
    )
    conversation_add_event = realtime.ConversationItemAdded(
        event_id=get_event_id(), item=item, type="conversation.item.added"
    )
    await session.send_event(conversation_add_event)
    conversation_done_event = realtime.ConversationItemDone(
        event_id=get_event_id(), item=item, type="conversation.item.done"
    )
    await session.send_event(conversation_done_event)

    logger.info(
        f"Sent transcription response: item_id={item_id}, is_final={is_final}, transcript='{transcript}'"
    )


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
    tool_call: Optional[ToolCallInfo] = None
    was_cancelled: bool = False  # 是否被打断
    
    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None
    
    @property
    def has_mcp_call(self) -> bool:
        return self.tool_call is not None and self.tool_call.is_mcp


async def process_chat_stream(
    session: "RealtimeSession",
    chat_stream: Iterable[ChatCompletionChunk],
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
    result = ChatStreamResult()
    text_ctx: Optional[TextResponseContext] = None
    func_ctx: Optional[FunctionCallResponseContext] = None
    mcp_ctx: Optional[McpCallResponseContext] = None
    
    # 用于累积工具调用参数
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    is_mcp_tool: bool = False
    mcp_server_label: Optional[str] = None
    
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
                # 延迟创建上下文，在第一个文本到达时才发送前置事件
                if text_ctx is None:
                    text_ctx = TextResponseContext(session)
                    await text_ctx.__aenter__()
                
                result.content += delta.content
                await text_ctx.send_text_delta(delta.content)
        
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
    
    return result


async def send_tool_result_response(
    session: "RealtimeSession",
    chat_stream: Iterable[ChatCompletionChunk],
):
    """
    发送工具调用结果后的响应流。
    使用 TextResponseContext 发送完整的事件序列（包括 response.created 等前置事件）。
    """
    async with TextResponseContext(session) as ctx:
        async for chunk in chat_stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                await ctx.send_text_delta(delta.content)
    
    logger.info(f"Tool result response sent: content='{ctx.content}'")


async def send_text_response(session: "RealtimeSession", content: str):
    """发送纯文本响应（使用上下文管理器）"""
    async with TextResponseContext(session) as ctx:
        await ctx.send_text_delta(content)


async def send_chat_stream_response(
    session: "RealtimeSession",
    response_chunk: Iterable[str],
):
    """发送流式聊天响应（使用上下文管理器）"""
    async with TextResponseContext(session) as ctx:
        async for chunk in response_chunk:
            await ctx.send_text_delta(chunk)
