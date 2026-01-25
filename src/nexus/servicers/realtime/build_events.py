import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeResponse,
    RealtimeConversationItemFunctionCall,
    RealtimeSessionCreateRequest,
    ResponseAudioDeltaEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageTokens,
)

logger = logging.getLogger(__name__)


def build_input_audio_transcription_completed(transcript: str):
    event_id = str(uuid.uuid4())
    item_id = str(uuid.uuid4())

    event = ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id=event_id,
        item_id=item_id,
        transcript=transcript,
        type="conversation.item.input_audio_transcription.completed",
        usage=UsageTranscriptTextUsageTokens(
            input_tokens=0,
            output_tokens=len(transcript),
            total_tokens=len(transcript),
            type="tokens",
        ),
    )
    return event


def build_session_created(session_id: str, model: str) -> dict:
    """构建 session.created 事件"""
    return {
        "type": "session.created",
        "session": {
            "id": session_id,
            "model": model,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        },
    }


def build_session_updated(
    session_id: str, model: str, output_modalities: List[str]
) -> dict:
    """构建 session.updated 事件"""
    return {
        "type": "session.updated",
        "session": {
            "id": session_id,
            "model": model,
            "output_modalities": output_modalities,
        },
    }


def build_error_event(error_type: str, message: str) -> dict:
    """构建 error 事件"""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def build_response_audio_delta(
    response_id: str, item_id: str, audio_delta: str
) -> dict:
    """构建 response.output_audio.delta 事件"""
    return {
        "type": "response.output_audio.delta",
        "response_id": "0",
        "item_id": "0",
        "delta": audio_delta,
    }


def build_response_audio_done(response_id: str, item_id: str) -> dict:
    """构建 response.output_audio.done 事件"""
    return {
        "type": "response.output_audio.done",
        "response_id": "0",
        "item_id": "0",
    }


def build_response_done(response_id: str) -> dict:
    """构建 response.done 事件"""
    return {
        "type": "response.done",
        "response_id": "0",
    }


def build_response_text_delta(text_delta: str):
    """构建 response.text.delta 事件"""
    event = ResponseTextDeltaEvent(
        content_index=0,
        delta=text_delta,
        event_id=str(uuid.uuid4()),
        item_id="0",
        output_index=0,
        response_id="0",
        type="response.output_text.delta",
    )
    return event


def build_response_text_done(text: str):
    """构建 response.text.done 事件"""
    event = ResponseTextDoneEvent(
        content_index=0,
        event_id=str(uuid.uuid4()),
        item_id="0",
        output_index=0,
        response_id="0",
        text=text,
        type="response.output_text.done",
    )
    return event


def build_item_function_call(
    name: str,
    arguments: str,
    call_id: str,
) -> RealtimeConversationItemFunctionCall:
    """构建 RealtimeConversationItemFunctionCall 对象"""
    return ResponseOutputItemAddedEvent(
        event_id=str(uuid.uuid4()),
        item=RealtimeConversationItemFunctionCall(
            name=name,
            arguments=arguments,
            type="function_call",
            call_id=call_id,
        ),
        output_index=0,
        response_id="0",
        type="response.output_item.added",
    )


def build_function_call_arguments_delta(
    arguments_delta: str,
    call_id: str,
    item_id: str = "0",
    response_id: str = "0",
) -> ResponseFunctionCallArgumentsDeltaEvent:
    """构建 response.function_call.arguments.delta 事件"""
    event = ResponseFunctionCallArgumentsDeltaEvent(
        event_id=str(uuid.uuid4()),
        call_id=call_id,
        item_id=item_id,
        output_index=0,
        response_id=response_id,
        delta=arguments_delta,
        type="response.function_call_arguments.delta",
    )
    return event


def build_function_call_arguments_done(
    arguments: str,
    call_id: str,
    item_id: str = "0",
    response_id: str = "0",
    name: str = None,
) -> ResponseFunctionCallArgumentsDoneEvent:
    """构建 response.function_call.arguments.done 事件"""
    event = ResponseFunctionCallArgumentsDoneEvent(
        event_id=str(uuid.uuid4()),
        call_id=call_id,
        item_id=item_id,
        output_index=0,
        arguments=arguments,
        response_id=response_id,
        type="response.function_call_arguments.done",
        name=name,
    )
    return event


# ============ Response lifecycle events ============

from openai.types.realtime import (
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ConversationItemAdded,
    ConversationItemDone,
    conversation_item,
    response_content_part_added_event,
    response_content_part_done_event,
    realtime_conversation_item_assistant_message,
)


def build_response_created_event(
    response_id: str,
    conversation_id: str,
    event_id: str,
    modalities: List[str] = None,
) -> ResponseCreatedEvent:
    """构建 response.created 事件"""
    if modalities is None:
        modalities = ["text"]
    return ResponseCreatedEvent(
        event_id=event_id,
        response=RealtimeResponse(
            id=response_id,
            conversation_id=conversation_id,
            max_output_tokens="inf",
            metadata=None,
            object="realtime.response",
            output=[],
            output_modalities=modalities,
            status="in_progress",
            status_details=None,
            usage=None,
        ),
        type="response.created",
    )


def build_response_done_event(
    response_id: str,
    conversation_id: str,
    event_id: str,
    modalities: List[str] = None,
) -> ResponseDoneEvent:
    """构建 response.done 事件"""
    if modalities is None:
        modalities = ["text"]
    return ResponseDoneEvent(
        event_id=event_id,
        response=RealtimeResponse(
            id=response_id,
            conversation_id=conversation_id,
            max_output_tokens="inf",
            metadata=None,
            object="realtime.response",
            output=[],
            output_modalities=modalities,
            status="completed",
            status_details=None,
            usage=None,
        ),
        type="response.done",
    )


def build_response_cancelled_event(
    response_id: str,
    conversation_id: str,
    event_id: str,
    modalities: List[str] = None,
) -> dict:
    """
    构建 response.cancelled 事件（当 LLM 生成被打断时发送）。
    
    注意：OpenAI SDK 没有定义 ResponseCancelledEvent 类型，
    因此返回字典格式，与 OpenAI Realtime API 保持兼容。
    """
    if modalities is None:
        modalities = ["text"]
    return {
        "type": "response.cancelled",
        "event_id": event_id,
        "response": {
            "id": response_id,
            "conversation_id": conversation_id,
            "max_output_tokens": "inf",
            "metadata": None,
            "object": "realtime.response",
            "output": [],
            "output_modalities": modalities,
            "status": "cancelled",
            "status_details": {
                "type": "cancelled",
                "reason": "turn_detected",  # 检测到新的用户输入
            },
            "usage": None,
        },
    }


def build_assistant_message_item(
    item_id: str,
    status: str = "in_progress",
) -> conversation_item.RealtimeConversationItemAssistantMessage:
    """构建助手消息 item"""
    return conversation_item.RealtimeConversationItemAssistantMessage(
        content=[],
        role="assistant",
        type="message",
        id=item_id,
        object=None,
        status=status,
    )


def build_output_item_added_event(
    item: conversation_item.RealtimeConversationItemAssistantMessage,
    response_id: str,
    event_id: str,
) -> ResponseOutputItemAddedEvent:
    """构建 response.output_item.added 事件"""
    return ResponseOutputItemAddedEvent(
        event_id=event_id,
        item=item,
        output_index=0,
        response_id=response_id,
        type="response.output_item.added",
    )


def build_conversation_item_added_event(
    item: conversation_item.RealtimeConversationItemAssistantMessage,
    event_id: str,
) -> ConversationItemAdded:
    """构建 conversation.item.added 事件"""
    return ConversationItemAdded(
        event_id=event_id,
        item=item,
        type="conversation.item.added",
    )


def build_content_part_added_event(
    item_id: str,
    response_id: str,
    event_id: str,
) -> ResponseContentPartAddedEvent:
    """构建 response.content_part.added 事件"""
    return ResponseContentPartAddedEvent(
        content_index=0,
        event_id=event_id,
        item_id=item_id,
        output_index=0,
        part=response_content_part_added_event.Part(
            audio=None, text="", transcript=None, type="text"
        ),
        response_id=response_id,
        type="response.content_part.added",
    )


def build_text_delta_event(
    delta: str,
    item_id: str,
    response_id: str,
    event_id: str,
) -> ResponseTextDeltaEvent:
    """构建 response.output_text.delta 事件"""
    return ResponseTextDeltaEvent(
        content_index=0,
        delta=delta,
        event_id=event_id,
        item_id=item_id,
        output_index=0,
        response_id=response_id,
        type="response.output_text.delta",
    )


def build_text_done_event(
    text: str,
    item_id: str,
    response_id: str,
    event_id: str,
) -> ResponseTextDoneEvent:
    """构建 response.output_text.done 事件"""
    return ResponseTextDoneEvent(
        content_index=0,
        event_id=event_id,
        item_id=item_id,
        output_index=0,
        response_id=response_id,
        text=text,
        type="response.output_text.done",
    )


def build_content_part_done_event(
    text: str,
    item_id: str,
    response_id: str,
    event_id: str,
) -> ResponseContentPartDoneEvent:
    """构建 response.content_part.done 事件"""
    return ResponseContentPartDoneEvent(
        content_index=0,
        event_id=event_id,
        item_id=item_id,
        output_index=0,
        part=response_content_part_done_event.Part(
            audio=None,
            text=text,
            transcript=None,
            type="text",
        ),
        response_id=response_id,
        type="response.content_part.done",
    )


def build_conversation_item_done_event(
    item: conversation_item.RealtimeConversationItemAssistantMessage,
    event_id: str,
) -> ConversationItemDone:
    """构建 conversation.item.done 事件"""
    return ConversationItemDone(
        event_id=event_id,
        item=item,
        type="conversation.item.done",
    )


def build_output_item_done_event(
    item,
    response_id: str,
    event_id: str,
) -> ResponseOutputItemDoneEvent:
    """构建 response.output_item.done 事件"""
    return ResponseOutputItemDoneEvent(
        event_id=event_id,
        item=item,
        output_index=0,
        response_id=response_id,
        type="response.output_item.done",
    )


# ============ Function Call specific events ============

def build_function_call_item(
    name: str,
    arguments: str,
    call_id: str,
    item_id: str,
    status: str = "in_progress",
) -> RealtimeConversationItemFunctionCall:
    """构建 function call item"""
    return RealtimeConversationItemFunctionCall(
        name=name,
        arguments=arguments,
        type="function_call",
        id=item_id,
        call_id=call_id,
        object=None,
        status=status,
    )


def build_function_call_output_item_added_event(
    item: RealtimeConversationItemFunctionCall,
    response_id: str,
    event_id: str,
) -> ResponseOutputItemAddedEvent:
    """构建 function call 的 response.output_item.added 事件"""
    return ResponseOutputItemAddedEvent(
        event_id=event_id,
        item=item,
        output_index=0,
        response_id=response_id,
        type="response.output_item.added",
    )


def build_function_call_conversation_item_added_event(
    item: RealtimeConversationItemFunctionCall,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> ConversationItemAdded:
    """构建 function call 的 conversation.item.added 事件"""
    return ConversationItemAdded(
        event_id=event_id,
        item=item,
        type="conversation.item.added",
        previous_item_id=previous_item_id,
    )


def build_function_call_conversation_item_done_event(
    item: RealtimeConversationItemFunctionCall,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> ConversationItemDone:
    """构建 function call 的 conversation.item.done 事件"""
    return ConversationItemDone(
        event_id=event_id,
        item=item,
        type="conversation.item.done",
        previous_item_id=previous_item_id,
    )


def build_conversation_item_created_event(
    item_id: str,
    item_type: str = "function_call_output",
) -> dict:
    """
    构建 conversation.item.created 事件
    用于确认客户端发送的 conversation.item.create 已被接收和处理
    """
    return {
        "type": "conversation.item.created",
        "event_id": str(uuid.uuid4()),
        "item": {
            "id": item_id,
            "type": item_type,
            "status": "completed",
        },
    }


# ============ MCP specific events ============

@dataclass
class McpListToolsItem:
    """MCP 工具列表 item"""
    id: str
    server_label: str
    tools: List[dict]
    type: str = "mcp_list_tools"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "server_label": self.server_label,
            "tools": self.tools,
        }


@dataclass
class McpCallItem:
    """MCP 工具调用 item"""
    id: str
    name: str
    arguments: str
    server_label: str
    output: Optional[str] = None
    error: Optional[str] = None
    approval_request_id: Optional[str] = None
    type: str = "mcp_call"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "arguments": self.arguments,
            "server_label": self.server_label,
            "output": self.output,
            "error": self.error,
            "approval_request_id": self.approval_request_id,
        }


def build_mcp_list_tools_item(
    item_id: str,
    server_label: str,
    tools: List[dict] = None,
) -> McpListToolsItem:
    """构建 mcp_list_tools item"""
    return McpListToolsItem(
        id=item_id,
        server_label=server_label,
        tools=tools or [],
    )


def build_mcp_list_tools_added_event(
    item: McpListToolsItem,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> dict:
    """构建 mcp_list_tools 的 conversation.item.added 事件"""
    return {
        "type": "conversation.item.added",
        "event_id": event_id,
        "item": item.to_dict(),
        "previous_item_id": previous_item_id,
    }


def build_mcp_list_tools_in_progress_event(
    item_id: str,
    event_id: str,
) -> dict:
    """构建 mcp_list_tools.in_progress 事件"""
    return {
        "type": "mcp_list_tools.in_progress",
        "event_id": event_id,
        "item_id": item_id,
    }


def build_mcp_list_tools_completed_event(
    item_id: str,
    event_id: str,
) -> dict:
    """构建 mcp_list_tools.completed 事件"""
    return {
        "type": "mcp_list_tools.completed",
        "event_id": event_id,
        "item_id": item_id,
    }


def build_mcp_list_tools_done_event(
    item: McpListToolsItem,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> dict:
    """构建 mcp_list_tools 的 conversation.item.done 事件"""
    return {
        "type": "conversation.item.done",
        "event_id": event_id,
        "item": item.to_dict(),
        "previous_item_id": previous_item_id,
    }


def build_mcp_call_item(
    item_id: str,
    name: str,
    arguments: str,
    server_label: str,
    output: Optional[str] = None,
    error: Optional[str] = None,
) -> McpCallItem:
    """构建 mcp_call item"""
    return McpCallItem(
        id=item_id,
        name=name,
        arguments=arguments,
        server_label=server_label,
        output=output,
        error=error,
    )


def build_mcp_call_output_item_added_event(
    item: McpCallItem,
    response_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 mcp_call 的 response.output_item.added 事件"""
    return {
        "type": "response.output_item.added",
        "event_id": event_id,
        "item": item.to_dict(),
        "output_index": output_index,
        "response_id": response_id,
    }


def build_mcp_call_conversation_item_added_event(
    item: McpCallItem,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> dict:
    """构建 mcp_call 的 conversation.item.added 事件"""
    return {
        "type": "conversation.item.added",
        "event_id": event_id,
        "item": item.to_dict(),
        "previous_item_id": previous_item_id,
    }


def build_mcp_call_arguments_delta_event(
    delta: str,
    item_id: str,
    response_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 response.mcp_call_arguments.delta 事件"""
    return {
        "type": "response.mcp_call_arguments.delta",
        "event_id": event_id,
        "delta": delta,
        "item_id": item_id,
        "output_index": output_index,
        "response_id": response_id,
    }


def build_mcp_call_arguments_done_event(
    arguments: str,
    item_id: str,
    response_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 response.mcp_call_arguments.done 事件"""
    return {
        "type": "response.mcp_call_arguments.done",
        "event_id": event_id,
        "arguments": arguments,
        "item_id": item_id,
        "output_index": output_index,
        "response_id": response_id,
    }


def build_mcp_call_in_progress_event(
    item_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 response.mcp_call.in_progress 事件"""
    return {
        "type": "response.mcp_call.in_progress",
        "event_id": event_id,
        "item_id": item_id,
        "output_index": output_index,
    }


def build_mcp_call_completed_event(
    item_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 response.mcp_call.completed 事件"""
    return {
        "type": "response.mcp_call.completed",
        "event_id": event_id,
        "item_id": item_id,
        "output_index": output_index,
    }


def build_mcp_call_output_item_done_event(
    item: McpCallItem,
    response_id: str,
    event_id: str,
    output_index: int = 0,
) -> dict:
    """构建 mcp_call 的 response.output_item.done 事件"""
    return {
        "type": "response.output_item.done",
        "event_id": event_id,
        "item": item.to_dict(),
        "output_index": output_index,
        "response_id": response_id,
    }


def build_mcp_call_conversation_item_done_event(
    item: McpCallItem,
    event_id: str,
    previous_item_id: Optional[str] = None,
) -> dict:
    """构建 mcp_call 的 conversation.item.done 事件"""
    return {
        "type": "conversation.item.done",
        "event_id": event_id,
        "item": item.to_dict(),
        "previous_item_id": previous_item_id,
    }
