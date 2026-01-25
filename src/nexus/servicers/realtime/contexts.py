"""
响应上下文管理器模块。

提供用于管理响应生命周期的上下文管理器，自动处理前置/后置事件发送。
"""
from typing import List, TYPE_CHECKING, Dict, Any
import logging

from .utils import get_event_id, get_item_id, get_response_id, get_conversation_id
from .build_events import (
    build_response_created_event,
    build_response_done_event,
    build_response_cancelled_event,
    build_assistant_message_item,
    build_output_item_added_event,
    build_conversation_item_added_event,
    build_content_part_added_event,
    build_text_delta_event,
    build_text_done_event,
    build_content_part_done_event,
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
    from nexus.sessions import RealtimeSession

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
        session: "RealtimeSession",
        modalities: List[str] = None,
    ):
        self.session = session
        self.modalities = modalities or ["text"]
        
        # 生成 IDs
        self.item_id = get_item_id()
        self.response_id = get_response_id()
        self.conversation_id = get_conversation_id()
        
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
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_conversation_item_added_event(
                item=self._item,
                event_id=get_event_id(),
            )
        )
        
        # 5. Content part added
        await self.session.send_event(
            build_content_part_added_event(
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 7. Content part done
        await self.session.send_event(
            build_content_part_done_event(
                text=self._content,
                item_id=self.item_id,
                response_id=self.response_id,
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 10. Conversation item done
        await self.session.send_event(
            build_conversation_item_done_event(
                item=self._item,
                event_id=get_event_id(),
            )
        )
        
        # 11. Response done 或 Response cancelled
        if cancelled:
            await self.session.send_event(
                build_response_cancelled_event(
                    response_id=self.response_id,
                    conversation_id=self.conversation_id,
                    event_id=get_event_id(),
                    modalities=self.modalities,
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
                    event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )


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
        session: "RealtimeSession",
        name: str,
        call_id: str,
        modalities: List[str] = None,
    ):
        self.session = session
        self.name = name
        self.call_id = call_id
        self.modalities = modalities or ["text"]
        
        # 生成 IDs
        self.item_id = get_item_id()
        self.response_id = get_response_id()
        self.conversation_id = get_conversation_id()
        
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
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_function_call_conversation_item_added_event(
                item=self._item,
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 8. Output item done
        await self.session.send_event(
            build_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=get_event_id(),
            )
        )
        
        # 9. Response done
        await self.session.send_event(
            build_response_done_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=get_event_id(),
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
        session: "RealtimeSession",
        server_label: str,
    ):
        self.session = session
        self.server_label = server_label
        
        # 生成 IDs
        self.item_id = get_item_id()
        
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
                event_id=get_event_id(),
            )
        )
        
        # 3. mcp_list_tools.in_progress
        await self.session.send_event(
            build_mcp_list_tools_in_progress_event(
                item_id=self.item_id,
                event_id=get_event_id(),
            )
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件"""
        # 4. mcp_list_tools.completed
        await self.session.send_event(
            build_mcp_list_tools_completed_event(
                item_id=self.item_id,
                event_id=get_event_id(),
            )
        )
        
        # 5. 更新 item 工具列表
        self._item = build_mcp_list_tools_item(
            item_id=self.item_id,
            server_label=self.server_label,
            tools=self._tools,
        )
        
        # 6. Conversation item done
        await self.session.send_event(
            build_mcp_list_tools_done_event(
                item=self._item,
                event_id=get_event_id(),
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
        session: "RealtimeSession",
        name: str,
        server_label: str,
        modalities: List[str] = None,
    ):
        self.session = session
        self.name = name
        self.server_label = server_label
        self.modalities = modalities or ["text"]
        
        # 生成 IDs (MCP call ID 以 mcp_ 开头)
        self.item_id = f"mcp_{get_item_id().split('_', 1)[1]}"
        self.response_id = get_response_id()
        self.conversation_id = get_conversation_id()
        
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
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 4. Conversation item added
        await self.session.send_event(
            build_mcp_call_conversation_item_added_event(
                item=self._item,
                event_id=get_event_id(),
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
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 6. Response done (第一个 response，包含 mcp_call 但没有 output)
        await self.session.send_event(
            build_response_done_event(
                response_id=self.response_id,
                conversation_id=self.conversation_id,
                event_id=get_event_id(),
                modalities=self.modalities,
            )
        )
        
        # 7. response.mcp_call.in_progress
        await self.session.send_event(
            build_mcp_call_in_progress_event(
                item_id=self.item_id,
                event_id=get_event_id(),
            )
        )
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """发送后置事件"""
        # 确保参数已完成
        if not self._arguments_finished:
            await self.finish_arguments()
        
        # 8. response.mcp_call.completed
        await self.session.send_event(
            build_mcp_call_completed_event(
                item_id=self.item_id,
                event_id=get_event_id(),
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
                event_id=get_event_id(),
            )
        )
        
        # 11. Output item done
        await self.session.send_event(
            build_mcp_call_output_item_done_event(
                item=self._item,
                response_id=self.response_id,
                event_id=get_event_id(),
            )
        )
        
        logger.info(
            f"McpCallResponseContext completed: item_id={self.item_id}, "
            f"name={self.name}, server_label={self.server_label}, "
            f"arguments='{self._arguments}', output_length={len(self._output or '')}"
        )
        
        return False  # 不抑制异常
