import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import Task

import numpy as np
from fastapi import WebSocket
from openai.types.chat import ChatCompletionChunk
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from nexus.sessions.chat_session import AsyncChatSession
from nexus.mcp import McpToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class RealtimeSession:
    """实时会话状态（异步版本，无队列）"""

    chat_session: AsyncChatSession
    chat_model: str
    websocket: WebSocket
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tools: List[RealtimeFunctionTool] = field(default_factory=list)
    
    # MCP 工具注册表
    mcp_registry: McpToolRegistry = field(default_factory=McpToolRegistry)

    sample_rate: int = 16000
    output_modalities: list[str] = field(default_factory=lambda: ["text"])

    # 仅保留音频输入队列（客户端 → ASR）
    audio_queue: asyncio.Queue[np.ndarray] = field(default_factory=asyncio.Queue)

    # WebSocket 写入锁，防止并发写入冲突
    _ws_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # 打断控制：当前聊天任务和取消事件
    _current_chat_task: Optional["Task"] = field(default=None, repr=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    
    def request_cancel(self) -> None:
        """请求取消当前 LLM 生成"""
        self._cancel_event.set()
        logger.info(f"Cancel requested for session {self.session_id}")
    
    def reset_cancel(self) -> None:
        """重置取消状态"""
        self._cancel_event.clear()
    
    def is_cancel_requested(self) -> bool:
        """检查是否请求了取消"""
        return self._cancel_event.is_set()
    
    def set_current_chat_task(self, task: Optional["Task"]) -> None:
        """设置当前聊天任务"""
        self._current_chat_task = task
    
    def get_current_chat_task(self) -> Optional["Task"]:
        """获取当前聊天任务"""
        return self._current_chat_task

    async def send_event(self, event):
        """发送事件到 WebSocket（线程安全）"""
        if not isinstance(event, dict):
            event = event.model_dump()
        async with self._ws_lock:
            await self.websocket.send_text(json.dumps(event, ensure_ascii=False))

    async def audio_iter(self) -> AsyncGenerator[np.ndarray, None]:
        """异步音频迭代器，供 ASR 使用"""
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def update_output_modalities(self, modalities: List[str]):
        self.output_modalities = modalities

    def get_output_modalities(self) -> List[str]:
        return self.output_modalities.copy()
    
    def get_all_tools(self) -> List[RealtimeFunctionTool]:
        """获取所有可用工具（普通工具 + MCP 工具）"""
        all_tools = list(self.tools)
        all_tools.extend(self.mcp_registry.to_realtime_function_tools())
        return all_tools
    
    def is_mcp_tool(self, tool_name: str) -> bool:
        """判断工具是否为 MCP 工具"""
        return self.mcp_registry.is_mcp_tool(tool_name)
    
    def get_mcp_server_for_tool(self, tool_name: str) -> Optional[str]:
        """获取 MCP 工具所属的服务器标签"""
        return self.mcp_registry.get_server_for_tool(tool_name)

    async def chat(
        self,
        user_message: str,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        返回原始的 ChatCompletionChunk 流。
        工具调用的检测和事件发送由 Servicer 层处理。
        """
        from nexus.servicers.realtime.utils import convert_to_chat_tools
        
        # 获取所有工具（包括 MCP 工具）
        all_tools = self.get_all_tools()
        
        chat_stream_resp = self.chat_session.chat(
            user_message=user_message,
            model=self.chat_model,
            stream=True,
            tools=convert_to_chat_tools(all_tools),
        )
        async for chunk in chat_stream_resp:
            yield chunk

    def add_tool_result(self, tool_call_id: str, content: str):
        """
        添加工具调用结果到对话历史。
        客户端发送 conversation.item.create 时调用。
        """
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.chat_session.chat_history.append(tool_msg)
        logger.info(f"Tool result added to history: tool_call_id={tool_call_id}")

    async def continue_conversation(
        self,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        基于当前对话历史继续生成响应。
        用于客户端发送 response.create 时。
        """
        from nexus.servicers.realtime.utils import convert_to_chat_tools
        
        # 获取所有工具（包括 MCP 工具）
        all_tools = self.get_all_tools()
        
        # 直接基于现有历史生成响应
        stream_resp = await self.chat_session.chat_inferencer.chat(
            messages=self.chat_session.chat_history,
            model=self.chat_model,
            stream=True,
            tools=convert_to_chat_tools(all_tools),
        )
        async for chunk in self.chat_session.get_result_record_itr(stream_resp):
            yield chunk
