import json
import logging
from typing import List, Optional

from fastapi import WebSocket
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from nexus.inferencers.asr.inferencer import AsyncInferencer
from nexus.inferencers.chat.inferencer import AsyncInferencer as AsyncChatInferencer
from nexus.inferencers.tts.inferencer import Inferencer as TTSInferencer
from nexus.sessions import RealtimeSession
from nexus.sessions.chat_session import AsyncChatSession

from .send_responses import (
    send_transcribe_response,
    process_chat_stream,
    send_tool_result_response,
    send_text_response,
)
from .contexts import McpCallResponseContext

logger = logging.getLogger(__name__)


class RealtimeServicer:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
        chat_base_url: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        tts_base_url: Optional[str] = None,
        tts_api_key: Optional[str] = None,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.inferencer = AsyncInferencer(self.grpc_addr)
        self.chat_inferencer = (
            AsyncChatInferencer(api_key=chat_api_key, base_url=chat_base_url)
            if chat_api_key
            else None
        )
        self.tts_inferencer = (
            TTSInferencer(base_url=tts_base_url, api_key=tts_api_key)
            if tts_api_key
            else None
        )

    async def close(self):
        """关闭所有资源"""
        if self.inferencer:
            await self.inferencer.close()
            logger.info("ASR inferencer closed")
        if self.chat_inferencer:
            await self.chat_inferencer.close()
            logger.info("Chat inferencer closed")

    def create_realtime_session(
        self,
        websocket: WebSocket,
        output_modalities: List[str],
        tools: List[RealtimeFunctionTool],
        chat_model: str,
    ) -> RealtimeSession:
        chat_session = AsyncChatSession(chat_inferencer=self.chat_inferencer)
        return RealtimeSession(
            chat_session=chat_session,
            chat_model=chat_model,
            websocket=websocket,
            output_modalities=output_modalities,
            tools=tools,
        )

    async def realtime_worker(self, session: RealtimeSession, is_chat: bool = False):
        """异步后台工作协程，直接发送结果到 WebSocket"""
        import asyncio
        
        # 直接使用异步 ASR 转录
        async for asr_result in self.inferencer.transcribe(
            session.audio_iter(),
            sample_rate=session.sample_rate,
            interim_results=self.interim_results,
        ):
            try:
                await send_transcribe_response(session, asr_result)
            except Exception as e:
                logger.error(f"Error sending transcribe response: {e}")
            # 未配置对话功能，不返回对话结果
            if not is_chat:
                continue
            
            # 检查是否有正在进行的 chat 任务，如果有则先取消
            current_task = session.get_current_chat_task()
            if current_task is not None and not current_task.done():
                logger.info(
                    f"New transcription received, cancelling current chat task"
                )
                session.request_cancel()  # 设置取消标志
                current_task.cancel()  # 真正取消任务
                try:
                    await current_task  # 等待任务被取消
                except asyncio.CancelledError:
                    logger.info("Chat task was cancelled successfully")
                except Exception as e:
                    logger.warning(f"Error waiting for cancelled chat task: {e}")
            
            # 重置取消状态，准备新的 chat
            session.reset_cancel()
            
            # 创建新的 chat 任务
            chat_task = asyncio.create_task(
                self.chat_worker(session, asr_result.transcript)
            )
            session.set_current_chat_task(chat_task)
            
            # 不等待任务完成，让它在后台运行
            # 这样新的转写事件可以立即触发取消

    async def chat_worker(self, session: RealtimeSession, user_message: str):
        """
        处理聊天请求，流式发送响应并检测工具调用。
        
        工作流程：
        1. 获取 session 的 chat 流
        2. 使用 process_chat_stream 流式处理，
           文本会立即发送给客户端（降低首字延迟）
        3. 如果检测到 MCP 工具调用，在服务端执行调用，
           将结果添加到对话历史后自动继续生成响应
        4. 如果检测到普通 function call，事件已在处理过程中发送，
           等待客户端 response.create
        """
        chat_stream = session.chat(user_message)
        result = await process_chat_stream(session, chat_stream)
        
        if result.has_mcp_call:
            # MCP 工具调用：服务端执行
            await self._execute_mcp_call(session, result.tool_call)
        elif result.has_tool_call:
            # 普通 function call：等待客户端返回结果
            logger.info(
                f"Function call sent: {result.tool_call.name}, "
                f"waiting for client to send tool result and response.create"
            )
    
    async def _execute_mcp_call(
        self,
        session: RealtimeSession,
        tool_call,
    ):
        """
        执行 MCP 工具调用并发送后续事件。
        
        MCP 调用在服务端执行，完成后：
        1. 执行实际的 MCP 工具调用
        2. 通过 mcp_ctx 发送 mcp_call.completed 事件
        3. 发送 conversation.item.done 和 output_item.done 事件
        4. 将结果添加到对话历史
        5. 自动继续生成响应
        """
        tool_name = tool_call.name
        server_label = tool_call.server_label
        arguments_str = tool_call.arguments
        mcp_ctx = tool_call.mcp_ctx
        
        if not mcp_ctx:
            logger.error(f"MCP context not found for tool call: {tool_name}")
            return
        
        logger.info(f"Executing MCP call: {tool_name} on {server_label}")
        
        # 解析参数
        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP call arguments: {e}")
            arguments = {}
        
        try:
            # 执行 MCP 调用
            output = await session.mcp_registry.call_tool(tool_name, arguments)
            mcp_ctx.set_output(output)
            
            # 添加工具调用到对话历史（作为 assistant 的工具调用）
            # 首先添加 assistant 的工具调用消息
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": mcp_ctx.item_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str,
                    }
                }]
            }
            session.chat_session.chat_history.append(assistant_msg)
            
            # 然后添加工具结果
            tool_msg = {
                "role": "tool",
                "tool_call_id": mcp_ctx.item_id,
                "content": output,
            }
            session.chat_session.chat_history.append(tool_msg)
            
            logger.info(
                f"MCP call {tool_name} completed, output length: {len(output)}"
            )
            
        except Exception as e:
            logger.error(f"MCP call {tool_name} failed: {e}")
            mcp_ctx.set_error(str(e))
            
            # 添加错误到对话历史
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": mcp_ctx.item_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str,
                    }
                }]
            }
            session.chat_session.chat_history.append(assistant_msg)
            
            tool_msg = {
                "role": "tool",
                "tool_call_id": mcp_ctx.item_id,
                "content": f"Error: {e}",
            }
            session.chat_session.chat_history.append(tool_msg)
        
        # 发送 MCP 调用完成事件（completed, item.done, output_item.done）
        await mcp_ctx.__aexit__(None, None, None)
        
        # 不自动生成响应，等待客户端发送 response.create 事件
        # 工具调用结果已保存在历史记录中，客户端可以选择何时继续对话
        logger.info(f"MCP call completed, waiting for client response.create event")

    async def generate_response(self, session: RealtimeSession):
        """
        基于当前对话历史生成响应。
        由客户端 response.create 事件触发调用。
        """
        chat_stream = session.continue_conversation()
        await send_tool_result_response(session, chat_stream)
        logger.info("Response generated based on conversation history")
