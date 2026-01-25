"""
FastAPI WebSocket 路由 - OpenAI Realtime API 兼容
处理实时语音转录的 WebSocket 连接
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import Depends, Query, WebSocket, WebSocketDisconnect
from openai.types.realtime import RealtimeFunctionTool
from nexus.servicers.realtime.build_events import (
    build_error_event,
    build_session_created,
)
from nexus.servicers.realtime.servicer import RealtimeServicer
from nexus.servicers.realtime.contexts import McpListToolsContext
from nexus.sessions import RealtimeSession
from nexus.mcp import McpServerConfig

from .depends import get_realtime_servicer
from .event_routes import handle_session_event

logger = logging.getLogger(__name__)


async def realtime_endpoint_worker(
    websocket: WebSocket,
    model: str = Query(default="gpt-4o-realtime-preview"),
    realtime_servicer: RealtimeServicer = Depends(get_realtime_servicer),
):
    is_chat_model = "transcribe" not in model.lower()
    logger.info(
        f"WebSocket connected for model: {model}, is_chat_model: {is_chat_model}"
    )
    await websocket.accept()
    # 创建会话
    update_event = await receive_update_event(websocket)
    # 如果首先发送的不是配置请求，则报错并关闭连接
    if not update_event:
        logger.error("Failed to receive initial session.update event")
        await _send_event(
            websocket,
            build_error_event("invalid_request", "First event must be session.update"),
        )
        await websocket.close()
        return
    update_event["type"] = "session.updated"
    await _send_event(websocket, update_event)  # 回显配置事件
    
    # 解析工具配置（分离普通工具和 MCP 配置）
    raw_tools = update_event.get("session", {}).get("tools", [])
    function_tools, mcp_configs = _parse_tools_config(raw_tools)
    
    if function_tools:
        logger.info(f"Initializing session with {len(function_tools)} function tools")
    if mcp_configs:
        logger.info(f"Initializing session with {len(mcp_configs)} MCP servers")
    
    # 初始化 RealtimeSession
    realtime_session = realtime_servicer.create_realtime_session(
        websocket=websocket,
        output_modalities=update_event.get("session", {}).get(
            "output_modalities", ["text"]
        ),
        tools=function_tools,
        chat_model=model,
    )

    await _send_event(
        websocket,
        build_session_created(realtime_session.session_id, model),
    )
    
    # 注册 MCP 服务器并获取工具列表
    for mcp_config in mcp_configs:
        try:
            await _register_mcp_server(realtime_session, mcp_config)
        except Exception as e:
            logger.error(f"Failed to register MCP server {mcp_config.server_label}: {e}")
            # 发送错误事件但不中断会话
            await _send_event(
                websocket,
                build_error_event(
                    "mcp_connection_error",
                    f"Failed to connect to MCP server {mcp_config.server_label}: {e}"
                ),
            )
            # 继续处理其他 MCP 服务器，不 re-raise
    
    # 启动异步后台工作任务
    worker_task = asyncio.create_task(
        realtime_servicer.realtime_worker(realtime_session, is_chat_model)
    )

    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            event = json.loads(data)
            event_type = event.get("type", "")
            names: List[str] = event_type.split(".")
            logger.debug(f"Received event: {event_type}")
            if event_type == "input_audio_buffer.append":
                # 追加音频数据到队列
                audio_base64 = event.get("audio", "")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                    await realtime_session.audio_queue.put(audio_chunk)
            elif names[0] == "session":
                # 处理所有 session.* 事件
                await handle_session_event(
                    websocket, event, names, realtime_session, model, _send_event
                )
            elif event_type == "response.cancel":
                logger.info("Response cancelled by client")
            elif event_type == "conversation.item.create":
                item = event.get("item", {})
                if item.get("type", "") == "function_call_output":
                    call_id = item["call_id"]
                    output = item["output"]
                    # 直接添加工具结果到对话历史
                    realtime_session.add_tool_result(
                        tool_call_id=call_id, 
                        content=output
                    )
            elif event_type == "response.create":
                # 客户端显式请求响应，基于当前对话历史生成
                asyncio.create_task(
                    realtime_servicer.generate_response(realtime_session)
                )
            else:
                logger.warning(f"Unknown event type received: {event_type}")
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected for session: {realtime_session.session_id}"
        )
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await _send_event(
            websocket,
            build_error_event("server_error", str(e)),
        )
    finally:
        # 清理异步任务
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        # 清理 MCP 连接
        await realtime_session.mcp_registry.close()


async def _send_event(websocket: WebSocket, event: dict):
    """发送事件到 WebSocket"""
    if not isinstance(event, dict):
        event = event.model_dump()
    await websocket.send_text(json.dumps(event, ensure_ascii=False))


async def receive_update_event(websocket: WebSocket) -> Dict:
    """接收 session.update 事件"""
    try:
        data = await websocket.receive_text()
        event = json.loads(data)
        event_type = event.get("type", "")
        if event_type == "session.update":
            logger.info("Received session.update event")
            return event
        else:
            logger.warning(f"Expected session.update, but received: {event_type}")
            return None
    except Exception as e:
        logger.error(f"Error receiving session.update event: {e}")
        return None


def _parse_tools_config(
    raw_tools: List[Dict[str, Any]]
) -> tuple[List[RealtimeFunctionTool], List[McpServerConfig]]:
    """解析工具配置，分离普通工具和 MCP 配置
    
    Args:
        raw_tools: 原始工具配置列表
        
    Returns:
        (function_tools, mcp_configs) 元组
    """
    function_tools = []
    mcp_configs = []
    
    for tool in raw_tools:
        tool_type = tool.get("type", "function")
        
        if tool_type == "mcp":
            # MCP 服务器配置
            mcp_config = McpServerConfig.from_dict(tool)
            mcp_configs.append(mcp_config)
            logger.debug(f"Parsed MCP config: {mcp_config.server_label}")
        else:
            # 普通 function tool
            function_tools.append(RealtimeFunctionTool(**tool))
            logger.debug(f"Parsed function tool: {tool.get('name', 'unknown')}")
    
    return function_tools, mcp_configs


async def _register_mcp_server(
    session: RealtimeSession,
    config: McpServerConfig,
):
    """注册 MCP 服务器并发送相关事件
    
    按照 OpenAI 官方时序发送事件：
    1. conversation.item.added (mcp_list_tools, tools=[])
    2. mcp_list_tools.in_progress
    3. mcp_list_tools.completed
    4. conversation.item.done (mcp_list_tools, tools=[...])
    """
    async with McpListToolsContext(session, config.server_label) as ctx:
        # 连接 MCP 服务器并获取工具列表
        tools = await session.mcp_registry.register_server(config)
        
        # 转换工具为事件格式
        tools_data = []
        for tool in tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "annotations": tool.annotations,
            })
        
        # 设置工具列表
        ctx.set_tools(tools_data)
    
    logger.info(
        f"Registered MCP server {config.server_label} with {len(tools)} tools"
    )
