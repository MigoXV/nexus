"""MCP 客户端

使用 Streamable HTTP 协议连接 MCP 服务器，获取工具列表并执行工具调用。

注意：MCP 库的 streamablehttp_client 内部使用 anyio TaskGroup，
要求 async with 必须在同一个 task 中进入和退出。
因此我们使用后台任务模式来管理连接生命周期。
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .models import McpServerConfig, McpTool

logger = logging.getLogger(__name__)


class McpClient:
    """MCP 客户端
    
    封装与单个 MCP 服务器的连接和交互。
    使用 Streamable HTTP 协议进行通信。
    
    使用后台任务模式管理连接，确保 async with 在同一个 task 中。
    """
    
    def __init__(self, config: McpServerConfig):
        self.config = config
        self._session: Optional[ClientSession] = None
        self._tools: List[McpTool] = []
        self._connected = False
        
        # 后台任务和同步事件
        self._connection_task: Optional[asyncio.Task] = None
        self._ready_event: Optional[asyncio.Event] = None
        self._close_event: Optional[asyncio.Event] = None
        self._error: Optional[Exception] = None
        
        # 用于跨任务调用的队列
        self._call_queue: Optional[asyncio.Queue] = None
    
    @property
    def server_label(self) -> str:
        return self.config.server_label
    
    @property
    def tools(self) -> List[McpTool]:
        return self._tools
    
    async def connect(self) -> List[McpTool]:
        """连接到 MCP 服务器并获取工具列表
        
        Returns:
            工具列表
        """
        logger.info(f"Connecting to MCP server: {self.config.server_label} at {self.config.server_url}")
        logger.debug(f"MCP headers: {list(self.config.headers.keys())}")
        
        # 初始化同步原语
        self._ready_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._call_queue = asyncio.Queue()
        self._error = None
        
        # 启动后台连接任务
        self._connection_task = asyncio.create_task(
            self._run_connection(),
            name=f"mcp-connection-{self.config.server_label}"
        )
        
        # 等待连接就绪或出错
        await self._ready_event.wait()
        
        if self._error:
            raise self._error
        
        return self._tools
    
    async def _run_connection(self):
        """后台任务：管理 MCP 连接的生命周期
        
        这个方法在独立的 task 中运行，使用 async with 来正确管理
        streamablehttp_client 的生命周期，避免 cancel scope 错误。
        """
        try:
            async with streamablehttp_client(
                url=self.config.server_url,
                headers=self.config.headers,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    self._session = session
                    
                    # 初始化连接
                    await session.initialize()
                    self._connected = True
                    
                    # 获取工具列表
                    self._tools = await self._fetch_tools(session)
                    
                    logger.info(
                        f"Connected to MCP server {self.config.server_label}, "
                        f"found {len(self._tools)} tools"
                    )
                    
                    # 通知主任务连接就绪
                    self._ready_event.set()
                    
                    # 处理工具调用请求，直到收到关闭信号
                    await self._process_calls(session)
                    
        except Exception as e:
            logger.error(f"MCP connection error for {self.config.server_label}: {e}")
            self._error = e
            self._connected = False
            self._ready_event.set()  # 通知等待者出错了
        finally:
            self._session = None
            self._connected = False
    
    async def _fetch_tools(self, session: ClientSession) -> List[McpTool]:
        """获取 MCP 服务器的工具列表"""
        result = await session.list_tools()
        tools = []
        
        for tool in result.tools:
            # 过滤工具（如果配置了 allowed_tools）
            if self.config.allowed_tools and tool.name not in self.config.allowed_tools:
                continue
            
            mcp_tool = McpTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                server_label=self.config.server_label,
                annotations=None,
            )
            tools.append(mcp_tool)
        
        return tools
    
    async def _process_calls(self, session: ClientSession):
        """处理工具调用请求队列"""
        while not self._close_event.is_set():
            try:
                # 使用超时来定期检查关闭事件
                request = await asyncio.wait_for(
                    self._call_queue.get(),
                    timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            
            name, arguments, result_future = request
            
            try:
                result = await session.call_tool(name, arguments)
                
                # 处理结果
                if result.content:
                    texts = []
                    for content in result.content:
                        if hasattr(content, 'text'):
                            texts.append(content.text)
                        else:
                            texts.append(str(content))
                    output = "\n".join(texts)
                else:
                    output = ""
                
                result_future.set_result(output)
                
            except Exception as e:
                result_future.set_exception(e)
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具
        
        Args:
            name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具调用结果（字符串）
        """
        if not self._connected or not self._call_queue:
            raise RuntimeError("Not connected to MCP server")
        
        logger.info(f"Calling MCP tool: {name} with arguments: {arguments}")
        
        # 创建 Future 来接收结果
        loop = asyncio.get_running_loop()
        result_future = loop.create_future()
        
        # 将请求放入队列
        await self._call_queue.put((name, arguments, result_future))
        
        # 等待结果
        try:
            output = await result_future
            logger.info(f"MCP tool {name} returned: {output[:200]}..." if len(output) > 200 else f"MCP tool {name} returned: {output}")
            return output
        except Exception as e:
            logger.error(f"MCP tool {name} call failed: {e}")
            raise
    
    async def close(self):
        """关闭连接"""
        if self._close_event:
            self._close_event.set()
        
        if self._connection_task and not self._connection_task.done():
            # 等待后台任务完成
            try:
                await asyncio.wait_for(self._connection_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"MCP connection task for {self.config.server_label} did not finish in time, cancelling")
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
        
        self._connected = False
        logger.info(f"MCP client {self.config.server_label} closed")
    
    async def __aenter__(self) -> "McpClient":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
