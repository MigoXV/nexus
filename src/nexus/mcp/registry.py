"""
MCP 工具注册表

管理多个 MCP 服务器及其工具，提供统一的工具查找和调用接口。
"""

import logging
from typing import Any, Dict, List, Optional

from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from .models import McpServerConfig, McpTool
from .client import McpClient

logger = logging.getLogger(__name__)


class McpToolRegistry:
    """MCP 工具注册表
    
    管理多个 MCP 服务器的连接和工具。
    提供工具查找、调用等统一接口。
    
    工具标识策略：
    - 内部使用 "{server_label}.{tool_name}" 作为唯一标识
    - 对 LLM 呈现时使用原始 tool_name
    - 当多个服务器有同名工具时，使用 server_label 区分
    """
    
    def __init__(self):
        self._clients: Dict[str, McpClient] = {}  # server_label -> client
        self._tools: Dict[str, McpTool] = {}  # tool_name -> McpTool (简单映射)
        self._tool_to_server: Dict[str, str] = {}  # tool_name -> server_label
    
    @property
    def server_labels(self) -> List[str]:
        """获取所有已注册的服务器标签"""
        return list(self._clients.keys())
    
    @property
    def tools(self) -> List[McpTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[McpTool]:
        """根据名称获取工具"""
        return self._tools.get(name)
    
    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """获取工具所属的服务器标签"""
        return self._tool_to_server.get(tool_name)
    
    def is_mcp_tool(self, tool_name: str) -> bool:
        """判断是否为 MCP 工具"""
        return tool_name in self._tools
    
    async def register_server(self, config: McpServerConfig) -> List[McpTool]:
        """注册一个 MCP 服务器
        
        连接到服务器，获取工具列表并注册。
        
        Args:
            config: MCP 服务器配置
            
        Returns:
            该服务器提供的工具列表
        """
        server_label = config.server_label
        
        # 如果已存在，先关闭旧连接
        if server_label in self._clients:
            await self.unregister_server(server_label)
        
        # 创建客户端并连接
        client = McpClient(config)
        try:
            tools = await client.connect()
        except Exception as e:
            # 确保清理客户端资源
            await client.close()
            raise
        
        # 注册客户端
        self._clients[server_label] = client
        
        # 注册工具
        for tool in tools:
            self._tools[tool.name] = tool
            self._tool_to_server[tool.name] = server_label
            logger.debug(f"Registered MCP tool: {tool.name} from {server_label}")
        
        logger.info(
            f"Registered MCP server {server_label} with {len(tools)} tools"
        )
        return tools
    
    async def unregister_server(self, server_label: str):
        """注销一个 MCP 服务器"""
        if server_label not in self._clients:
            return
        
        # 移除该服务器的工具
        tools_to_remove = [
            name for name, label in self._tool_to_server.items()
            if label == server_label
        ]
        for name in tools_to_remove:
            del self._tools[name]
            del self._tool_to_server[name]
        
        # 关闭客户端
        client = self._clients.pop(server_label)
        await client.close()
        
        logger.info(f"Unregistered MCP server {server_label}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            调用结果
        """
        server_label = self._tool_to_server.get(tool_name)
        if not server_label:
            raise ValueError(f"Unknown MCP tool: {tool_name}")
        
        client = self._clients.get(server_label)
        if not client:
            raise RuntimeError(f"MCP server {server_label} not connected")
        
        return await client.call_tool(tool_name, arguments)
    
    def to_realtime_function_tools(self) -> List[RealtimeFunctionTool]:
        """将所有 MCP 工具转换为 RealtimeFunctionTool 列表"""
        return [tool.to_realtime_function_tool() for tool in self._tools.values()]
    
    async def close(self):
        """关闭所有连接"""
        for server_label in list(self._clients.keys()):
            await self.unregister_server(server_label)
        logger.info("MCP tool registry closed")
