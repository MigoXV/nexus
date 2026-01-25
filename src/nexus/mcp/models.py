"""
MCP 数据模型

定义 MCP 服务器配置、工具和调用相关的数据结构。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class McpServerConfig:
    """MCP 服务器配置
    
    对应 OpenAI Realtime API 中 tools 数组里 type="mcp" 的配置项。
    
    示例配置：
    {
        "type": "mcp",
        "server_label": "mcd-mcp",
        "server_url": "https://mcp.mcd.cn/mcp-servers/mcd-mcp",
        "headers": {"Authorization": "Bearer xxx"},
        "require_approval": "never"
    }
    """
    server_label: str
    server_url: str
    headers: Dict[str, str] = field(default_factory=dict)
    require_approval: str = "never"  # "never" | "always" | "auto"
    allowed_tools: Optional[List[str]] = None
    server_description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "McpServerConfig":
        """从字典创建配置
        
        支持两种授权方式：
        1. headers: {"Authorization": "Bearer xxx"}
        2. authorization: "xxx" (会自动转换为 Bearer header)
        """
        headers = dict(data.get("headers") or {})
        
        # 处理 authorization 字段，转换为 Bearer header
        authorization = data.get("authorization")
        if authorization and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {authorization}"
        
        return cls(
            server_label=data.get("server_label", ""),
            server_url=data.get("server_url", ""),
            headers=headers,
            require_approval=data.get("require_approval", "never"),
            allowed_tools=data.get("allowed_tools"),
            server_description=data.get("server_description"),
        )


@dataclass
class McpTool:
    """MCP 工具定义
    
    从 MCP 服务器获取的工具信息。
    """
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_label: str  # 所属的 MCP 服务器标签
    annotations: Optional[Dict[str, Any]] = None
    
    def to_function_tool_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI function tool 格式"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
    
    def to_realtime_function_tool(self):
        """转换为 RealtimeFunctionTool"""
        from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool
        return RealtimeFunctionTool(
            type="function",
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
        )


@dataclass
class McpToolCall:
    """MCP 工具调用
    
    记录一次 MCP 工具调用的信息。
    """
    id: str  # 调用 ID，格式如 "mcp_xxx"
    name: str  # 工具名称
    arguments: str  # JSON 格式的参数
    server_label: str  # 所属的 MCP 服务器标签
    output: Optional[str] = None  # 调用结果
    error: Optional[str] = None  # 错误信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": "mcp_call",
            "name": self.name,
            "arguments": self.arguments,
            "server_label": self.server_label,
            "output": self.output,
            "error": self.error,
        }
