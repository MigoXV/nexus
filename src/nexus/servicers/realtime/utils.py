import uuid
from typing import List

from openai.types.chat import ChatCompletionFunctionTool
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool
from openai.types.shared import FunctionDefinition


def get_event_id() -> str:
    return f"event_{str(uuid.uuid4())}"


def get_response_id() -> str:
    return f"resp_{str(uuid.uuid4())}"


def get_item_id() -> str:
    return f"item_{str(uuid.uuid4())}"


def get_conversation_id() -> str:
    return f"conv_{str(uuid.uuid4())}"


def convert_to_chat_tools(
    tools: List[RealtimeFunctionTool],
) -> List[ChatCompletionFunctionTool]:
    """将 RealtimeFunctionTool 列表转换为适用于聊天的工具列表"""
    chat_tools = []
    for tool in tools:
        chat_tool = ChatCompletionFunctionTool(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ),
        )
        chat_tools.append(chat_tool)
    return chat_tools
