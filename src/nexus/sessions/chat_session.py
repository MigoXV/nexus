import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Generator, Iterable
from dataclasses import dataclass, field
from typing import List, Optional

from openai.types.chat import (
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function

from nexus.infrastructure.chat.inferencer import (
    AsyncInferencer as AsyncChatInferencer,
    Inferencer as ChatInferencer,
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个中文语音助手。"
    "你的回答必须是自然口语，适合直接朗读。"
    "只输出纯文本口语内容。"
    "严禁输出 Markdown 代码块 标题 列表 链接 emoji 表情符号 或任何装饰性符号。"
    "尽量不用标点和书面化表达。"
    "如果输入里给了当前说话人的信息，只把它当作理解上下文，不要直接复述声纹标签或识别元数据。"
)


def _replace_assistant_message_content_in_history(
    chat_history: List[ChatCompletionMessageParam],
    content: str,
) -> None:
    for index in range(len(chat_history) - 1, -1, -1):
        message = chat_history[index]
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                message["content"] = content
                return
            continue
        if getattr(message, "role", None) != "assistant":
            continue
        chat_history[index] = ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=getattr(message, "tool_calls", []) or [],
        )
        return


@dataclass
class ChatSession:
    """聊天会话状态"""

    chat_inferencer: ChatInferencer
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # 聊天历史记录
    chat_history: List[ChatCompletionMessageParam] = field(default_factory=list)

    def __post_init__(self):
        # 初始化时添加系统提示到聊天历史
        system_message = {
            "role": "system",
            "content": self.system_prompt,
        }
        self.chat_history.append(system_message)

    def chat(
        self,
        user_message: str,
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """添加用户消息并返回聊天历史"""
        user_msg = {
            "role": "user",
            "content": user_message,
        }
        self.chat_history.append(user_msg)
        stream_resp = self.chat_inferencer.chat(
            messages=self.chat_history,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        stream_resp = self.get_result_record_itr(stream_resp)
        return stream_resp

    def use_tool(
        self,
        tool_call_id: str,
        content: str,
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Generator[ChatCompletionChunk, None, None]:
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.chat_history.append(tool_msg)
        stream_resp = self.chat_inferencer.chat(
            messages=self.chat_history,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        stream_resp = self.get_result_record_itr(stream_resp)
        return stream_resp

    def get_result_record_itr(
        self,
        stream_resp: Iterable[ChatCompletionChunk],
    ) -> Generator[ChatCompletionChunk, None, None]:
        """从流响应中提取结果记录，从而将user和tool消息添加到聊天历史中"""
        result = ""
        tool_name = None
        content = ""
        tool_call_id = None
        for chunk in stream_resp:
            yield chunk
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
            stream_tool_calls = chunk.choices[0].delta.tool_calls
            if stream_tool_calls:
                if stream_tool_calls[0].function.name:
                    tool_name = stream_tool_calls[0].function.name
                if stream_tool_calls[0].function.arguments:
                    content += stream_tool_calls[0].function.arguments
                if stream_tool_calls[0].id:
                    tool_call_id = stream_tool_calls[0].id
            if chunk.choices[0].finish_reason:
                tool_calls = (
                    [
                        ChatCompletionMessageFunctionToolCall(
                            id=tool_call_id,
                            type="function",
                            function=Function(
                                arguments=content,
                                name=tool_name if tool_name else "",
                            ),
                        )
                    ]
                    if tool_name
                    else []
                )
                message = ChatCompletionMessage(
                    role="assistant",
                    content=result,
                    tool_calls=tool_calls,
                )
                self.chat_history.append(message)

    def replace_last_assistant_message_content(self, content: str) -> None:
        _replace_assistant_message_content_in_history(self.chat_history, content)


@dataclass
class AsyncChatSession:
    """异步聊天会话状态"""

    chat_inferencer: AsyncChatInferencer
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # 聊天历史记录
    chat_history: List[ChatCompletionMessageParam] = field(default_factory=list)

    def __post_init__(self):
        # 初始化时添加系统提示到聊天历史
        system_message = {
            "role": "system",
            "content": self.system_prompt,
        }
        self.chat_history.append(system_message)

    async def chat(
        self,
        user_message: str,
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """添加用户消息并返回聊天历史（异步版本）"""
        user_msg = {
            "role": "user",
            "content": user_message,
        }
        self.chat_history.append(user_msg)
        stream_resp = await self.chat_inferencer.chat(
            messages=self.chat_history,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        async for chunk in self.get_result_record_itr(stream_resp):
            yield chunk

    async def use_tool(
        self,
        tool_call_id: str,
        content: str,
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """使用工具结果继续对话（异步版本）"""
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        self.chat_history.append(tool_msg)
        stream_resp = await self.chat_inferencer.chat(
            messages=self.chat_history,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        async for chunk in self.get_result_record_itr(stream_resp):
            yield chunk

    async def get_result_record_itr(
        self,
        stream_resp: AsyncIterable[ChatCompletionChunk],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """从流响应中提取结果记录，从而将user和tool消息添加到聊天历史中（异步版本）"""
        result = ""
        tool_name = None
        content = ""
        tool_call_id = None
        async for chunk in stream_resp:
            yield chunk
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
            stream_tool_calls = chunk.choices[0].delta.tool_calls
            if stream_tool_calls:
                if stream_tool_calls[0].function.name:
                    tool_name = stream_tool_calls[0].function.name
                if stream_tool_calls[0].function.arguments:
                    content += stream_tool_calls[0].function.arguments
                if stream_tool_calls[0].id:
                    tool_call_id = stream_tool_calls[0].id
            if chunk.choices[0].finish_reason:
                tool_calls = (
                    [
                        ChatCompletionMessageFunctionToolCall(
                            id=tool_call_id,
                            type="function",
                            function=Function(
                                arguments=content,
                                name=tool_name if tool_name else "",
                            ),
                        )
                    ]
                    if tool_name
                    else []
                )
                message = ChatCompletionMessage(
                    role="assistant",
                    content=result,
                    tool_calls=tool_calls,
                )
                self.chat_history.append(message)

    def replace_last_assistant_message_content(self, content: str) -> None:
        _replace_assistant_message_content_in_history(self.chat_history, content)
