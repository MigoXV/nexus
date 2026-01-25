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

from nexus.inferencers.chat.inferencer import (
    AsyncInferencer as AsyncChatInferencer,
    Inferencer as ChatInferencer,
)

DEFAULT_SYSTEM_PROMPT = "你是一个有帮助的助手。请根据用户的输入提供有用的信息。" ""


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
