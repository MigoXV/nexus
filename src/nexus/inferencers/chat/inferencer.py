import logging
from typing import AsyncIterator, Iterator, List, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)


class Inferencer:
    """
    Chat 推理器，封装 OpenAI 兼容的 Chat Completions API 调用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
    ):
        """
        初始化 Chat 推理器。

        :param base_url: API 服务器地址
        :param api_key: API 密钥
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        stream: bool = False,
    ):

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                audio=audio,
                tools=tools,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return response
        except Exception as err:
            logger.error("Chat inference error: %s", err)
            raise
            return ""

    def chat_stream(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Iterator[str]:
        """
        流式 Chat 推理，返回响应文本迭代器。

        :param messages: 完整的消息列表
        :param model: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :return: 生成器，逐步返回模型生成的文本片段
        """
        try:
            stream_resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as err:
            logger.error("Chat stream inference error: %s", err)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭客户端连接。"""
        if hasattr(self.client, "close"):
            self.client.close()


class AsyncInferencer:
    """
    异步 Chat 推理器，封装 OpenAI 兼容的 Chat Completions API 调用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
    ):
        """
        初始化异步 Chat 推理器。

        :param base_url: API 服务器地址
        :param api_key: API 密钥
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        audio: Optional[ChatCompletionAudioParam] = None,
        tools: List[ChatCompletionToolUnionParam] = [],
        frequency_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        stream: bool = False,
    ):
        """
        异步 Chat 推理。

        :param messages: 完整的消息列表
        :param model: 模型名称
        :param audio: 音频参数
        :param tools: 工具列表
        :param frequency_penalty: 频率惩罚
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :param stream: 是否流式返回
        :return: ChatCompletion 或流式响应
        """
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model,
                audio=audio,
                tools=tools if tools else None,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            return response
        except Exception as err:
            logger.error("Async chat inference error: %s", err)
            raise

    async def chat_stream(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        异步流式 Chat 推理，返回响应文本迭代器。

        :param messages: 完整的消息列表
        :param model: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大 token 数
        :return: 异步生成器，逐步返回模型生成的文本片段
        """
        try:
            stream_resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            async for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as err:
            logger.error("Async chat stream inference error: %s", err)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """关闭客户端连接。"""
        if hasattr(self.client, "close"):
            await self.client.close()
