import logging
from typing import Iterator

from openai import OpenAI


logger = logging.getLogger(__name__)


class Inferencer:
    """
    TTS 推理器，封装 OpenAI 兼容的 Text-to-Speech API 调用（透明代理模式）。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        """
        初始化 TTS 推理器。

        :param base_url: API 服务器地址
        :param api_key: API 密钥
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def speech(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "rita",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        非流式 TTS 推理，返回完整音频数据。

        :param input: 要转换为语音的文本
        :param model: 模型名称
        :param voice: 语音类型
        :param response_format: 音频格式 (mp3, opus, aac, flac, wav, pcm)
        :param speed: 语速 (0.25 - 4.0)
        :return: 音频二进制数据
        """
        try:
            params = {
                "model": model,
                "input": input,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            }
            params.update(kwargs)

            response = self.client.audio.speech.create(**params)
            return response.content
        except Exception as err:
            logger.error("TTS inference error: %s", err)
            raise

    def speech_stream(
        self,
        input: str,
        model: str = "tts-1",
        voice: str = "rita",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> Iterator[bytes]:
        """
        流式 TTS 推理，返回音频数据迭代器。

        :param input: 要转换为语音的文本
        :param model: 模型名称
        :param voice: 语音类型
        :param response_format: 音频格式 (mp3, opus, aac, flac, wav, pcm)
        :param speed: 语速 (0.25 - 4.0)
        :return: 生成器，逐步返回音频数据块
        """
        try:
            params = {
                "model": model,
                "input": input,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            }
            params.update(kwargs)

            with self.client.audio.speech.with_streaming_response.create(
                **params
            ) as response:
                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
        except Exception as err:
            logger.error("TTS stream inference error: %s", err)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭客户端连接。"""
        if hasattr(self.client, "close"):
            self.client.close()
