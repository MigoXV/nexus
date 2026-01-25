import logging
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator, Generator, Iterator, List, Optional, Tuple

import grpc
import grpc.aio
import numpy as np

from nexus.protos.asr import ux_speech_pb2 as pb2
from nexus.protos.asr import ux_speech_pb2_grpc as pb2_grpc


logger = logging.getLogger(__name__)


def request_iter(
    streaming_config, audio_iterator, chunk_size: int = 3200
) -> Iterator[pb2.StreamingRecognizeRequest]:
    yield pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
    buffer = np.array([], dtype=np.int16)
    for audio_chunk in audio_iterator:
        buffer = np.concatenate((buffer, audio_chunk))
        while len(buffer) >= chunk_size:  # since dtype=np.int16, 2 bytes per sample
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            audio_bytes = chunk.tobytes()
            yield pb2.StreamingRecognizeRequest(audio_content=audio_bytes)
    if len(buffer) > 0:
        audio_bytes = buffer.tobytes()
        yield pb2.StreamingRecognizeRequest(audio_content=audio_bytes)


async def async_request_iter(
    streaming_config, audio_iterator: AsyncIterator[np.ndarray], chunk_size: int = 3200
) -> AsyncGenerator[pb2.StreamingRecognizeRequest, None]:
    """异步请求迭代器"""
    yield pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
    buffer = np.array([], dtype=np.int16)
    async for audio_chunk in audio_iterator:
        buffer = np.concatenate((buffer, audio_chunk))
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            yield pb2.StreamingRecognizeRequest(audio_content=chunk.tobytes())
    if len(buffer) > 0:
        yield pb2.StreamingRecognizeRequest(audio_content=buffer.tobytes())


@dataclass
class TranscriptionResult:
    transcript: str
    is_final: bool
    words: List[Tuple[str, float, float]] = None  # [(word, start_time, end_time), ...]
    
    def get_end_time(self) -> float:
        """获取最后一个词的结束时间戳"""
        if self.words and len(self.words) > 0:
            return self.words[-1][2]  # end_time of last word
        return 0.0


class Inferencer:
    """
    离线语音识别推理器，封装 gRPC StreamingRecognize 调用。
    """

    def __init__(self, server_addr: str):
        """
        初始化离线 ASR 推理器。

        :param server_addr: gRPC 服务器地址 (host:port)
        """
        self.channel = grpc.insecure_channel(server_addr)
        self.stub = pb2_grpc.UxSpeechStub(self.channel)

    def transcribe(
        self,
        audio: Iterator[np.ndarray],
        sample_rate: int = 16000,
        language_code: str = "zh-CN",
        encoding: int = pb2.RecognitionConfig.LINEAR16,
        enable_automatic_punctuation: bool = True,
        hotwords: Optional[List[str]] = None,
        hotword_bias: float = 0.0,
        interim_results: bool = True,
    ) -> Generator[TranscriptionResult, None, None]:
        """
        执行一次性离线语音识别。

        :param audio: 音频数据迭代器，每个元素为 np.ndarray (建议 dtype=int16)
        :param sample_rate: 音频采样率 (Hz)
        :param language_code: 语言编码
        :param encoding: 音频编码类型
        :param enable_automatic_punctuation: 是否启用自动标点
        :param hotwords: 热词列表
        :param hotword_bias: 热词偏置
        :param interim_results: 是否启用中间结果返回
        :return: 生成器，返回 (文本, is_final) 元组。is_final=True 表示最终结果，False 表示中间结果
        """
        hotwords = hotwords or []

        streaming_config = pb2.StreamingRecognitionConfig(
            config=pb2.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                hotwords=hotwords,
                hotword_bias=hotword_bias,
            ),
            interim_results=interim_results,
        )

        try:
            responses = self.stub.StreamingRecognize(
                request_iter(streaming_config, audio)
            )

            for response in responses:
                for result in response.results:
                    alternative = result.alternative
                    transcript = alternative.transcript
                    is_final = result.is_final
                    if not interim_results and not is_final:
                        continue
                    
                    # 解析词级时间戳
                    words = []
                    for word_info in alternative.words:
                        start_time = word_info.start_time.seconds + word_info.start_time.nanos / 1e9
                        end_time = word_info.end_time.seconds + word_info.end_time.nanos / 1e9
                        words.append((word_info.word, start_time, end_time))
                    
                    yield TranscriptionResult(
                        transcript=transcript,
                        is_final=is_final,
                        words=words if words else None
                    )
        except grpc.RpcError as err:
            logger.error("gRPC error during ASR inference: %s", err)
        except Exception as err:
            logger.error("Unexpected ASR inference error: %s", err)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        关闭 gRPC 通道。
        """
        self.channel.close()


class AsyncInferencer:
    """
    异步语音识别推理器，封装 gRPC aio StreamingRecognize 调用。
    """

    def __init__(self, server_addr: str):
        """
        初始化异步 ASR 推理器。

        :param server_addr: gRPC 服务器地址 (host:port)
        """
        self.server_addr = server_addr
        self.channel = grpc.aio.insecure_channel(server_addr)
        self.stub = pb2_grpc.UxSpeechStub(self.channel)

    async def transcribe(
        self,
        audio: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
        language_code: str = "zh-CN",
        encoding: int = pb2.RecognitionConfig.LINEAR16,
        enable_automatic_punctuation: bool = True,
        hotwords: Optional[List[str]] = None,
        hotword_bias: float = 0.0,
        interim_results: bool = True,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        异步执行流式语音识别。

        :param audio: 异步音频数据迭代器
        :param sample_rate: 音频采样率 (Hz)
        :param language_code: 语言编码
        :param encoding: 音频编码类型
        :param enable_automatic_punctuation: 是否启用自动标点
        :param hotwords: 热词列表
        :param hotword_bias: 热词偏置
        :param interim_results: 是否启用中间结果返回
        :return: 异步生成器，返回 TranscriptionResult
        """
        hotwords = hotwords or []

        streaming_config = pb2.StreamingRecognitionConfig(
            config=pb2.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=enable_automatic_punctuation,
                hotwords=hotwords,
                hotword_bias=hotword_bias,
            ),
            interim_results=interim_results,
        )

        try:
            responses = self.stub.StreamingRecognize(
                async_request_iter(streaming_config, audio)
            )

            async for response in responses:
                for result in response.results:
                    alternative = result.alternative
                    transcript = alternative.transcript
                    is_final = result.is_final
                    if not interim_results and not is_final:
                        continue
                    
                    # 解析词级时间戳
                    words = []
                    for word_info in alternative.words:
                        start_time = word_info.start_time.seconds + word_info.start_time.nanos / 1e9
                        end_time = word_info.end_time.seconds + word_info.end_time.nanos / 1e9
                        words.append((word_info.word, start_time, end_time))
                    
                    yield TranscriptionResult(
                        transcript=transcript,
                        is_final=is_final,
                        words=words if words else None
                    )
        except grpc.RpcError as err:
            logger.error("gRPC error during async ASR inference: %s", err)
        except Exception as err:
            logger.error("Unexpected async ASR inference error: %s", err)

    async def close(self):
        """关闭 gRPC 通道"""
        await self.channel.close()
