import logging
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple

import grpc
import numpy as np

from nexus.protos.asr import ux_speech_pb2 as pb2
from nexus.protos.asr import ux_speech_pb2_grpc as pb2_grpc


logger = logging.getLogger(__name__)


def request_iter(streaming_config, audio_iterator):
    yield pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
    for audio_chunk in audio_iterator:
        audio_bytes = audio_chunk.tobytes()
        yield pb2.StreamingRecognizeRequest(audio_content=audio_bytes)


@dataclass
class TranscriptionResult:
    transcript: str
    words: str
    start_time: float
    end_time: float


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
    ) -> Generator[Tuple[str, bool], None, None]:
        """
        执行一次性离线语音识别。

        :param audio: 音频数据迭代器，每个元素为 np.ndarray (建议 dtype=int16)
        :param sample_rate: 音频采样率 (Hz)
        :param language_code: 语言编码
        :param encoding: 音频编码类型
        :param enable_automatic_punctuation: 是否启用自动标点
        :param hotwords: 热词列表
        :param hotword_bias: 热词偏置
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
            interim_results=True,
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
                    yield (transcript, is_final)
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
