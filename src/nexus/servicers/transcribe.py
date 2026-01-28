"""
音频转录服务 - 业务逻辑层
处理音频二进制数据，调用 gRPC Inferencer 完成转录
"""

import base64
import logging
from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional

import numpy as np

from nexus.inferencers.asr.inferencer import Inferencer, TranscriptionResult

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """转录结果"""

    text: str
    language: Optional[str] = None


class TranscribeService:
    """
    音频转录服务
    封装 gRPC Inferencer 调用，处理音频数据格式转换
    """

    def __init__(self, grpc_addr: str, interim_results: bool = False):
        """
        初始化转录服务

        :param grpc_addr: gRPC 服务器地址 (host:port)
        :param interim_results: 是否启用中间结果返回
        """
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results

    def _pcm_to_chunks(
        self, pcm_data: bytes, chunk_size: int = 3200
    ) -> Iterator[np.ndarray]:
        """
        将 PCM 二进制数据分块转换为 numpy 数组

        :param pcm_data: PCM 音频二进制数据 (int16 little-endian)
        :param chunk_size: 每个分块的字节数 (默认 3200 = 100ms @ 16kHz)
        :return: numpy 数组迭代器
        """
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            if chunk:
                yield np.frombuffer(chunk, dtype=np.int16)

    def transcribe_pcm(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh-CN",
        hotwords: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        """
        转录 PCM 音频数据

        :param pcm_data: PCM 音频二进制数据 (int16 little-endian)
        :param sample_rate: 采样率 (Hz)
        :param language: 语言代码
        :param hotwords: 热词列表
        :return: 转录结果
        """
        transcripts: List[str] = []

        with Inferencer(self.grpc_addr) as inferencer:
            audio_iter = self._pcm_to_chunks(pcm_data)

            for result in inferencer.transcribe(
                audio=audio_iter,
                sample_rate=sample_rate,
                language_code=language,
                hotwords=hotwords,
                interim_results=self.interim_results,
            ):
                if result.is_final:
                    transcripts.append(result.transcript)

        full_text = "".join(transcripts)
        return TranscriptionResult(text=full_text, language=language)

    def transcribe_pcm_stream(
        self,
        pcm_data: bytes,
        sample_rate: int = 16000,
        language: str = "zh-CN",
        hotwords: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        流式转录 PCM 音频数据

        :param pcm_data: PCM 音频二进制数据 (int16 little-endian)
        :param sample_rate: 采样率 (Hz)
        :param language: 语言代码
        :param hotwords: 热词列表
        :return: 生成器，逐步返回转录文本
        """
        with Inferencer(self.grpc_addr) as inferencer:
            audio_iter = self._pcm_to_chunks(pcm_data)

            for result in inferencer.transcribe(
                audio=audio_iter,
                sample_rate=sample_rate,
                language_code=language,
                hotwords=hotwords,
                interim_results=self.interim_results,
            ):
                if result.transcript:
                    yield result.transcript

    def transcribe_base64(
        self,
        base64_data: str,
        sample_rate: int = 16000,
        language: str = "zh-CN",
        hotwords: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        """
        转录 Base64 编码的 PCM 音频数据

        :param base64_data: Base64 编码的 PCM 音频数据
        :param sample_rate: 采样率 (Hz)
        :param language: 语言代码
        :param hotwords: 热词列表
        :return: 转录结果
        """
        pcm_data = base64.b64decode(base64_data)
        return self.transcribe_pcm(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language,
            hotwords=hotwords,
        )
