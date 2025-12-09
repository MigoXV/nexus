from __future__ import annotations

from typing import Iterable, List, Optional

import asyncio
import grpc

from ..clients.grpc.stubs import UxSpeechClient
from ..generated import ux_speech_pb2


def _parse_hotwords(hotwords: str) -> List[str]:
    if not hotwords:
        return []
    return [
        w.strip()
        for w in hotwords.replace(";", " ").replace(",", " ").split()
        if w.strip()
    ]


def _build_request_stream(
    audio_bytes: bytes,
    language: str = "zh-CN",
    sample_rate: int = 16000,
    interim_results: bool = False,
    hotwords: Optional[List[str]] = None,
    hotword_bias: float = 0.0,
) -> Iterable[ux_speech_pb2.StreamingRecognizeRequest]:
    """
    Yield a stream of StreamingRecognizeRequest for the gRPC API.

    First yields the streaming config, then yields audio chunks.
    """
    config = ux_speech_pb2.StreamingRecognitionConfig(
        config=ux_speech_pb2.RecognitionConfig(
            encoding=ux_speech_pb2.RecognitionConfig.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            enable_automatic_punctuation=True,
            hotwords=hotwords or [],
            hotword_bias=hotword_bias,
        ),
        interim_results=interim_results,
    )

    yield ux_speech_pb2.StreamingRecognizeRequest(streaming_config=config)

    # 你当前逻辑：一次性发送全部音频
    yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=audio_bytes)

    # 如果未来要真·流式分片：
    # chunk_size = 4096
    # for start in range(0, len(audio_bytes), chunk_size):
    #     chunk = audio_bytes[start : start + chunk_size]
    #     yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=chunk)


class TranscriptionsService:
    """
    Application service for the /v1/audio/transcriptions endpoint.
    """

    async def transcribe(
        self,
        *,
        audio_bytes: bytes,
        language: str,
        sample_rate: int,
        interim_results: bool,
        hotwords: str,
        hotword_bias: float,
        grpc_host: str,
        grpc_port: int,
        timeout_s: float = 180.0,
    ) -> str:
        hotwords_list = _parse_hotwords(hotwords)

        client = UxSpeechClient(host=grpc_host, port=grpc_port)

        requests_iter = _build_request_stream(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=sample_rate,
            interim_results=interim_results,
            hotwords=hotwords_list,
            hotword_bias=hotword_bias,
        )

        final_texts: List[str] = []

        try:
            # stub.StreamingRecognize 是阻塞式迭代器
            # 用 asyncio.to_thread 放到默认线程池执行
            def _call():
                return client.stub.StreamingRecognize(requests_iter, timeout=timeout_s)

            responses = await asyncio.to_thread(_call)

            for resp in responses:
                results = getattr(resp, "results", None) or []
                for result in results:
                    alts = getattr(result, "alternative", None)
                    transcript = getattr(alts, "transcript", "") if alts else ""
                    is_final = bool(getattr(result, "is_final", False))
                    if is_final and transcript:
                        final_texts.append(transcript)

        except grpc.RpcError as rpc_err:
            # 让上层（路由）决定如何映射 HTTP
            raise rpc_err
        finally:
            client.close()

        return " ".join(final_texts).strip()
