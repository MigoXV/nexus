import logging
import queue
from dataclasses import dataclass
from typing import List, Optional

from nexus.inferencers.asr.inferencer import Inferencer
import numpy as np
from .session import RealtimeSession
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class RealtimeServicer:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.inferencer = Inferencer(self.grpc_addr)

    def realtime_worker(self, session: RealtimeSession) -> None:
        """实时转录工作线程（普通函数，非生成器）"""
        audio_iterator = self._audio_iterator(session)
        try:
            for transcript, is_final in self.inferencer.transcribe(
                audio_iterator,
                sample_rate=session.sample_rate,
                interim_results=self.interim_results,
            ):
                print(f"Transcript: {transcript} (final: {is_final})")
                # 将结果放入队列传递给主线程
                session.result_queue.put({
                    "type": "transcript",
                    "text": transcript,
                    "is_final": is_final,
                })
        except Exception as e:
            logger.exception(f"Transcription error: {e}")
            session.result_queue.put({
                "type": "error",
                "message": str(e),
            })
        finally:
            # 发送结束标记
            session.result_queue.put(None)
            logger.info("Realtime worker finished")

    def _audio_iterator(self, session: RealtimeSession) -> Iterable[np.ndarray]:
        """持续从队列读取音频，使用超时机制避免阻塞"""
        while True:
            try:
                chunk: np.ndarray = session.audio_queue.get(timeout=0.5)
                if chunk is None:
                    # 收到结束信号
                    break
                yield chunk
            except queue.Empty:
                # 队列暂时为空，继续等待
                continue
