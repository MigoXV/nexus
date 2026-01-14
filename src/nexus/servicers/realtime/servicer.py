import queue
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nexus.inferencers.asr.inferencer import Inferencer

from .session import RealtimeSession

class RealtimeServicer:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.inferencer = Inferencer(self.grpc_addr)

    def realtime_worker(
        self, session: RealtimeSession
    ) -> Iterable[str]:
        """后台工作线程，将转录结果放入 result_queue"""
        audio_iterator = self._audio_iterator(session)
        for transcript, is_final in self.inferencer.transcribe(
            audio_iterator,
            sample_rate=session.sample_rate,
            interim_results=self.interim_results,
        ):
            print("Transcript:", transcript, "Final:", is_final)
            # 将结果放入队列，由异步任务处理发送
            session.result_queue.put((transcript, is_final))

    def _audio_iterator(self, session: RealtimeSession) -> Iterable[np.ndarray]:
        while True:
            chunk: np.ndarray = session.audio_queue.get()
            if chunk is None:
                break
            yield chunk
