import queue
from dataclasses import dataclass
from typing import List, Optional

from nexus.inferencers.asr.inferencer import Inferencer
import numpy as np
from .session import RealtimeSession
from collections.abc import Iterable


class RealtimeServicer:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.inferencer = Inferencer(self.grpc_addr)

    def realtime_worker(self, session: RealtimeSession) -> Iterable[str]:
        audio_iterator = self._audio_iterator(session)
        for transcript, is_final in self.inferencer.transcribe(
            audio_iterator,
            sample_rate=session.sample_rate,
            interim_results=self.interim_results,
        ):
            print(f"Transcript: {transcript} (final: {is_final})")
            session.result_queue.put(transcript)

    def _audio_iterator(self, session: RealtimeSession) -> Iterable[np.ndarray]:
        while True:
            chunk: np.ndarray = session.audio_queue.get()
            if chunk is None:
                break
            yield chunk
