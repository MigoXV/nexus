from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class DuplexAudioChunk:
    data: bytes
    sample_rate: int


class DuplexTTSSession(ABC):
    @abstractmethod
    async def send_text(self, delta: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def end_input(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def iter_audio(self) -> AsyncIterator[DuplexAudioChunk]:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


class TTSBackend(ABC):
    supports_duplex = False

    @abstractmethod
    async def speech_stream(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def open_duplex_session(
        self,
        *,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
        **kwargs,
    ) -> DuplexTTSSession:
        raise NotImplementedError("This TTS backend does not support duplex streaming")

    async def close(self) -> None:
        return None
