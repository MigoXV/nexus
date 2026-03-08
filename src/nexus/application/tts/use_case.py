from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import struct

from nexus.infrastructure.tts import TTSBackend


@dataclass
class TextToSpeechUseCase:
    backend: TTSBackend

    async def stream_audio(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
    ) -> AsyncIterator[bytes]:
        if getattr(self.backend, "supports_duplex", False):
            return await self._stream_grpc_backend_audio(
                text=text,
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
            )

        return self.backend.speech_stream(
            text=text,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )

    async def _stream_grpc_backend_audio(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
    ) -> AsyncIterator[bytes]:
        if response_format not in {"pcm", "wav"}:
            raise ValueError(
                "gRPC TTS backend only supports response_format 'pcm' and 'wav'"
            )
        if speed != 1.0:
            raise ValueError("gRPC TTS backend only supports speed=1.0")

        if response_format == "pcm":
            return self.backend.speech_stream(
                text=text,
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
            )

        duplex_session = await self.backend.open_duplex_session(
            model=model,
            voice=voice,
            response_format="pcm",
            speed=speed,
        )
        audio_iter = duplex_session.iter_audio()

        try:
            await duplex_session.send_text(text)
            await duplex_session.end_input()
            first_chunk = await anext(audio_iter)
        except StopAsyncIteration as exc:
            await duplex_session.aclose()
            raise RuntimeError("gRPC TTS backend produced no audio") from exc
        except Exception:
            await duplex_session.aclose()
            raise

        async def wav_stream() -> AsyncIterator[bytes]:
            try:
                yield _build_streaming_wav_header(first_chunk.sample_rate)
                if first_chunk.data:
                    yield first_chunk.data
                async for chunk in audio_iter:
                    if chunk.data:
                        yield chunk.data
            finally:
                await duplex_session.aclose()

        return wav_stream()


def _build_streaming_wav_header(sample_rate: int) -> bytes:
    channels = 1
    bits_per_sample = 16
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    data_size = 0xFFFFFFFF
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size & 0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
