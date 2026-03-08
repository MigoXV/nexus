from __future__ import annotations

import asyncio
import csv
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import grpc.aio
import numpy as np
import soundfile as sf

from nexus.protos.tts import tts_pb2, tts_pb2_grpc

from .backend import DuplexAudioChunk, DuplexTTSSession, TTSBackend


def _load_ref_audio_as_pcm(path: str) -> tuple[bytes, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes(), sample_rate


@dataclass(frozen=True)
class GrpcTTSBackendConfig:
    grpc_addr: str
    preset_voice_id: str | None = None
    ref_voice_dir: str | None = None
    language: str = "auto"
    decoder_chunk_size: int = 50
    text_chunk_size: int = 1


@dataclass(frozen=True)
class ReferenceVoice:
    voice_name: str
    ref_text: str
    pcm_s16le: bytes
    sample_rate: int


def _is_tsv_header(row: list[str]) -> bool:
    if len(row) < 3:
        return False
    return tuple(cell.strip().lower() for cell in row[:3]) in {
        ("filename", "text", "voice"),
        ("file", "text", "voice"),
        ("audio", "text", "voice"),
        ("ref_audio", "ref_text", "voice"),
    }


def _load_reference_voices(directory: str) -> tuple[dict[str, ReferenceVoice], str]:
    voice_dir = Path(directory)
    tsv_files = sorted(voice_dir.glob("*.tsv"))
    if len(tsv_files) != 1:
        raise ValueError("gRPC reference voice directory must contain exactly one TSV file")

    tsv_path = tsv_files[0]
    voices: dict[str, ReferenceVoice] = {}
    default_voice_name: str | None = None
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_index, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].strip().startswith("#"):
                continue
            if row_index == 1 and _is_tsv_header(row):
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Invalid voice TSV row {row_index} in {tsv_path}: expected 3 columns"
                )

            filename, ref_text, voice_name = (cell.strip() for cell in row[:3])
            if not filename or not ref_text or not voice_name:
                raise ValueError(
                    f"Invalid voice TSV row {row_index} in {tsv_path}: empty filename/text/voice"
                )

            audio_path = voice_dir / filename
            if not audio_path.exists():
                raise ValueError(
                    f"Voice audio file '{filename}' from {tsv_path} was not found in {voice_dir}"
                )

            pcm_s16le, sample_rate = _load_ref_audio_as_pcm(str(audio_path))
            if voice_name in voices:
                raise ValueError(f"Duplicate gRPC voice name '{voice_name}' in {tsv_path}")

            voices[voice_name] = ReferenceVoice(
                voice_name=voice_name,
                ref_text=ref_text,
                pcm_s16le=pcm_s16le,
                sample_rate=sample_rate,
            )
            if default_voice_name is None:
                default_voice_name = voice_name

    if not voices or default_voice_name is None:
        raise ValueError(f"No voices were loaded from {tsv_path}")

    return voices, default_voice_name


class GrpcDuplexTTSSession(DuplexTTSSession):
    def __init__(
        self,
        *,
        stub: tts_pb2_grpc.TtsServiceStub,
        config_frame: tts_pb2.DuplexSynthesizeRequest,
        text_chunk_size: int,
    ) -> None:
        self._stub = stub
        self._config_frame = config_frame
        self._text_chunk_size = max(int(text_chunk_size), 1)
        self._request_queue: asyncio.Queue[tts_pb2.DuplexSynthesizeRequest | None] = asyncio.Queue()
        self._input_closed = False
        self._iter_started = False
        self._call = self._stub.DuplexSynthesize(self._request_iter())

    async def _request_iter(self) -> AsyncIterator[tts_pb2.DuplexSynthesizeRequest]:
        yield self._config_frame
        while True:
            item = await self._request_queue.get()
            if item is None:
                break
            yield item

    async def send_text(self, delta: str) -> None:
        if self._input_closed or not delta:
            return
        for index in range(0, len(delta), self._text_chunk_size):
            await self._request_queue.put(
                tts_pb2.DuplexSynthesizeRequest(
                    text_chunk=delta[index : index + self._text_chunk_size]
                )
            )

    async def end_input(self) -> None:
        if self._input_closed:
            return
        self._input_closed = True
        await self._request_queue.put(None)

    async def iter_audio(self) -> AsyncIterator[DuplexAudioChunk]:
        if self._iter_started:
            raise RuntimeError("Duplex audio iterator can only be consumed once")
        self._iter_started = True
        async for audio_chunk in self._call:
            yield DuplexAudioChunk(
                data=audio_chunk.pcm_s16le,
                sample_rate=audio_chunk.sample_rate,
            )

    async def aclose(self) -> None:
        await self.end_input()
        if hasattr(self._call, "cancel"):
            self._call.cancel()


class GrpcDuplexTTSBackend(TTSBackend):
    supports_duplex = True

    def __init__(self, config: GrpcTTSBackendConfig) -> None:
        self._config = config
        self._reference_voices: dict[str, ReferenceVoice] = {}
        self._default_voice_name: str | None = None
        if config.ref_voice_dir:
            self._reference_voices, self._default_voice_name = _load_reference_voices(
                config.ref_voice_dir
            )
        self._channel = grpc.aio.insecure_channel(config.grpc_addr)
        self._stub = tts_pb2_grpc.TtsServiceStub(self._channel)

    def _resolve_reference_voice(self, voice: str) -> ReferenceVoice:
        if not self._reference_voices:
            raise ValueError("Reference voice directory is not configured")

        voice_name = (voice or "").strip() or (self._default_voice_name or "")
        if voice_name in self._reference_voices:
            return self._reference_voices[voice_name]

        available_voices = ", ".join(sorted(self._reference_voices))
        raise ValueError(
            f"Unknown gRPC TTS voice '{voice_name}'. Available voices: {available_voices}"
        )

    def _make_config_frame(self, voice: str) -> tts_pb2.DuplexSynthesizeRequest:
        request = tts_pb2.DuplexSynthesizeRequest()
        config = request.config
        if self._config.preset_voice_id:
            config.voice.preset_voice_id = self._config.preset_voice_id
        else:
            ref_voice = self._resolve_reference_voice(voice)
            config.voice.reference_voice.pcm_s16le = ref_voice.pcm_s16le
            config.voice.reference_voice.sample_rate = ref_voice.sample_rate
            config.voice.reference_voice.ref_text = ref_voice.ref_text
        config.language = self._config.language
        config.decoder_chunk_size = self._config.decoder_chunk_size
        return request

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
        del model, response_format, speed, kwargs
        session = await self.open_duplex_session(
            model="tts-1",
            voice=voice,
            response_format="pcm",
            speed=1.0,
        )
        try:
            await session.send_text(text)
            await session.end_input()
            async for chunk in session.iter_audio():
                if chunk.data:
                    yield chunk.data
        finally:
            await session.aclose()

    async def open_duplex_session(
        self,
        *,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
        **kwargs,
    ) -> DuplexTTSSession:
        del model, response_format, speed, kwargs
        return GrpcDuplexTTSSession(
            stub=self._stub,
            config_frame=self._make_config_frame(voice),
            text_chunk_size=self._config.text_chunk_size,
        )

    async def close(self) -> None:
        await self._channel.close()
