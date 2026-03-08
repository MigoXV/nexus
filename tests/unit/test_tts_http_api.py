from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus.api.v1.tts import router
from nexus.application.container import get_container
from nexus.application.tts import TextToSpeechUseCase
from nexus.infrastructure.tts.backend import DuplexAudioChunk


class FakeStreamBackend:
    supports_duplex = False

    async def speech_stream(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
        **kwargs,
    ):
        del model, voice, response_format, speed, kwargs
        yield text.encode("utf-8")


class FakeDuplexSession:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[DuplexAudioChunk | None] = asyncio.Queue()
        self.closed = False

    async def send_text(self, delta: str) -> None:
        midpoint = max(len(delta) // 2, 1)
        await self._queue.put(
            DuplexAudioChunk(data=delta[:midpoint].encode("utf-8"), sample_rate=24000)
        )
        await self._queue.put(
            DuplexAudioChunk(data=delta[midpoint:].encode("utf-8"), sample_rate=24000)
        )

    async def end_input(self) -> None:
        await self._queue.put(None)

    async def iter_audio(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def aclose(self) -> None:
        self.closed = True


class FakeGrpcBackend:
    supports_duplex = True

    def __init__(self) -> None:
        self.session = FakeDuplexSession()
        self.last_voice = None

    async def speech_stream(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        response_format: str,
        speed: float,
        **kwargs,
    ):
        del model, response_format, speed, kwargs
        self.last_voice = voice
        yield text.encode("utf-8")

    async def open_duplex_session(self, **kwargs):
        self.last_voice = kwargs.get("voice")
        self.session = FakeDuplexSession()
        return self.session


def _build_client(tts_use_case: TextToSpeechUseCase) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(tts=tts_use_case)
    return TestClient(app)


def test_audio_speech_openai_backend_streams_audio() -> None:
    client = _build_client(TextToSpeechUseCase(backend=FakeStreamBackend()))

    with client.stream(
        "POST",
        "/audio/speech",
        json={
            "input": "hello",
            "voice": "alloy",
            "response_format": "pcm",
            "speed": 1.0,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"hello"


def test_audio_speech_grpc_backend_streams_pcm() -> None:
    backend = FakeGrpcBackend()
    client = _build_client(TextToSpeechUseCase(backend=backend))

    with client.stream(
        "POST",
        "/audio/speech",
        json={
            "input": "hello",
            "voice": "alloy",
            "response_format": "pcm",
            "speed": 1.0,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"hello"
    assert backend.last_voice == "alloy"


def test_audio_speech_grpc_backend_streams_wav_header_then_audio() -> None:
    backend = FakeGrpcBackend()
    client = _build_client(TextToSpeechUseCase(backend=backend))

    with client.stream(
        "POST",
        "/audio/speech",
        json={
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.0,
        },
    ) as response:
        body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    assert body[44:] == b"hello"
    assert backend.last_voice == "alloy"


def test_audio_speech_grpc_backend_rejects_unsupported_format() -> None:
    client = _build_client(TextToSpeechUseCase(backend=FakeGrpcBackend()))

    response = client.post(
        "/audio/speech",
        json={
            "input": "hello",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        },
    )

    assert response.status_code == 400
    assert "only supports" in response.json()["detail"]


def test_audio_speech_grpc_backend_rejects_non_default_speed() -> None:
    client = _build_client(TextToSpeechUseCase(backend=FakeGrpcBackend()))

    response = client.post(
        "/audio/speech",
        json={
            "input": "hello",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.2,
        },
    )

    assert response.status_code == 400
    assert "speed=1.0" in response.json()["detail"]
