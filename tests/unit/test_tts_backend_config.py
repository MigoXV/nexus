from __future__ import annotations

import wave

import pytest

from nexus.configs.config import NexusConfig
from nexus.infrastructure.tts import GrpcDuplexTTSBackend, OpenAITTSBackend, create_tts_backend


def _base_kwargs() -> dict:
    return {
        "asr_grpc_addr": "localhost:5000",
        "chat_base_url": "http://localhost:10002/v1",
        "chat_api_key": "dummy-key",
    }


def _write_test_wav(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


@pytest.mark.asyncio
async def test_openai_tts_backend_configuration_is_valid() -> None:
    config = NexusConfig(
        **_base_kwargs(),
        tts_backend="openai",
        tts_base_url="http://localhost:10003/v1",
        tts_api_key="no-key",
    )

    backend = create_tts_backend(config)

    assert isinstance(backend, OpenAITTSBackend)
    await backend.close()


@pytest.mark.asyncio
async def test_grpc_tts_backend_configuration_is_valid_with_preset_voice() -> None:
    config = NexusConfig(
        **_base_kwargs(),
        tts_backend="grpc",
        tts_grpc_addr="localhost:30036",
        tts_grpc_preset_voice_id="speaker-1",
    )

    backend = create_tts_backend(config)

    assert isinstance(backend, GrpcDuplexTTSBackend)
    await backend.close()


@pytest.mark.asyncio
async def test_grpc_tts_backend_configuration_is_valid_with_reference_voice_dir(
    tmp_path,
) -> None:
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    _write_test_wav(voice_dir / "speaker.wav")
    (voice_dir / "voices.tsv").write_text(
        "speaker.wav\thello world\talloy\n",
        encoding="utf-8",
    )

    config = NexusConfig(
        **_base_kwargs(),
        tts_backend="grpc",
        tts_grpc_addr="localhost:30036",
        tts_grpc_ref_voice_dir=str(voice_dir),
    )

    backend = create_tts_backend(config)

    assert isinstance(backend, GrpcDuplexTTSBackend)
    await backend.close()


def test_grpc_tts_backend_requires_voice_source() -> None:
    with pytest.raises(ValueError, match="voice source"):
        NexusConfig(
            **_base_kwargs(),
            tts_backend="grpc",
            tts_grpc_addr="localhost:30036",
        )


def test_grpc_tts_backend_rejects_conflicting_voice_sources(tmp_path) -> None:
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    _write_test_wav(voice_dir / "speaker.wav")
    (voice_dir / "voices.tsv").write_text(
        "speaker.wav\thello world\talloy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        NexusConfig(
            **_base_kwargs(),
            tts_backend="grpc",
            tts_grpc_addr="localhost:30036",
            tts_grpc_preset_voice_id="speaker-1",
            tts_grpc_ref_voice_dir=str(voice_dir),
        )
