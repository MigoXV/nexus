from __future__ import annotations

from nexus.configs.config import NexusConfig

from .backend import TTSBackend
from .grpc_backend import GrpcDuplexTTSBackend, GrpcTTSBackendConfig
from .inferencer import OpenAITTSBackend


def create_tts_backend(config: NexusConfig) -> TTSBackend:
    if config.tts_backend == "grpc":
        return GrpcDuplexTTSBackend(
            GrpcTTSBackendConfig(
                grpc_addr=config.tts_grpc_addr or "",
                preset_voice_id=config.tts_grpc_preset_voice_id or None,
                ref_voice_dir=config.tts_grpc_ref_voice_dir or None,
                language=config.tts_grpc_language,
                decoder_chunk_size=config.tts_grpc_decoder_chunk_size,
                text_chunk_size=config.tts_grpc_text_chunk_size,
            )
        )

    return OpenAITTSBackend(
        base_url=config.tts_base_url or "",
        api_key=config.tts_api_key or "",
    )
