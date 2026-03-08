from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from nexus.application.chat import ChatCompletionUseCase
from nexus.application.realtime.service import RealtimeApplicationService
from nexus.application.transcribe import TranscribeUseCase
from nexus.application.tts import TextToSpeechUseCase
from nexus.configs.config import NexusConfig
from nexus.infrastructure.tts import create_tts_backend

logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    config: NexusConfig
    realtime: RealtimeApplicationService
    chat: ChatCompletionUseCase
    transcribe: TranscribeUseCase
    tts: TextToSpeechUseCase

    async def close(self) -> None:
        await self.realtime.close()


_container: Optional[AppContainer] = None


def configure(engine_config: NexusConfig) -> None:
    global _container
    if _container is not None:
        return

    tts_backend = create_tts_backend(engine_config)

    realtime = RealtimeApplicationService(
        grpc_addr=engine_config.asr_grpc_addr,
        interim_results=engine_config.asr_interim_results,
        asr_hide_metadata=engine_config.asr_hide_metadata,
        chat_base_url=engine_config.chat_base_url,
        chat_api_key=engine_config.chat_api_key,
        tts_backend=tts_backend,
    )

    _container = AppContainer(
        config=engine_config,
        realtime=realtime,
        chat=ChatCompletionUseCase(
            base_url=engine_config.chat_base_url,
            api_key=engine_config.chat_api_key,
        ),
        transcribe=TranscribeUseCase(
            grpc_addr=engine_config.asr_grpc_addr,
            interim_results=engine_config.asr_interim_results,
        ),
        tts=TextToSpeechUseCase(
            backend=tts_backend,
        ),
    )
    logger.info("Application container initialized")


async def shutdown() -> None:
    global _container
    if _container is not None:
        await _container.close()
        _container = None
        logger.info("Application container shutdown complete")


def get_container() -> AppContainer:
    if _container is None:
        raise RuntimeError("Container not configured. Call configure() first.")
    return _container
