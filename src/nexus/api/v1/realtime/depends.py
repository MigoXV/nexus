import logging
from typing import Optional

from nexus.configs.config import NexusConfig
from nexus.servicers.realtime.servicer import RealtimeServicer

logger = logging.getLogger(__name__)

realtime_servicer: Optional[RealtimeServicer] = None


def configure(
    engine_config: NexusConfig,
):
    """
    初始化 RealtimeServicer。
    注意：此函数必须在 uvicorn event loop 运行后调用（如 FastAPI lifespan 中），
    以确保 grpc.aio channel 绑定到正确的 event loop。
    """
    logger.info("Used Realtime settings: %s", engine_config)
    global realtime_servicer
    if realtime_servicer is None:
        realtime_servicer = RealtimeServicer(
            grpc_addr=engine_config.asr_grpc_addr,
            interim_results=engine_config.asr_interim_results,
            chat_base_url=engine_config.chat_base_url,
            chat_api_key=engine_config.chat_api_key,
            tts_base_url=engine_config.tts_base_url,
            tts_api_key=engine_config.tts_api_key,
        )


async def shutdown():
    """关闭 RealtimeServicer 及其资源"""
    global realtime_servicer
    if realtime_servicer is not None:
        await realtime_servicer.close()
        realtime_servicer = None
        logger.info("RealtimeServicer closed")


def get_realtime_servicer() -> RealtimeServicer:
    if realtime_servicer is None:
        raise RuntimeError("RealtimeServicer not configured. Call configure() first.")
    return realtime_servicer
