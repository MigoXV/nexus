"""
Nexus CLI 应用 - Typer 入口
"""

import logging
from contextlib import asynccontextmanager

import typer
import uvicorn
from fastapi import FastAPI
from pathlib import Path
from nexus.api.v1 import chat as chat_api
from nexus.api.v1 import depends
from nexus.api.v1 import realtime as realtime_api
from nexus.api.v1 import transcribe as transcribe_api
from nexus.api.v1 import tts as tts_api
from omegaconf import OmegaConf, ListConfig
from nexus.configs.config import NexusConfig

app = typer.Typer(
    name="nexus",
    help="Nexus - 语音识别服务",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def create_fastapi_app(
    engine_config: NexusConfig,
) -> FastAPI:
    """创建 FastAPI 应用实例"""
    # 配置 gRPC 地址 (同步服务，可以在启动前初始化)
    transcribe_api.configure(
        grpc_addr=engine_config.asr_grpc_addr,
        interim_results=engine_config.asr_interim_results,
    )
    # 配置 Chat API
    depends.configure_chat(
        base_url=engine_config.chat_base_url, api_key=engine_config.chat_api_key
    )
    # 配置 TTS API
    tts_api.configure(
        base_url=engine_config.tts_base_url, api_key=engine_config.tts_api_key
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        FastAPI lifespan 上下文管理器。
        在此处初始化需要绑定到 uvicorn event loop 的异步资源（如 grpc.aio channel）。
        """
        # 在 uvicorn event loop 中初始化 Realtime API（包含 grpc.aio channel）
        realtime_api.configure(engine_config=engine_config)
        logger.info("Realtime API initialized in uvicorn event loop")
        yield
        # 关闭时清理资源
        await realtime_api.shutdown()
        logger.info("Realtime API shutdown complete")

    fastapi_app = FastAPI(
        title="Nexus ASR API",
        description="语音识别服务 API - 兼容 OpenAI Whisper API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 注册路由
    fastapi_app.include_router(transcribe_api.router, prefix="/v1")
    fastapi_app.include_router(realtime_api.router, prefix="/v1")
    fastapi_app.include_router(chat_api.router, prefix="/v1")
    fastapi_app.include_router(tts_api.router, prefix="/v1")

    @fastapi_app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return fastapi_app


@app.command()
def serve(
    engine_config: Path = typer.Argument(
        ...,
        help="引擎配置文件路径 (YAML 格式)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        envvar="NEXUS_ENGINE_CONFIG",
    ),
    log_level: str = typer.Option(
        "info",
        help="日志级别 (debug, info, warning, error)",
        envvar="NEXUS_LOG_LEVEL",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        help="HTTP 服务监听地址",
        envvar="NEXUS_HOST",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="HTTP 服务监听端口",
        envvar="NEXUS_PORT",
    ),
    ssl_certfile: Path = typer.Option(
        None,
        help="SSL 证书文件路径 (server.crt)",
        envvar="NEXUS_SSL_CERTFILE",
    ),
    ssl_keyfile: Path = typer.Option(
        None,
        help="SSL 私钥文件路径 (server.key)",
        envvar="NEXUS_SSL_KEYFILE",
    ),
    ssl_ca_certs: Path = typer.Option(
        None,
        help="CA 证书链文件（可选）",
        envvar="NEXUS_SSL_CA_CERTS",
    ),
):
    """
    启动 HTTP API 服务器
    """
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Nexus API server on {host}:{port}")
    logger.info(f"using engine config: {engine_config}")
    logger.info(
        f"ssl_certfile: {ssl_certfile}, ssl_keyfile: {ssl_keyfile}, ssl_ca_certs: {ssl_ca_certs}"
    )

    engine_config: ListConfig = OmegaConf.load(engine_config)
    engine_config = OmegaConf.to_container(engine_config, resolve=True)
    engine_config = NexusConfig(**engine_config)

    fastapi_app = create_fastapi_app(
        engine_config=engine_config,
    )

    # 启动 uvicorn
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        ssl_ca_certs=ssl_ca_certs,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ws_ping_interval=20,  # 每 20 秒发送一次 ping 保持连接
        ws_ping_timeout=60,   # 60 秒内无 pong 响应才断开
    )


if __name__ == "__main__":
    app()
