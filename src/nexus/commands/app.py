"""
Nexus CLI 应用 - Typer 入口
"""

import logging

import typer
import uvicorn
from fastapi import FastAPI

from nexus.api.v1 import chat as chat_api
from nexus.api.v1 import depends
from nexus.api.v1 import realtime as realtime_api
from nexus.api.v1 import transcribe as transcribe_api
from nexus.api.v1 import tts as tts_api

app = typer.Typer(
    name="nexus",
    help="Nexus - 语音识别服务",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def create_fastapi_app(
    grpc_addr: str,
    chat_base_url: str,
    chat_api_key: str,
    tts_base_url: str,
    tts_api_key: str,
    interim_results: bool = False,
) -> FastAPI:
    """创建 FastAPI 应用实例"""
    # 配置 gRPC 地址
    transcribe_api.configure(grpc_addr=grpc_addr, interim_results=interim_results)
    realtime_api.configure(grpc_addr=grpc_addr, interim_results=interim_results)
    # 配置 Chat API
    depends.configure_chat(base_url=chat_base_url, api_key=chat_api_key)
    # 配置 TTS API
    tts_api.configure(base_url=tts_base_url, api_key=tts_api_key)

    fastapi_app = FastAPI(
        title="Nexus ASR API",
        description="语音识别服务 API - 兼容 OpenAI Whisper API",
        version="0.1.0",
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
    grpc_addr: str = typer.Option(
        "localhost:50051",
        "--grpc-addr",
        "-g",
        help="gRPC ASR 服务地址",
        envvar="NEXUS_GRPC_ADDR",
    ),
    interim_results: bool = typer.Option(
        False,
        "--interim-results/--no-interim-results",
        help="是否启用中间结果返回",
        envvar="NEXUS_INTERIM_RESULTS",
    ),
    # Chat API 参数
    chat_base_url: str = typer.Option(
        "http://localhost:8080/v1",
        "--chat-base-url",
        help="Chat 后端 API 地址",
        envvar="NEXUS_CHAT_BASE_URL",
    ),
    chat_api_key: str = typer.Option(
        "no-key",
        "--chat-api-key",
        help="Chat 后端 API 密钥",
        envvar="NEXUS_CHAT_API_KEY",
    ),
    # TTS API 参数
    tts_base_url: str = typer.Option(
        "http://localhost:8080/v1",
        "--tts-base-url",
        help="TTS 后端 API 地址",
        envvar="NEXUS_TTS_BASE_URL",
    ),
    tts_api_key: str = typer.Option(
        "no-key",
        "--tts-api-key",
        help="TTS 后端 API 密钥",
        envvar="NEXUS_TTS_API_KEY",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        "-l",
        help="日志级别 (debug, info, warning, error)",
        envvar="NEXUS_LOG_LEVEL",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
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
    # 🔐 新增 SSL 参数
    ssl_certfile: str = typer.Option(
        None,
        "--ssl-certfile",
        help="SSL 证书文件路径 (server.crt)",
        envvar="NEXUS_SSL_CERTFILE",
    ),
    ssl_keyfile: str = typer.Option(
        None,
        "--ssl-keyfile",
        help="SSL 私钥文件路径 (server.key)",
        envvar="NEXUS_SSL_KEYFILE",
    ),
    ssl_ca_certs: str = typer.Option(
        None,
        "--ssl-ca-certs",
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
    logger.info(f"gRPC ASR backend: {grpc_addr}")
    logger.info(f"Chat backend: {chat_base_url}")
    logger.info(f"TTS backend: {tts_base_url}")
    logger.info(
        f"ssl_certfile: {ssl_certfile}, ssl_keyfile: {ssl_keyfile}, ssl_ca_certs: {ssl_ca_certs}"
    )
    # 创建 FastAPI 应用
    fastapi_app = create_fastapi_app(
        grpc_addr=grpc_addr,
        chat_base_url=chat_base_url,
        chat_api_key=chat_api_key,
        tts_base_url=tts_base_url,
        tts_api_key=tts_api_key,
        interim_results=interim_results,
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
    )


if __name__ == "__main__":
    app()
