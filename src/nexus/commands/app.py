"""
Nexus CLI 应用 - Typer 入口
"""

import logging

import typer
import uvicorn
from fastapi import FastAPI

from nexus.api.v1 import transcribe as transcribe_api

app = typer.Typer(
    name="nexus",
    help="Nexus - 语音识别服务",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def create_fastapi_app(grpc_addr: str) -> FastAPI:
    """创建 FastAPI 应用实例"""
    # 配置 gRPC 地址
    transcribe_api.configure(grpc_addr=grpc_addr)

    fastapi_app = FastAPI(
        title="Nexus ASR API",
        description="语音识别服务 API - 兼容 OpenAI Whisper API",
        version="0.1.0",
    )

    # 注册路由
    fastapi_app.include_router(transcribe_api.router, prefix="/v1")

    @fastapi_app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return fastapi_app


@app.command()
def serve(
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
    grpc_addr: str = typer.Option(
        "localhost:50051",
        "--grpc-addr",
        "-g",
        help="gRPC ASR 服务地址",
        envvar="NEXUS_GRPC_ADDR",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        "-l",
        help="日志级别 (debug, info, warning, error)",
        envvar="NEXUS_LOG_LEVEL",
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

    # 创建 FastAPI 应用
    fastapi_app = create_fastapi_app(grpc_addr=grpc_addr)

    # 启动 uvicorn
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    app()
