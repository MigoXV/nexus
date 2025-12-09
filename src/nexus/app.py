from __future__ import annotations

from fastapi import FastAPI

from .api.v1.routes_transcriptions import router as transcriptions_router


def create_app() -> FastAPI:
    app = FastAPI(title="UxSpeech HTTP -> gRPC proxy")

    # 你后续如果要加 /api/v1 前缀，可在这里统一加
    app.include_router(transcriptions_router)

    return app
