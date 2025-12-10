#在处理多层复杂api时，通过router进行模块化管理，通常每个router对应一个文件，内部注册多个路由，通过文件结构可以清晰的概括路由结构，这也是
#v1文件夹的作用，统一管理各层路由，参差多时可以逐步INCLUDE。

from __future__ import annotations

from typing import Optional

import grpc
from fastapi import APIRouter, File, UploadFile, Form, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ...services.transcriptions_service import TranscriptionsService
from ...core.config import settings

router = APIRouter()
# prefix 决定“URL 长什么样”
# tags 决定“文档里怎么分类展示”
#可以在router_include时写入或覆盖prefix和tags

#把一些可能多次利用或需要依赖注入的服务打包成类或函数，方便在路由函数中使用通过Depends注入
def get_transcriptions_service(request: Request) -> TranscriptionsService:
    default_client = getattr(request.app.state, "default_grpc_client", None)
    return TranscriptionsService(default_client=default_client)


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form("ux_speech_grpc_proxy"),
    file: UploadFile = File(...),
    language: str = Form("zh-CN"),
    sample_rate: int = Form(16000),
    interim_results: bool = Form(False),
    hotwords: str = Form(""),
    hotword_bias: float = Form(0.0),

    # 关键：不在路由层写死默认值
    grpc_host: Optional[str] = Form(None),
    grpc_port: Optional[int] = Form(None),
    grpc_timeout_s: Optional[float] = Form(None),

    response_format: str = Form("json"),
    temperature: float = Form(0.0),  # 兼容 OpenAI 形状，当前不使用
    svc: TranscriptionsService = Depends(get_transcriptions_service),
):
    try:
        audio_bytes = await file.read()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"failed to read uploaded file: {e}"},
        )

    try:
        combined = await svc.transcribe(
            audio_bytes=audio_bytes,
            language=language,
            sample_rate=sample_rate,
            interim_results=interim_results,
            hotwords=hotwords,
            hotword_bias=hotword_bias,
            grpc_host=grpc_host,              # 允许覆盖
            grpc_port=grpc_port,              # 允许覆盖
            timeout_s=grpc_timeout_s,         # 允许覆盖
        )

    except grpc.RpcError as rpc_err:
        err_message = rpc_err.details() if hasattr(rpc_err, "details") else str(rpc_err)
        return JSONResponse(
            status_code=502,
            content={"error": "gRPC error", "detail": err_message},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "internal_proxy_error", "detail": str(e)},
        )

    fmt = (response_format or "json").lower()

    if fmt in ("text", "txt", "plain"):
        return PlainTextResponse(content=combined, status_code=200)

    if fmt == "srt":
        srt = "1\n00:00:00,000 --> 00:10:00,000\n" + (combined or "") + "\n"
        return PlainTextResponse(content=srt, media_type="text/plain", status_code=200)

    if fmt == "vtt":
        vtt = "WEBVTT\n\n00:00.000 --> 00:10.000\n" + (combined or "") + "\n"
        return PlainTextResponse(content=vtt, media_type="text/vtt", status_code=200)

    return JSONResponse(
        status_code=200,
        content={"text": combined, "model": model, "default_grpc": f"{settings.default_grpc_host}:{settings.default_grpc_port}"},
    )
