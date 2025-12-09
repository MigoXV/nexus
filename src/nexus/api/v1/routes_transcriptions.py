# src/ux_speech_gateway/api/v1/routes_transcriptions.py
from __future__ import annotations

import grpc
from fastapi import APIRouter, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse, PlainTextResponse

from ...services.transcriptions_service import TranscriptionsService

router = APIRouter()


def get_transcriptions_service() -> TranscriptionsService:
    return TranscriptionsService()


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    model: str = Form("ux_speech_grpc_proxy"),
    file: UploadFile = File(...),
    language: str = Form("zh-CN"),
    sample_rate: int = Form(16000),
    interim_results: bool = Form(False),
    hotwords: str = Form(""),
    hotword_bias: float = Form(0.0),
    grpc_host: str = Form("39.106.1.132"),
    grpc_port: int = Form(30029),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),  # 兼容 OpenAI 形状，当前不使用
    svc: TranscriptionsService = Depends(get_transcriptions_service),
):
    """
    Mimic OpenAI speech_to_text endpoint.

    Accepts a multipart file upload and optional form params.
    Forwards audio to the remote gRPC `UxSpeech` service and returns
    a JSON with the combined final transcript.
    """
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
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            timeout_s=180.0,
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

    # default json
    result = {"text": combined, "model": model}
    return JSONResponse(status_code=200, content=result)
