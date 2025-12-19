"""
FastAPI 路由 - 音频转录 API
兼容 OpenAI Whisper API 格式
"""

import io
import json
from typing import Annotated, Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai.types.audio import Transcription

from nexus.models.transcribe import Settings, TranscriptionBase64Request
from nexus.servicers.transcribe import TranscribeService


router = APIRouter(prefix="/audio", tags=["Audio"])


# ============== 配置依赖 ==============


_settings = Settings()


def get_settings() -> Settings:
    return _settings


def configure(grpc_addr: str):
    """配置全局设置"""
    _settings.grpc_addr = grpc_addr


def get_transcribe_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> TranscribeService:
    return TranscribeService(grpc_addr=settings.grpc_addr)


# ============== API 端点 ==============


def _generate_sse_stream(service: TranscribeService, pcm_data: bytes, sample_rate: int, language: str):
    """
    生成 SSE 流式响应（兼容 OpenAI 流式格式）
    """
    for text in service.transcribe_pcm_stream(
        pcm_data=pcm_data,
        sample_rate=sample_rate,
        language=language,
    ):
        # OpenAI 兼容的流式响应格式
        event_data = {
            "type": "transcript.text.delta",
            "text": text,
        }
        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
    
    # 发送结束事件
    yield "data: [DONE]\n\n"


@router.post("/transcriptions")
async def create_transcription(
    service: Annotated[TranscribeService, Depends(get_transcribe_service)],
    file: Annotated[Optional[UploadFile], File()] = None,
    model: Annotated[str, Form()] = "whisper-1",
    language: Annotated[Optional[str], Form()] = "zh-CN",
    sample_rate: Annotated[int, Form()] = 16000,
    stream: Annotated[bool, Form()] = False,
):
    """
    创建音频转录 (multipart/form-data)

    兼容 OpenAI Audio API 格式，接收上传的音频文件（支持 wav/mp3/m4a 等格式）
    支持 stream=True 参数返回 SSE 流式响应
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        # 读取上传的文件内容
        file_content = await file.read()
        
        # 使用 soundfile 解码音频文件（支持 wav/flac/ogg 等格式）
        audio_data, file_sample_rate = sf.read(
            io.BytesIO(file_content), dtype="int16"
        )
        
        # 如果采样率不是目标采样率，进行重采样
        if file_sample_rate != sample_rate:
            # 转换为 float32 进行重采样
            audio_float = audio_data.astype(np.float32) / 32768.0
            audio_float = librosa.resample(
                audio_float, orig_sr=file_sample_rate, target_sr=sample_rate
            )
            # 归一化并转换回 int16
            audio_float = audio_float / (np.max(np.abs(audio_float)) + 1e-8)
            audio_data = (audio_float * 32767).astype(np.int16)
        
        # 转换为 PCM bytes
        pcm_data = audio_data.tobytes()

        # 流式响应
        if stream:
            return StreamingResponse(
                _generate_sse_stream(
                    service=service,
                    pcm_data=pcm_data,
                    sample_rate=sample_rate,
                    language=language or "zh-CN",
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 非流式响应
        result = service.transcribe_pcm(
            pcm_data=pcm_data,
            sample_rate=sample_rate,
            language=language or "zh-CN",
        )

        return Transcription(text=result.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/transcriptions/base64", response_model=Transcription)
async def create_transcription_base64(
    service: Annotated[TranscribeService, Depends(get_transcribe_service)],
    request: TranscriptionBase64Request,
):
    """
    创建音频转录 (JSON + Base64)

    接收 Base64 编码的 PCM 音频数据
    """
    try:
        result = service.transcribe_base64(
            base64_data=request.audio,
            sample_rate=request.sample_rate,
            language=request.language or "zh-CN",
            hotwords=request.hotwords,
        )

        return Transcription(text=result.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/transcriptions/raw", response_model=Transcription)
async def create_transcription_raw(
    service: Annotated[TranscribeService, Depends(get_transcribe_service)],
    body: Annotated[bytes, Body(media_type="application/octet-stream")],
    language: str = "zh-CN",
    sample_rate: int = 16000,
):
    """
    创建音频转录 (原始二进制)

    直接接收 PCM 音频二进制数据
    """
    try:
        result = service.transcribe_pcm(
            pcm_data=body,
            sample_rate=sample_rate,
            language=language,
        )

        return Transcription(text=result.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
