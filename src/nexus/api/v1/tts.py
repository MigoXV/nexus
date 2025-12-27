"""
FastAPI 路由 - Text-to-Speech API
兼容 OpenAI TTS API 格式（透明代理模式）
"""

import logging
from dataclasses import dataclass
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nexus.inferencers.tts.inferencer import Inferencer


router = APIRouter(prefix="/audio", tags=["Audio"])

logger = logging.getLogger(__name__)


# ============== 配置 ==============


@dataclass
class TTSSettings:
    """TTS API 配置"""
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "no-key"


_settings: Optional[TTSSettings] = None


def get_settings() -> TTSSettings:
    if _settings is None:
        raise RuntimeError("TTS settings not configured. Call configure() first.")
    return _settings


def configure(base_url: str, api_key: str):
    """配置全局设置"""
    global _settings
    _settings = TTSSettings(base_url=base_url, api_key=api_key)


def get_inferencer(
    settings: Annotated[TTSSettings, Depends(get_settings)],
) -> Inferencer:
    return Inferencer(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )


# ============== 请求模型 ==============


class CreateSpeechRequest(BaseModel):
    """TTS 请求模型，兼容 OpenAI API"""
    model: str = Field(default="tts-1", description="TTS 模型名称")
    input: str = Field(..., description="要转换为语音的文本", max_length=4096)
    voice: str = Field(default="alloy", description="语音类型")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3", description="音频输出格式"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速")


# 音频格式对应的 MIME 类型
AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


# ============== API 端点 ==============


@router.post("/speech")
async def create_speech(
    request: CreateSpeechRequest,
    inferencer: Annotated[Inferencer, Depends(get_inferencer)],
):
    """
    创建语音（Text-to-Speech）

    兼容 OpenAI TTS API 格式
    默认返回流式音频响应
    """
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    mime_type = AUDIO_MIME_TYPES.get(request.response_format, "audio/mpeg")

    try:
        # 使用流式响应
        def generate_audio():
            yield from inferencer.speech_stream(
                input=request.input,
                model=request.model,
                voice=request.voice,
                response_format=request.response_format,
                speed=request.speed,
            )

        return StreamingResponse(
            generate_audio(),
            media_type=mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"',
                "Transfer-Encoding": "chunked",
            },
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
