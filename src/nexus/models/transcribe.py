"""
转录相关的数据模型
"""

from typing import List, Optional

from pydantic import BaseModel


class Settings:
    """应用配置"""

    grpc_addr: str = "localhost:50051"
    interim_results: bool = False


class TranscriptionBase64Request(BaseModel):
    """Base64 编码音频的请求体"""

    audio: str  # Base64 编码的 PCM 音频
    model: str = "whisper-1"
    language: Optional[str] = "zh-CN"
    sample_rate: int = 16000
    hotwords: Optional[List[str]] = None
