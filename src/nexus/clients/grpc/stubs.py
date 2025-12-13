from __future__ import annotations

from dataclasses import dataclass

import grpc

from .channel import create_insecure_channel
from ...generated import ux_speech_pb2_grpc


@dataclass
class UxSpeechClient:
    """
    Thin wrapper for UxSpeech gRPC stub.
    """

    host: str
    port: int
    channel: grpc.Channel | None = None
    stub: ux_speech_pb2_grpc.UxSpeechStub | None = None
#额外的初始化逻辑放在 __post_init__ 里，便于区分，dataclass中自动生成init，运行后，直接运行post-init,
    def __post_init__(self) -> None:
        if self.channel is None:
            self.channel = create_insecure_channel(self.host, self.port)
        if self.stub is None:
            self.stub = ux_speech_pb2_grpc.UxSpeechStub(self.channel)

    def close(self) -> None:
        if self.channel:
            self.channel.close()
