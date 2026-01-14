"""
FastAPI WebSocket 路由 - OpenAI Realtime API 兼容
处理实时语音转录的 WebSocket 连接
"""

import asyncio
import base64
import json
import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from nexus.inferencers.asr.inferencer import Inferencer
from nexus.models.transcribe import Settings
from nexus.servicers.realtime.servicer import RealtimeServicer
from nexus.servicers.realtime.session import RealtimeSession

router = APIRouter(tags=["Realtime"])

logger = logging.getLogger(__name__)


# ============== 配置依赖 ==============

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings not configured. Call configure() first.")
    return _settings


def configure(grpc_addr: str, interim_results: bool = False):
    """配置全局设置"""
    global _settings
    if _settings is None:
        _settings = Settings()
    _settings.grpc_addr = grpc_addr
    _settings.interim_results = interim_results


# ============== WebSocket 端点 ==============


@router.websocket("/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str = Query(default="gpt-4o-realtime-preview"),
    settings: Settings = Depends(get_settings),
):
    # 发送 session.created 事件
    await websocket.accept()
    await _send_event(
        websocket,
        {
            "type": "session.created",
            "session": {
                "id": "dummy-session-id",
                "model": "dummy-model",
                "input_audio_format": "pcm16",
            },
        },
    )

    # 后台任务：发送转录结果
    send_results_task: Optional[asyncio.Task] = None
    # audio: np.ndarray = np.array([], dtype=np.int16)
    session: RealtimeSession = RealtimeSession(model=model)
    servicer: RealtimeServicer = RealtimeServicer(
        grpc_addr=settings.grpc_addr, interim_results=settings.interim_results
    )
    threading.Thread(
        target=servicer.realtime_worker,
        args=(session,),
        daemon=True,
    ).start()

    # 启动异步任务：从 result_queue 读取结果并发送
    send_results_task = asyncio.create_task(
        _send_transcription_results(websocket, session)
    )

    try:
        while True:
            # 同时处理：接收消息 和 发送转录结果
            receive_task = asyncio.create_task(websocket.receive_text())
            # 构建等待任务列表
            pending_tasks = {receive_task}
            if send_results_task and not send_results_task.done():
                pending_tasks.add(send_results_task)
            # 同时等待消息和发送结果
            done, pending = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # 检查发送任务是否完成
            if send_results_task in done:
                send_results_task = None
            # 检查是否收到消息
            if receive_task in done:
                data = receive_task.result()
                event = json.loads(data)
                event_type = event.get("type", "")
                logger.debug(f"Received event: {event_type}")
                if event_type == "input_audio_buffer.append":
                    # 追加音频数据到队列
                    audio_base64 = event.get("audio", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                        # audio = np.concatenate((audio, audio_chunk))
                        # sf.write("data-bin/base64_audio.wav", audio, samplerate=16000)
                        session.audio_queue.put(audio_chunk)
                elif event_type == "response.cancel":
                    print("Response cancelled")
                    # # 清空音频队列
                    # while not session.audio_queue.empty():
                    #     try:
                    #         session.audio_queue.get_nowait()
                    #     except queue.Empty:
                    #         break

                    # logger.info("Response cancelled, session reset")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await _send_event(
            websocket,
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": str(e),
                },
            },
        )


async def _send_event(websocket: WebSocket, event: dict):
    """发送事件到 WebSocket"""
    await websocket.send_text(json.dumps(event, ensure_ascii=False))


async def _send_transcription_results(websocket: WebSocket, session: RealtimeSession):
    """异步任务：从 result_queue 读取转录结果并发送到 WebSocket"""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # 使用 run_in_executor 异步等待同步队列
            result = await loop.run_in_executor(
                None, lambda: session.result_queue.get(timeout=0.1)
            )
            transcript, is_final = result
            event = {
                "type": "response.output_audio_transcript.delta",
                "response_id": "dummy-response-id",
                "item_id": "dummy-response-id",
                "delta": transcript,
            }
            await _send_event(websocket, event)
        except queue.Empty:
            # 队列为空，继续等待
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.exception(f"Error sending transcription result: {e}")
            break
