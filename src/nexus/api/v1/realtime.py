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
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from nexus.inferencers.asr.inferencer import Inferencer
from nexus.models.transcribe import Settings

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


# ============== 数据结构 ==============


@dataclass
class RealtimeSession:
    """实时会话状态"""

    session_id: str
    model: str = "gpt-4o-realtime-preview"
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    sample_rate: int = 16000
    language: str = "zh-CN"
    is_streaming: bool = False
    stream_done: bool = False


# ============== WebSocket 端点 ==============


@router.websocket("/realtime")
async def realtime_endpoint(
    websocket: WebSocket,
    model: str = Query(default="gpt-4o-realtime-preview"),
    settings: Settings = Depends(get_settings),
):
    """
    OpenAI Realtime API 兼容的 WebSocket 端点

    处理以下事件类型：
    - input_audio_buffer.append: 追加音频数据
    - input_audio_buffer.commit: 提交音频缓冲区进行转录
    - response.create: 创建响应（触发转录）
    """
    await websocket.accept()

    session = RealtimeSession(
        session_id=str(uuid.uuid4()),
        model=model,
    )

    # 发送 session.created 事件
    await _send_event(
        websocket,
        {
            "type": "session.created",
            "session": {
                "id": session.session_id,
                "model": session.model,
                "input_audio_format": "pcm16",
            },
        },
    )

    # 用于接收转录结果的队列
    result_queue: asyncio.Queue = asyncio.Queue()

    # 后台任务：发送转录结果
    send_results_task: Optional[asyncio.Task] = None
    response_id: Optional[str] = None
    item_id: Optional[str] = None

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
                        session.audio_queue.put(audio_bytes)
                        logger.debug(
                            f"Audio chunk added: {len(audio_bytes)} bytes, queue size: {session.audio_queue.qsize()}"
                        )

                elif event_type == "input_audio_buffer.commit":
                    # 标记音频流结束
                    session.stream_done = True
                    session.audio_queue.put(None)  # 发送结束标记
                    logger.info("Audio buffer committed (stream done)")

                elif event_type == "response.create":
                    # 开始流式转录
                    response_id = f"resp_{uuid.uuid4().hex[:24]}"
                    item_id = f"item_{uuid.uuid4().hex[:24]}"

                    logger.info(
                        f"Starting streaming transcription: response_id={response_id}"
                    )

                    # 发送 response.created 事件
                    await _send_event(
                        websocket,
                        {
                            "type": "response.created",
                            "response": {
                                "id": response_id,
                                "status": "in_progress",
                            },
                        },
                    )

                    session.is_streaming = True

                    # 启动转录线程
                    loop = asyncio.get_event_loop()
                    transcription_thread = threading.Thread(
                        target=_transcribe_stream_worker,
                        args=(
                            session,
                            settings,
                            result_queue,
                            loop,
                            response_id,
                            item_id,
                        ),
                        daemon=True,
                    )
                    transcription_thread.start()

                    # 启动后台任务发送转录结果（不阻塞主循环）
                    send_results_task = asyncio.create_task(
                        _send_streaming_results(
                            websocket, session, result_queue, response_id, item_id
                        )
                    )

                elif event_type == "session.update":
                    # 更新会话配置
                    session_config = event.get("session", {})
                    if "input_audio_transcription" in session_config:
                        transcription_config = session_config[
                            "input_audio_transcription"
                        ]
                        if "language" in transcription_config:
                            session.language = transcription_config["language"]
                    logger.info(f"Session updated: language={session.language}")

                else:
                    logger.warning(f"Unknown event type: {event_type}")
            else:
                # 消息接收被取消，重新循环
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session.session_id}")
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


async def _send_streaming_results(
    websocket: WebSocket,
    session: RealtimeSession,
    result_queue: asyncio.Queue,
    response_id: str,
    item_id: str,
):
    """后台任务：持续发送转录结果（边收边发）"""
    transcripts: List[str] = []
    done = False

    while not done:
        try:
            # 等待结果，超时 50ms 以便快速响应
            event = await asyncio.wait_for(result_queue.get(), timeout=0.05)

            if event is None:
                # 结束标记
                done = True
            elif event.get("type") == "transcript":
                # 转录结果
                transcript = event.get("text", "")
                is_final = event.get("is_final", False)

                if transcript:
                    # 立即发送增量转录结果
                    await _send_event(
                        websocket,
                        {
                            "type": "response.audio_transcript.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "delta": transcript,
                        },
                    )

                if is_final and transcript:
                    transcripts.append(transcript)
            elif event.get("type") == "error":
                # 错误
                await _send_event(
                    websocket,
                    {
                        "type": "error",
                        "error": {
                            "type": "transcription_error",
                            "message": event.get("message", "Unknown error"),
                        },
                    },
                )
                done = True
        except asyncio.TimeoutError:
            # 超时，继续等待
            pass

    # 发送完成的转录结果
    full_transcript = "".join(transcripts)
    await _send_event(
        websocket,
        {
            "type": "response.audio_transcript.done",
            "response_id": response_id,
            "item_id": item_id,
            "transcript": full_transcript,
        },
    )

    # 发送 response.done 事件
    await _send_event(
        websocket,
        {
            "type": "response.done",
            "response": {
                "id": response_id,
                "status": "completed",
                "output": [
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "audio_transcript",
                                "transcript": full_transcript,
                            }
                        ],
                    }
                ],
            },
        },
    )

    # 清理会话状态
    session.is_streaming = False
    session.stream_done = False
    # 清空队列，为下一轮准备
    while not session.audio_queue.empty():
        try:
            session.audio_queue.get_nowait()
        except queue.Empty:
            break

    logger.info(
        f"Transcription completed: {full_transcript[:100] if full_transcript else '(empty)'}..."
    )


def _transcribe_stream_worker(
    session: RealtimeSession,
    settings: Settings,
    result_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    response_id: str,
    item_id: str,
):
    """后台线程：从音频队列读取数据并进行流式转录"""

    def audio_generator():
        """从队列生成音频块"""
        while True:
            try:
                chunk = session.audio_queue.get(timeout=0.5)
                if chunk is None:
                    # 结束标记
                    break
                yield np.frombuffer(chunk, dtype=np.int16)
            except queue.Empty:
                # 队列为空但未结束，继续等待
                if session.stream_done:
                    break
                continue

    try:
        with Inferencer(settings.grpc_addr) as inferencer:
            for transcript, is_final in inferencer.transcribe(
                audio=audio_generator(),
                sample_rate=session.sample_rate,
                language_code=session.language,
                interim_results=settings.interim_results,
            ):
                # 将结果放入队列
                loop.call_soon_threadsafe(
                    result_queue.put_nowait,
                    {"type": "transcript", "text": transcript, "is_final": is_final},
                )
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        loop.call_soon_threadsafe(
            result_queue.put_nowait, {"type": "error", "message": str(e)}
        )
    finally:
        # 发送结束标记
        loop.call_soon_threadsafe(result_queue.put_nowait, None)
