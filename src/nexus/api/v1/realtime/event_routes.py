"""
Realtime API 事件路由处理模块
处理各类 WebSocket 事件
"""

import logging
from typing import List

from fastapi import WebSocket
from nexus.servicers.realtime.build_events import build_session_updated
from nexus.sessions import RealtimeSession

logger = logging.getLogger(__name__)


async def handle_session_event(
    websocket: WebSocket,
    event: dict,
    names: List[str],
    realtime_session: RealtimeSession,
    model: str,
    send_event_func,
) -> bool:
    if len(names) < 2:
        logger.warning(f"Invalid session event format: {'.'.join(names)}")
        return False
    action = names[1]
    if action == "update":
        return await _handle_session_update(
            websocket, event, realtime_session, model, send_event_func
        )
    else:
        logger.warning(f"Unhandled session event: {'.'.join(names)}")
        return False


async def _handle_session_update(
    websocket: WebSocket,
    event: dict,
    realtime_session: RealtimeSession,
    model: str,
    send_event_func,
) -> bool:
    """
    处理 session.update 事件
    """
    session_settings = event.get("session", {})
    output_modalities = (
        session_settings.get("output_modalities", ["text"])
        if session_settings
        else ["text"]
    )
    realtime_session.update_output_modalities(output_modalities)
    logger.info(f"Session update received. output_modalities: {output_modalities}")

    # 发送 session.updated 事件
    await send_event_func(
        websocket,
        build_session_updated(realtime_session.session_id, model, output_modalities),
    )
    return True
