from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from openai.types.realtime import (
    RealtimeFunctionTool,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)
from openai.types.realtime.realtime_tools_config_union import Mcp

from nexus.application.realtime.emitters.response_contexts import McpListToolsContext
from nexus.application.realtime.orchestrators.response_orchestrator import (
    process_chat_stream,
)
from nexus.application.realtime.text_processing import PreparedRealtimeUserTurn
from nexus.application.realtime.orchestrators.tool_call_orchestrator import (
    execute_mcp_tool_call,
)
from nexus.application.realtime.orchestrators.transcription_worker import (
    run_transcription_worker,
)
from nexus.application.realtime.protocol.ids import event_id
from nexus.domain.realtime import RealtimeSessionState
from nexus.infrastructure.asr import AsyncInferencer as ASRInferencer
from nexus.infrastructure.chat import AsyncInferencer as AsyncChatInferencer
from nexus.infrastructure.tts import TTSBackend
from nexus.infrastructure.mcp import McpServerConfig
from nexus.sessions.chat_session import AsyncChatSession

logger = logging.getLogger(__name__)


@dataclass
class EffectiveResponseConfig:
    modalities: list[str]
    audio_format_type: str
    audio_voice: str
    audio_speed: float


class RealtimeApplicationService:
    REALTIME_PCM_FORMAT = "audio/pcm"
    REALTIME_AUDIO_SAMPLE_RATE = 24000

    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
        asr_hide_metadata: bool = True,
        chat_base_url: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        tts_backend: Optional[TTSBackend] = None,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.asr_hide_metadata = asr_hide_metadata
        self.asr_inferencer = ASRInferencer(self.grpc_addr)
        self.chat_inferencer = (
            AsyncChatInferencer(api_key=chat_api_key, base_url=chat_base_url)
            if chat_api_key
            else None
        )
        self.tts_backend = tts_backend

    async def close(self) -> None:
        if self.asr_inferencer:
            await self.asr_inferencer.close()
        if self.chat_inferencer:
            await self.chat_inferencer.close()
        if self.tts_backend:
            await self.tts_backend.close()

    def create_session(
        self,
        *,
        writer,
        output_modalities: Sequence[str],
        tools: Sequence[RealtimeFunctionTool],
        chat_model: str,
    ) -> RealtimeSessionState:
        if "transcribe" not in chat_model.lower() and self.chat_inferencer is None:
            raise RuntimeError(
                "Chat inferencer is not configured. Set chat_api_key/chat_base_url for realtime chat models."
            )
        normalized_modalities = self._normalize_output_modalities(list(output_modalities or ["text"]))
        if "audio" in normalized_modalities and self.tts_backend is None:
            raise RuntimeError(
                "TTS backend is not configured. Configure tts_backend for realtime audio output."
            )
        chat_session = AsyncChatSession(chat_inferencer=self.chat_inferencer)
        return RealtimeSessionState(
            chat_session=chat_session,
            chat_model=chat_model,
            writer=writer,
            output_modalities=normalized_modalities,
            tools=list(tools),
        )

    async def emit_session_created(self, session: RealtimeSessionState, model: str) -> None:
        await session.send_event(
            SessionCreatedEvent(
                type="session.created",
                event_id=event_id(),
                session=self._session_payload(
                    session=session,
                    model=model,
                ),
            )
        )

    async def apply_session_update(self, session: RealtimeSessionState, update, *, model: str) -> None:
        if model:
            session.chat_model = model

        try:
            self._validate_audio_input_update(update)
        except ValueError as exc:
            await session.writer.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_audio_input_format",
            )
            return

        output_modalities = getattr(update, "output_modalities", None)
        if output_modalities is not None:
            try:
                normalized_modalities = self._normalize_output_modalities(list(output_modalities))
            except ValueError as exc:
                await session.writer.send_error(
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_output_modalities",
                )
            else:
                if "audio" in normalized_modalities and self.tts_backend is None:
                    await session.writer.send_error(
                        message=(
                            "TTS backend is not configured. "
                            "Configure tts_backend for realtime audio output."
                        ),
                        error_type="invalid_request_error",
                        code="audio_output_not_configured",
                    )
                else:
                    session.update_output_modalities(normalized_modalities)

        try:
            self._apply_audio_output_update(session, update)
        except ValueError as exc:
            await session.writer.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_audio_output_format",
            )
            return

        raw_tools = getattr(update, "tools", None)
        if raw_tools is not None:
            function_tools, mcp_configs = self._split_tools(raw_tools)
            session.tools = function_tools
            await self._sync_mcp_servers(session, mcp_configs)

        await session.send_event(
            SessionUpdatedEvent(
                type="session.updated",
                event_id=event_id(),
                session=self._session_payload(session=session, model=model),
            )
        )

    async def start_transcription_worker(
        self,
        session: RealtimeSessionState,
        is_chat_model: bool,
    ) -> asyncio.Task:
        return asyncio.create_task(
            run_transcription_worker(
                inferencer=self.asr_inferencer,
                session=session,
                interim_results=self.interim_results,
                hide_metadata=self.asr_hide_metadata,
                is_chat_model=is_chat_model,
                chat_worker=self.chat_worker,
            )
        )

    async def chat_worker(self, session: RealtimeSessionState, user_turn: PreparedRealtimeUserTurn) -> None:
        chat_stream = session.chat(user_turn)
        response_cfg = self._resolve_response_config(session)
        if "audio" in response_cfg.modalities:
            if self.tts_backend is None:
                raise RuntimeError(
                    "TTS backend is not configured. Configure tts_backend for realtime audio output."
                )
            self._ensure_audio_output_supported(response_cfg.audio_format_type)
        result = await process_chat_stream(
            session=session,
            chat_stream=chat_stream,
            modalities=response_cfg.modalities,
            tts_backend=self.tts_backend,
            audio_output_format_type=response_cfg.audio_format_type,
            audio_output_voice=response_cfg.audio_voice,
            audio_output_speed=response_cfg.audio_speed,
        )

        if result.has_mcp_call and result.tool_call:
            await execute_mcp_tool_call(session=session, tool_call=result.tool_call)
        elif result.has_tool_call and result.tool_call:
            logger.info(
                "Function call sent: %s; waiting for function_call_output + response.create",
                result.tool_call.name,
            )

    async def generate_response(self, session: RealtimeSessionState, event=None) -> None:
        try:
            response_cfg = self._resolve_response_config(
                session=session,
                response=getattr(event, "response", None),
            )
        except ValueError as exc:
            await session.writer.send_error(
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_output_modalities",
            )
            return

        if "audio" in response_cfg.modalities:
            if self.tts_backend is None:
                await session.writer.send_error(
                    message=(
                        "TTS backend is not configured. "
                        "Configure tts_backend for realtime audio output."
                    ),
                    error_type="invalid_request_error",
                    code="audio_output_not_configured",
                )
                return
            try:
                self._ensure_audio_output_supported(response_cfg.audio_format_type)
            except ValueError as exc:
                await session.writer.send_error(
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="unsupported_audio_output_format",
                )
                return

        chat_stream = session.continue_conversation()
        result = await process_chat_stream(
            session=session,
            chat_stream=chat_stream,
            modalities=response_cfg.modalities,
            tts_backend=self.tts_backend,
            audio_output_format_type=response_cfg.audio_format_type,
            audio_output_voice=response_cfg.audio_voice,
            audio_output_speed=response_cfg.audio_speed,
        )

        if result.has_mcp_call and result.tool_call:
            await execute_mcp_tool_call(session=session, tool_call=result.tool_call)
        elif result.has_tool_call and result.tool_call:
            logger.info(
                "Function call sent: %s; waiting for function_call_output + response.create",
                result.tool_call.name,
            )

    async def handle_response_create(self, session: RealtimeSessionState, event) -> None:
        asyncio.create_task(self.generate_response(session, event))

    async def handle_response_cancel(self, session: RealtimeSessionState, _event) -> None:
        session.request_cancel(reason="client_cancelled")
        task = session.get_current_chat_task()
        if task and not task.done():
            task.cancel()

    async def handle_input_audio_commit(self, session: RealtimeSessionState, _event) -> None:
        # Current backend emits transcription from continuous stream; commit acts as a no-op marker.
        logger.debug("input_audio_buffer.commit received for session %s", session.session_id)

    async def close_session(self, session: RealtimeSessionState) -> None:
        await session.mcp_registry.close()

    def _normalize_output_modalities(self, modalities: Sequence[str]) -> list[str]:
        if not modalities:
            return ["text"]

        normalized = {str(modality).strip().lower() for modality in modalities if modality}
        if not normalized:
            return ["text"]

        unsupported = normalized - {"audio", "text"}
        if unsupported:
            raise ValueError(
                f"Unsupported output modalities: {sorted(unsupported)}. Allowed values: ['audio', 'text']"
            )

        if len(normalized) > 1:
            raise ValueError(
                "Passing both 'audio' and 'text' in output_modalities is not allowed. "
                "Please pass only one modality."
            )

        if "audio" in normalized:
            return ["audio"]
        return ["text"]

    def _ensure_audio_output_supported(self, format_type: str) -> None:
        if format_type != self.REALTIME_PCM_FORMAT:
            raise ValueError(
                f"Unsupported realtime audio output format '{format_type}'. "
                f"Only '{self.REALTIME_PCM_FORMAT}' is currently supported."
            )

    def _extract_format_type(self, format_config) -> Optional[str]:
        if format_config is None:
            return None
        if isinstance(format_config, str):
            return format_config
        if isinstance(format_config, dict):
            return format_config.get("type")
        return getattr(format_config, "type", None)

    def _extract_format_rate(self, format_config) -> Optional[int]:
        if format_config is None:
            return None
        if isinstance(format_config, dict):
            rate = format_config.get("rate")
        else:
            rate = getattr(format_config, "rate", None)
        if rate is None:
            return None
        return int(rate)

    def _model_to_dict(self, value) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return {}

    def _validate_audio_input_update(self, update) -> None:
        audio_config = update.get("audio") if isinstance(update, dict) else getattr(update, "audio", None)
        if audio_config is None:
            return

        input_config = (
            audio_config.get("input")
            if isinstance(audio_config, dict)
            else getattr(audio_config, "input", None)
        )
        if input_config is None:
            return

        input_data = self._model_to_dict(input_config)
        format_config = input_data.get("format")
        if format_config is None:
            return

        format_type = self._extract_format_type(format_config)
        format_rate = self._extract_format_rate(format_config)

        if format_type is not None and format_type != self.REALTIME_PCM_FORMAT:
            raise ValueError(
                f"Unsupported realtime audio input format '{format_type}'. "
                f"Only '{self.REALTIME_PCM_FORMAT}' is currently supported."
            )

        if format_rate is not None and format_rate != self.REALTIME_AUDIO_SAMPLE_RATE:
            raise ValueError(
                "Unsupported realtime audio input sample rate "
                f"'{format_rate}'. Only '{self.REALTIME_AUDIO_SAMPLE_RATE}' is supported."
            )

    def _apply_audio_output_update(self, session: RealtimeSessionState, update) -> None:
        audio_config = getattr(update, "audio", None)
        if audio_config is None:
            return

        output_config = (
            audio_config.get("output")
            if isinstance(audio_config, dict)
            else getattr(audio_config, "output", None)
        )
        if output_config is None:
            return

        output_data = (
            output_config if isinstance(output_config, dict) else output_config.model_dump(exclude_none=True)
        )

        format_type = self._extract_format_type(output_data.get("format"))
        voice = output_data.get("voice")
        speed = output_data.get("speed")

        if format_type is not None:
            self._ensure_audio_output_supported(format_type)

        session.update_audio_output_config(
            format_type=format_type,
            voice=voice,
            speed=speed,
        )

    def _resolve_response_config(self, session: RealtimeSessionState, response=None) -> EffectiveResponseConfig:
        modalities = session.get_output_modalities()
        session_audio_cfg = session.get_audio_output_config()
        audio_format_type = session_audio_cfg["format_type"]
        audio_voice = session_audio_cfg["voice"]
        audio_speed = session_audio_cfg["speed"]

        if response is not None:
            response_modalities = (
                response.get("output_modalities")
                if isinstance(response, dict)
                else getattr(response, "output_modalities", None)
            )
            if response_modalities is not None:
                modalities = self._normalize_output_modalities(list(response_modalities))

            response_audio = (
                response.get("audio")
                if isinstance(response, dict)
                else getattr(response, "audio", None)
            )
            if response_audio is not None:
                output_cfg = (
                    response_audio.get("output")
                    if isinstance(response_audio, dict)
                    else getattr(response_audio, "output", None)
                )
                if output_cfg is not None:
                    output_data = (
                        output_cfg if isinstance(output_cfg, dict) else output_cfg.model_dump(exclude_none=True)
                    )
                    format_type = self._extract_format_type(output_data.get("format"))
                    if format_type:
                        audio_format_type = format_type
                    if output_data.get("voice"):
                        audio_voice = output_data["voice"]

        return EffectiveResponseConfig(
            modalities=modalities,
            audio_format_type=audio_format_type,
            audio_voice=audio_voice,
            audio_speed=audio_speed,
        )

    def _split_tools(
        self,
        raw_tools: Iterable[RealtimeFunctionTool | Mcp],
    ) -> Tuple[List[RealtimeFunctionTool], List[McpServerConfig]]:
        function_tools: List[RealtimeFunctionTool] = []
        mcp_configs: List[McpServerConfig] = []

        for tool in raw_tools:
            if isinstance(tool, RealtimeFunctionTool):
                function_tools.append(tool)
                continue

            if isinstance(tool, Mcp):
                mcp_configs.append(McpServerConfig.from_dict(tool.model_dump(exclude_none=True)))
                continue

            payload = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
            if isinstance(payload, dict) and payload.get("type") == "mcp":
                mcp_configs.append(McpServerConfig.from_dict(payload))
            elif isinstance(payload, dict):
                function_tools.append(RealtimeFunctionTool(**payload))

        return function_tools, mcp_configs

    async def _sync_mcp_servers(
        self,
        session: RealtimeSessionState,
        configs: Sequence[McpServerConfig],
    ) -> None:
        target_labels = {config.server_label for config in configs}
        current_labels = set(session.mcp_registry.server_labels)

        for stale_label in current_labels - target_labels:
            await session.mcp_registry.unregister_server(stale_label)

        for config in configs:
            try:
                await self._register_mcp_server(session, config)
            except Exception as exc:
                logger.error(
                    "Failed to register MCP server %s: %s",
                    config.server_label,
                    exc,
                )
                await session.writer.send_error(
                    message=f"Failed to connect MCP server '{config.server_label}': {exc}",
                    error_type="server_error",
                    code="mcp_connection_error",
                )

    async def _register_mcp_server(
        self,
        session: RealtimeSessionState,
        config: McpServerConfig,
    ) -> None:
        async with McpListToolsContext(session, config.server_label) as ctx:
            tools = await session.mcp_registry.register_server(config)

            tools_data = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "annotations": tool.annotations,
                }
                for tool in tools
            ]
            ctx.set_tools(tools_data)

    def _session_payload(
        self,
        *,
        session: RealtimeSessionState,
        model: str,
    ) -> dict:
        input_cfg = session.get_audio_input_config()
        audio_cfg = session.get_audio_output_config()
        input_format = {"type": input_cfg["format_type"]}
        if input_cfg["format_type"] == self.REALTIME_PCM_FORMAT:
            input_format["rate"] = input_cfg["sample_rate"]

        output_format = {"type": audio_cfg["format_type"]}
        if audio_cfg["format_type"] == self.REALTIME_PCM_FORMAT:
            output_format["rate"] = self.REALTIME_AUDIO_SAMPLE_RATE

        return {
            "id": session.session_id,
            "type": "realtime",
            "model": model,
            "output_modalities": session.get_output_modalities(),
            "audio": {
                "input": {
                    "format": input_format,
                },
                "output": {
                    "format": output_format,
                    "voice": audio_cfg["voice"],
                    "speed": audio_cfg["speed"],
                }
            },
            "tools": [tool.model_dump(exclude_none=True) for tool in session.get_all_tools()],
        }
