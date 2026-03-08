# Nexus

OpenAI-compatible ASR/Realtime/Chat/TTS server with a clean layered architecture.

## Architecture

- `src/nexus/api`: FastAPI HTTP/WebSocket entrypoints.
- `src/nexus/application`: use cases, orchestration, protocol parsing/writing, DI container.
- `src/nexus/domain`: session/domain state.
- `src/nexus/infrastructure`: adapters for OpenAI/gRPC/MCP clients.

## Realtime refactor highlights

- Inbound WebSocket events are validated with `TypeAdapter(RealtimeClientEvent)`.
- Outbound server events are validated with `TypeAdapter(RealtimeServerEvent)` before send.
- Event dispatch is registry-based (`application.realtime.dispatch`), replacing `if/elif` chains.
- Realtime worker logic is split into orchestrators (`transcription_worker`, `response_orchestrator`, `tool_call_orchestrator`).
- MCP failure paths now emit `mcp_list_tools.failed` and `response.mcp_call.failed`.
- Realtime audio contract is strict `audio/pcm` at `24000Hz` for both input and output.
- ASR path performs streaming resampling `24kHz -> 16kHz` before gRPC inference.

## Testing

Default automated suite:

```bash
poetry run pytest -q
```

Manual/E2E scripts live under `tests/e2e` and are excluded from default pytest runs.
