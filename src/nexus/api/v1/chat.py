"""
FastAPI 路由 - Chat Completions API
兼容 OpenAI Chat API 格式
"""

import json
import logging
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Annotated, Optional, Union, List

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from openai.types.completion_usage import CompletionUsage

from nexus.inferencers.chat.inferencer import Inferencer

from .depends import get_chat_inferencer as get_inferencer

router = APIRouter(prefix="/chat", tags=["Chat"])

logger = logging.getLogger(__name__)


def _generate_sse_stream(
    inferencer: Inferencer,
    messages: list[dict],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """
    生成 SSE 流式响应（兼容 OpenAI 流式格式）
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    for chunk in inferencer.chat_stream(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        event_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    end_data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(end_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def build_response(response_text: str, model: str) -> ChatCompletion:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    return ChatCompletion(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=0, total_tokens=0, completion_tokens=0),
        object="chat.completion",
    )


@router.post("/completions")
async def create_chat_completion(
    inferencer: Annotated[Inferencer, Depends(get_inferencer)],
    messages: Annotated[List[ChatCompletionMessageParam], Body(..., embed=True)],
    model: Annotated[str, Body(..., embed=True)],
    audio: Annotated[Optional[ChatCompletionAudioParam], Body(embed=True)] = None,
    tools: Annotated[List[ChatCompletionToolUnionParam], Body(embed=True)] = [],
    frequency_penalty: Annotated[Optional[float], Body(embed=True)] = None,
    temperature: Annotated[Optional[float], Body(embed=True)] = None,
    max_tokens: Annotated[Optional[int], Body(embed=True)] = None,
    stream: Annotated[bool, Body(embed=True)] = False,
):
    """
    创建 Chat Completion

    兼容 OpenAI Chat API 格式
    支持 stream=True 参数返回 SSE 流式响应
    """
    logger.info(f"Request received: model={model}, len(messages)={len(messages)}")
    if stream:
        # 流式响应
        return StreamingResponse(
            _generate_sse_stream(
                inferencer=inferencer,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # 非流式响应
    try:
        response = inferencer.chat(
            messages=messages,
            model=model,
            audio=audio,
            tools=tools,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
