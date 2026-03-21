from __future__ import annotations

import json
from typing import Any

from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
)


def encode_chat_stream_event(
    event: ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent,
    *,
    completion_id: str,
    model: str,
    created_at: int,
    choice_index: int = 0,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": choice_index,
                "delta": {},
                "finish_reason": None,
            }
        ],
    }
    choice = base["choices"][0]
    delta = choice["delta"]

    if isinstance(event, ProxyReasoningDeltaEvent):
        delta["reasoning_content"] = event.delta
        delta["reasoning"] = event.delta
        return base
    if isinstance(event, ProxyContentDeltaEvent):
        delta["content"] = event.delta
        return base
    if isinstance(event, ProxyToolCallEvent):
        delta["tool_calls"] = [
            {
                "index": event.index,
                "id": event.call.id,
                "type": "function",
                "function": {
                    "name": event.call.name,
                    "arguments": event.call.arguments_json,
                },
            }
        ]
        return base
    if isinstance(event, ProxyFinishEvent):
        choice["delta"] = {}
        choice["finish_reason"] = event.reason
        return base
    raise TypeError(f"unsupported event type: {type(event).__name__}")


def encode_chat_stream_sse(
    event: ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent,
    *,
    completion_id: str,
    model: str,
    created_at: int,
    choice_index: int = 0,
) -> bytes:
    payload = encode_chat_stream_event(
        event,
        completion_id=completion_id,
        model=model,
        created_at=created_at,
        choice_index=choice_index,
    )
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def encode_chat_stream_done() -> bytes:
    return b"data: [DONE]\n\n"


def build_chat_completion_response(
    *,
    completion_id: str,
    model: str,
    created_at: int,
    content: str,
    finish_reason: str,
    reasoning: str = "",
    tool_calls: tuple[Any, ...] = (),
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if reasoning:
        message["reasoning_content"] = reasoning
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_json,
                },
            }
            for tool_call in tool_calls
        ]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
