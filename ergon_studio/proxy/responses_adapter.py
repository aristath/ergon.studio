from __future__ import annotations

from typing import Any
from uuid import uuid4

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyReasoningDeltaEvent, ProxyToolCallEvent


def build_responses_response(
    *,
    response_id: str,
    model: str,
    created_at: int,
    content: str,
    reasoning: str = "",
    tool_calls: tuple[Any, ...] = (),
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if reasoning:
        output.append(
            {
                "id": f"rs_{uuid4().hex}",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": reasoning,
                    }
                ],
                "content": [],
            }
        )
    if tool_calls:
        for tool_call in tool_calls:
            output.append(
                {
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_json,
                    "status": "completed",
                }
            )
    if content or not tool_calls:
        output.append(
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": content,
    }


def encode_responses_stream_events(
    *,
    event: ProxyReasoningDeltaEvent | ProxyContentDeltaEvent | ProxyToolCallEvent | ProxyFinishEvent,
    response_id: str,
    model: str,
    created_at: int,
    sequence_number: int,
    reasoning_item_id: str,
    message_item_id: str,
    reasoning_output_index: int = 0,
    message_output_index: int = 0,
    tool_output_offset: int = 0,
    include_output_done: bool = True,
) -> list[dict[str, Any]]:
    if isinstance(event, ProxyReasoningDeltaEvent):
        return [
            {
                "type": "response.reasoning_text.delta",
                "event_id": f"event_{uuid4().hex}",
                "response_id": response_id,
                "item_id": reasoning_item_id,
                "output_index": reasoning_output_index,
                "content_index": 0,
                "delta": event.delta,
                "sequence_number": sequence_number,
            }
        ]
    if isinstance(event, ProxyContentDeltaEvent):
        return [
            {
                "type": "response.output_text.delta",
                "event_id": f"event_{uuid4().hex}",
                "response_id": response_id,
                "item_id": message_item_id,
                "output_index": message_output_index,
                "content_index": 0,
                "delta": event.delta,
                "sequence_number": sequence_number,
            }
        ]
    if isinstance(event, ProxyToolCallEvent):
        return [
            {
                "type": "response.output_item.added",
                "event_id": f"event_{uuid4().hex}",
                "response_id": response_id,
                "output_index": tool_output_offset + event.index,
                "item": {
                    "id": f"fc_{uuid4().hex}",
                    "type": "function_call",
                    "call_id": event.call.id,
                    "name": event.call.name,
                    "arguments": event.call.arguments_json,
                    "status": "completed",
                },
                "sequence_number": sequence_number,
            }
        ]
    if isinstance(event, ProxyFinishEvent):
        payloads: list[dict[str, Any]] = []
        if include_output_done:
            payloads.append(
                {
                    "type": "response.output_text.done",
                    "event_id": f"event_{uuid4().hex}",
                    "response_id": response_id,
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": "",
                    "sequence_number": sequence_number,
                }
            )
            sequence_number += 1
        payloads.append(
            {
                "type": "response.completed",
                "event_id": f"event_{uuid4().hex}",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": model,
                    "status": "completed",
                },
                "sequence_number": sequence_number,
            }
        )
        return payloads
    raise TypeError(f"unsupported event type: {type(event).__name__}")


def encode_responses_stream_sse(payload: dict[str, Any]) -> bytes:
    import json

    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
