from __future__ import annotations

from typing import Any
from uuid import uuid4

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyOutputItemRef, ProxyReasoningDeltaEvent, ProxyToolCallEvent


def build_responses_response(
    *,
    response_id: str,
    model: str,
    created_at: int,
    content: str,
    reasoning: str = "",
    tool_calls: tuple[Any, ...] = (),
    output_items: tuple[ProxyOutputItemRef, ...] = (),
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    ordered_items = _normalize_output_items(output_items, reasoning=reasoning, content=content, tool_calls=tool_calls)
    tool_calls_by_id = {tool_call.id: tool_call for tool_call in tool_calls}
    tool_calls_emitted: set[str] = set()
    for item in ordered_items:
        if item.kind == "reasoning":
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
        elif item.kind == "tool_call":
            tool_call = tool_calls_by_id.get(item.call_id or "")
            if tool_call is None or tool_call.id in tool_calls_emitted:
                continue
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
            tool_calls_emitted.add(tool_call.id)
        elif item.kind == "content":
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
    tool_output_index: int = 0,
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
                "output_index": tool_output_index,
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


def _normalize_output_items(
    output_items: tuple[ProxyOutputItemRef, ...],
    *,
    reasoning: str,
    content: str,
    tool_calls: tuple[Any, ...],
) -> tuple[ProxyOutputItemRef, ...]:
    ordered: list[ProxyOutputItemRef] = []
    for item in output_items:
        if item in ordered:
            continue
        if item.kind == "reasoning" and not reasoning:
            continue
        if item.kind == "content" and not (content or not tool_calls):
            continue
        if item.kind == "tool_call" and not any(tool_call.id == item.call_id for tool_call in tool_calls):
            continue
        ordered.append(item)
    if reasoning and ProxyOutputItemRef(kind="reasoning") not in ordered:
        ordered.append(ProxyOutputItemRef(kind="reasoning"))
    for tool_call in tool_calls:
        ref = ProxyOutputItemRef(kind="tool_call", call_id=tool_call.id)
        if ref not in ordered:
            ordered.append(ref)
    content_ref = ProxyOutputItemRef(kind="content")
    if (content or not tool_calls) and content_ref not in ordered:
        ordered.append(content_ref)
    return tuple(ordered)
