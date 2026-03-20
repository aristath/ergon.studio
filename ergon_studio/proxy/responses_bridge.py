from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.proxy.parse_utils import normalize_message_content, optional_non_empty_text, parse_function_tool, parse_function_tool_call
from ergon_studio.proxy.tool_policy import validate_tool_choice


def parse_responses_request(payload: dict[str, Any]) -> ProxyTurnRequest:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("responses request must include a non-empty model")

    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a bool")

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        raise ValueError("tool_choice must be a string, object, or null")
    parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None and type(parallel_tool_calls) is not bool:
        raise ValueError("parallel_tool_calls must be a bool or null")

    tools = tuple(parse_function_tool(item) for item in payload.get("tools", []) or [])
    tool_choice = validate_tool_choice(tool_choice, tools=tools)
    messages = tuple(_parse_input_item(item) for item in _normalize_input(payload.get("input")))
    return ProxyTurnRequest(
        model=model.strip(),
        messages=messages,
        tools=tools,
        stream=stream,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _normalize_input(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [{"type": "message", "role": "user", "content": value}]
    if isinstance(value, list):
        return value
    raise ValueError("responses input must be a string or an array")


def _parse_input_item(payload: Any) -> ProxyInputMessage:
    if not isinstance(payload, dict):
        raise ValueError("responses input items must be objects")
    item_type = payload.get("type", "message")
    if item_type == "function_call":
        return ProxyInputMessage(
            role="assistant",
            content="",
            tool_calls=(
                parse_function_tool_call(
                    {
                        "id": payload.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": payload.get("name"),
                            "arguments": payload.get("arguments", ""),
                        },
                    }
                ),
            ),
        )
    if item_type == "function_call_output":
        return ProxyInputMessage(
            role="tool",
            content=normalize_message_content(payload.get("output")),
            tool_call_id=optional_non_empty_text(payload.get("call_id") or payload.get("tool_call_id")),
        )
    if item_type != "message":
        raise ValueError(f"unsupported responses input item type: {item_type}")
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("responses message items must include a non-empty role")
    return ProxyInputMessage(
        role=role.strip(),
        content=normalize_message_content(payload.get("content")),
        name=optional_non_empty_text(payload.get("name")),
    )
