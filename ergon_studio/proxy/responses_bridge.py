from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.proxy.parse_utils import (
    normalize_message_content,
    optional_non_empty_text,
    parse_function_tool,
    parse_function_tool_call,
)
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

    raw_tools = payload.get("tools")
    if raw_tools is None:
        raw_tools = []
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be a list or null")
    tools = tuple(parse_function_tool(item) for item in raw_tools)
    tool_choice = validate_tool_choice(tool_choice, tools=tools)
    messages: list[ProxyInputMessage] = []
    instructions = (
        optional_non_empty_text(payload.get("instructions"))
        if payload.get("instructions") is not None
        else None
    )
    if instructions is not None:
        messages.append(
            ProxyInputMessage(
                role="system",
                content=instructions,
            )
        )

    raw_input = payload.get("input")
    if raw_input is None:
        if not messages:
            raise ValueError("responses input must be a string or an array")
    else:
        messages.extend(_parse_input_item(item) for item in _normalize_input(raw_input))
    return ProxyTurnRequest(
        model=model.strip(),
        messages=tuple(messages),
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
            tool_call_id=_parse_function_call_output_id(payload),
        )
    if item_type != "message":
        raise ValueError(f"unsupported responses input item type: {item_type}")
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("responses message items must include a non-empty role")
    return ProxyInputMessage(
        role=_normalize_message_role(role),
        content=normalize_message_content(payload.get("content")),
        name=optional_non_empty_text(payload.get("name")),
    )


def _normalize_message_role(role: str) -> str:
    stripped = role.strip()
    if stripped.casefold() == "developer":
        return "system"
    return stripped


def _parse_function_call_output_id(payload: dict[str, Any]) -> str:
    for field_name in ("call_id", "tool_call_id"):
        raw_value = payload.get(field_name)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            raise ValueError(
                f"responses function_call_output {field_name} must be a string"
            )
        stripped = raw_value.strip()
        if stripped:
            return stripped
    if "call_id" in payload or "tool_call_id" in payload:
        raise ValueError(
            "responses function_call_output call_id/tool_call_id must be non-empty"
        )
    raise ValueError(
        "responses function_call_output items must include call_id or tool_call_id"
    )
