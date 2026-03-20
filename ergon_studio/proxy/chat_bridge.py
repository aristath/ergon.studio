from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyInputMessage, ProxyToolCall, ProxyTurnRequest
from ergon_studio.proxy.parse_utils import normalize_message_content, optional_non_empty_text, parse_function_tool, parse_function_tool_call


def parse_chat_completion_request(payload: dict[str, Any]) -> ProxyTurnRequest:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("chat completion request must include a non-empty model")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("chat completion request must include a messages list")

    messages = tuple(_parse_message(item) for item in raw_messages)
    tools = tuple(parse_function_tool(item) for item in payload.get("tools", []) or [])
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a bool")

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        raise ValueError("tool_choice must be a string, object, or null")

    parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None and type(parallel_tool_calls) is not bool:
        raise ValueError("parallel_tool_calls must be a bool or null")

    return ProxyTurnRequest(
        model=model.strip(),
        messages=messages,
        tools=tools,
        stream=stream,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )


def _parse_message(payload: Any) -> ProxyInputMessage:
    if not isinstance(payload, dict):
        raise ValueError("messages must contain objects")
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("message role must be a non-empty string")

    tool_calls = payload.get("tool_calls")
    parsed_tool_calls: tuple[ProxyToolCall, ...] = ()
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        parsed_tool_calls = tuple(parse_function_tool_call(item) for item in tool_calls)

    return ProxyInputMessage(
        role=role.strip(),
        content=normalize_message_content(payload.get("content")),
        name=optional_non_empty_text(payload.get("name")),
        tool_call_id=optional_non_empty_text(payload.get("tool_call_id")),
        tool_calls=parsed_tool_calls,
    )
