from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyInputMessage, ProxyToolCall, ProxyTurnRequest
from ergon_studio.proxy.parse_utils import normalize_message_content, optional_non_empty_text, parse_function_tool, parse_function_tool_call
from ergon_studio.proxy.tool_policy import validate_tool_choice


def parse_chat_completion_request(payload: dict[str, Any]) -> ProxyTurnRequest:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("chat completion request must include a non-empty model")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("chat completion request must include a messages list")

    messages = tuple(_parse_messages(raw_messages))
    tools = tuple(parse_function_tool(item) for item in payload.get("tools", []) or [])
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a bool")

    tool_choice = payload.get("tool_choice")
    if tool_choice is not None and not isinstance(tool_choice, (str, dict)):
        raise ValueError("tool_choice must be a string, object, or null")
    tool_choice = validate_tool_choice(tool_choice, tools=tools)

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


def _parse_messages(raw_messages: list[Any]) -> list[ProxyInputMessage]:
    messages: list[ProxyInputMessage] = []
    pending_legacy_call_ids: list[str] = []
    for index, payload in enumerate(raw_messages):
        message = _parse_message(payload, index=index, pending_legacy_call_ids=pending_legacy_call_ids)
        messages.append(message)
        if message.role == "assistant" and message.tool_calls:
            pending_legacy_call_ids.extend(tool_call.id for tool_call in message.tool_calls)
        elif message.role == "tool" and pending_legacy_call_ids:
            pending_legacy_call_ids.pop(0)
    return messages


def _parse_message(payload: Any, *, index: int, pending_legacy_call_ids: list[str]) -> ProxyInputMessage:
    if not isinstance(payload, dict):
        raise ValueError("messages must contain objects")
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("message role must be a non-empty string")
    normalized_role = _normalize_message_role(role)

    tool_calls = payload.get("tool_calls")
    parsed_tool_calls: tuple[ProxyToolCall, ...] = ()
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        parsed_tool_calls = tuple(parse_function_tool_call(item) for item in tool_calls)
    elif normalized_role == "assistant":
        legacy_function_call = payload.get("function_call")
        if legacy_function_call is not None:
            parsed_tool_calls = (_parse_legacy_function_call(legacy_function_call, index=index),)

    tool_call_id = optional_non_empty_text(payload.get("tool_call_id"))
    if normalized_role == "function":
        normalized_role = "tool"
        tool_call_id = pending_legacy_call_ids[0] if pending_legacy_call_ids else f"legacy_call_{index}"

    return ProxyInputMessage(
        role=normalized_role,
        content=normalize_message_content(payload.get("content")),
        name=optional_non_empty_text(payload.get("name")),
        tool_call_id=tool_call_id,
        tool_calls=parsed_tool_calls,
    )


def _normalize_message_role(role: str) -> str:
    stripped = role.strip()
    if stripped.casefold() == "developer":
        return "system"
    return stripped


def _parse_legacy_function_call(payload: Any, *, index: int) -> ProxyToolCall:
    if not isinstance(payload, dict):
        raise ValueError("assistant function_call must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("assistant function_call must include a non-empty name")
    arguments = payload.get("arguments", "")
    if not isinstance(arguments, str):
        raise ValueError("assistant function_call arguments must be a string")
    return ProxyToolCall(
        id=f"legacy_call_{index}",
        name=name.strip(),
        arguments_json=arguments,
    )
