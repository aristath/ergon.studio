from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyInputMessage, ProxyToolCall, ProxyTurnRequest


def parse_chat_completion_request(payload: dict[str, Any]) -> ProxyTurnRequest:
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("chat completion request must include a non-empty model")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("chat completion request must include a messages list")

    messages = tuple(_parse_message(item) for item in raw_messages)
    tools = tuple(_parse_tool(item) for item in payload.get("tools", []) or [])
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
        parsed_tool_calls = tuple(_parse_tool_call(item) for item in tool_calls)

    return ProxyInputMessage(
        role=role.strip(),
        content=_normalize_content(payload.get("content")),
        name=_optional_non_empty_text(payload.get("name")),
        tool_call_id=_optional_non_empty_text(payload.get("tool_call_id")),
        tool_calls=parsed_tool_calls,
    )


def _parse_tool(payload: Any) -> ProxyFunctionTool:
    if not isinstance(payload, dict):
        raise ValueError("tools must contain objects")
    if payload.get("type") != "function":
        raise ValueError("only function tools are supported")
    function = payload.get("function")
    if not isinstance(function, dict):
        raise ValueError("function tools must include a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("function tools must include a non-empty name")
    description = function.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ValueError("function tool description must be a string")
    parameters = function.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("function tool parameters must be an object")
    strict = function.get("strict", payload.get("strict", False))
    if type(strict) is not bool:
        raise ValueError("function tool strict must be a bool")
    return ProxyFunctionTool(
        name=name.strip(),
        description=description,
        parameters=parameters,
        strict=strict,
    )


def _parse_tool_call(payload: Any) -> ProxyToolCall:
    if not isinstance(payload, dict):
        raise ValueError("tool_calls must contain objects")
    if payload.get("type") != "function":
        raise ValueError("only function tool calls are supported")
    call_id = payload.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("tool calls must include a non-empty id")
    function = payload.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool calls must include a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool calls must include a non-empty function name")
    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        raise ValueError("tool call arguments must be a string")
    return ProxyToolCall(
        id=call_id.strip(),
        name=name.strip(),
        arguments_json=arguments,
    )


def _normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            part_type = item.get("type")
            if part_type in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise ValueError("message content must be a string, content-part list, or null")


def _optional_non_empty_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text fields must be strings when provided")
    stripped = value.strip()
    if not stripped:
        raise ValueError("optional text fields must be non-empty when provided")
    return stripped
