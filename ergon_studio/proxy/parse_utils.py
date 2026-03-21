from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall


def parse_function_tool(payload: Any) -> ProxyFunctionTool:
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


def parse_function_tool_call(payload: Any) -> ProxyToolCall:
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


def normalize_message_content(value: Any) -> str:
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


def optional_non_empty_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text fields must be strings when provided")
    stripped = value.strip()
    if not stripped:
        raise ValueError("optional text fields must be non-empty when provided")
    return stripped
