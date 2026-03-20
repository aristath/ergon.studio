from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_framework import FunctionTool

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall


def build_declaration_tools(tools: Sequence[ProxyFunctionTool]) -> list[FunctionTool]:
    declarations: list[FunctionTool] = []
    for tool in tools:
        declarations.append(
            FunctionTool(
                name=tool.name,
                description=tool.description,
                func=None,
                input_model=tool.parameters,
            )
        )
    return declarations


def extract_tool_calls(response: Any) -> tuple[ProxyToolCall, ...]:
    messages = getattr(response, "messages", None)
    if not isinstance(messages, list):
        return ()
    tool_calls: list[ProxyToolCall] = []
    for message in messages:
        contents = getattr(message, "contents", None)
        if not isinstance(contents, list):
            continue
        for content in contents:
            if getattr(content, "type", None) != "function_call":
                continue
            call_id = getattr(content, "call_id", "") or ""
            name = getattr(content, "name", "") or ""
            arguments = getattr(content, "arguments", "") or ""
            if not isinstance(arguments, str):
                arguments = str(arguments)
            if not call_id or not name:
                continue
            tool_calls.append(
                ProxyToolCall(
                    id=call_id,
                    name=name,
                    arguments_json=arguments,
                )
            )
    return tuple(tool_calls)
