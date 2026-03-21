from __future__ import annotations

from typing import Any

from ergon_studio.proxy.models import ProxyFunctionTool


def validate_tool_choice(
    tool_choice: str | dict[str, Any] | None,
    *,
    tools: tuple[ProxyFunctionTool, ...],
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "required", "none"}:
            raise ValueError(
                "tool_choice must be auto, required, none, or a function selector"
            )
        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' requires at least one tool")
        return tool_choice
    if not isinstance(tool_choice, dict):
        raise ValueError("tool_choice must be a string, object, or null")
    if tool_choice.get("type") != "function":
        raise ValueError("tool_choice objects must use type='function'")
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise ValueError(
            "tool_choice function selectors must include a function object"
        )
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "tool_choice function selectors must include a non-empty function name"
        )
    stripped_name = name.strip()
    if stripped_name not in {tool.name for tool in tools}:
        raise ValueError(f"tool_choice requested unknown tool: {stripped_name}")
    return {
        "type": "function",
        "function": {"name": stripped_name},
    }


def resolve_agent_tool_policy(
    *,
    tools: tuple[ProxyFunctionTool, ...],
    tool_choice: str | dict[str, Any] | None,
    parallel_tool_calls: bool | None,
) -> tuple[tuple[ProxyFunctionTool, ...], dict[str, Any]]:
    tool_choice = validate_tool_choice(tool_choice, tools=tools)
    resolved_tools = tools
    run_options: dict[str, Any] = {}
    if tool_choice == "none":
        resolved_tools = ()
        run_options["tool_choice"] = "none"
    elif tool_choice == "required":
        run_options["tool_choice"] = "required"
    elif tool_choice == "auto":
        run_options["tool_choice"] = "auto"
    elif isinstance(tool_choice, dict):
        name = tool_choice["function"]["name"]
        resolved_tools = tuple(tool for tool in tools if tool.name == name)
        run_options["tool_choice"] = {
            "mode": "required",
            "required_function_name": name,
        }
    if parallel_tool_calls is not None:
        run_options["allow_multiple_tool_calls"] = parallel_tool_calls
    return resolved_tools, run_options
