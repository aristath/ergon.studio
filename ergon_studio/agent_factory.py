from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.registry import RuntimeRegistry


def build_agent(
    registry: RuntimeRegistry,
    agent_id: str,
    *,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
) -> Agent[Any]:
    definition = registry.agent_definitions[agent_id]
    role = str(definition.metadata.get("role", definition.id))
    provider_name = _resolve_provider_name(registry.config, role, definition.id)
    provider_config = registry.config["providers"][provider_name]
    client = _build_client(provider_config)
    tools = _resolve_tools(definition, tool_registry or {})

    default_options: dict[str, Any] = {}
    for key in ("temperature", "max_tokens"):
        if key in definition.metadata:
            default_options[key] = definition.metadata[key]

    return Agent(
        client=client,
        id=definition.id,
        name=str(definition.metadata.get("name", definition.id)),
        description=role,
        instructions=compose_instructions(definition),
        tools=tools or None,
        default_options=default_options or None,
    )


def compose_instructions(definition: DefinitionDocument) -> str:
    if not definition.sections:
        return definition.body

    parts: list[str] = []
    for title, content in definition.sections.items():
        parts.append(f"## {title}")
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip()


def _resolve_provider_name(config: dict[str, Any], role: str, agent_id: str) -> str:
    role_assignments = config.get("role_assignments", {})
    provider_name = role_assignments.get(role) or role_assignments.get(agent_id)
    if not provider_name:
        raise ValueError(f"no provider assigned for role '{role}'")
    if provider_name not in config.get("providers", {}):
        raise ValueError(f"provider '{provider_name}' is not defined")
    return provider_name


def _build_client(provider_config: dict[str, Any]) -> OpenAIChatClient:
    provider_type = provider_config.get("type", "openai_chat")
    if provider_type != "openai_chat":
        raise ValueError(f"unsupported provider type: {provider_type}")

    model_id = provider_config.get("model")
    if not model_id:
        raise ValueError("provider config must define a model")

    return OpenAIChatClient(
        model_id=model_id,
        api_key=provider_config.get("api_key"),
        base_url=provider_config.get("base_url"),
    )


def _resolve_tools(
    definition: DefinitionDocument,
    tool_registry: Mapping[str, Callable[..., Any]],
) -> list[Callable[..., Any]]:
    tool_names = definition.metadata.get("tools", [])
    if not tool_names:
        return []
    if not isinstance(tool_names, list):
        raise ValueError(f"tools for '{definition.id}' must be a list")

    resolved: list[Callable[..., Any]] = []
    for tool_name in tool_names:
        if tool_name not in tool_registry:
            raise ValueError(f"unknown tool: {tool_name}")
        resolved.append(tool_registry[tool_name])
    return resolved
