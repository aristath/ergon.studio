from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from ergon_studio.context_providers import AgentProfileContextProvider
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.registry import RuntimeRegistry


def build_agent(
    registry: RuntimeRegistry,
    agent_id: str,
    *,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
    model_id_override: str | None = None,
) -> Agent[Any]:
    definition = registry.agent_definitions[agent_id]
    role = str(definition.metadata.get("role", definition.id))
    client = _build_client(registry, model_id_override=model_id_override)
    tools = _resolve_tools(definition, tool_registry)
    context_providers = _build_context_providers(registry=registry, definition=definition)

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
        context_providers=context_providers or None,
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


def provider_supports_tool_calling(registry: RuntimeRegistry) -> bool:
    return registry.upstream.tool_calling


def _build_client(
    registry: RuntimeRegistry,
    *,
    model_id_override: str | None = None,
) -> OpenAIChatClient:
    model_id = model_id_override
    if not model_id:
        raise ValueError("proxy requests must supply a model")

    api_key = registry.upstream.api_key
    base_url = registry.upstream.base_url
    if not api_key and base_url:
        api_key = "not-needed"

    return OpenAIChatClient(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        instruction_role=registry.upstream.instruction_role,
    )


def _resolve_tools(
    definition: DefinitionDocument,
    tool_registry: Mapping[str, Callable[..., Any]] | None,
) -> list[object]:
    if tool_registry is None:
        return []
    tool_names = definition.metadata.get("tools", [])
    if tool_names and not isinstance(tool_names, list):
        raise ValueError(f"tools for '{definition.id}' must be a list")

    resolved: list[Callable[..., Any] | object] = []
    for tool_name in tool_names or []:
        if tool_name not in tool_registry:
            raise ValueError(f"unknown tool: {tool_name}")
        resolved.append(tool_registry[tool_name])
    return resolved


def _build_context_providers(
    *,
    registry: RuntimeRegistry,
    definition: DefinitionDocument,
) -> list[object]:
    return [AgentProfileContextProvider(definition, registry=registry)]
