from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.context_providers import AgentProfileContextProvider, ArtifactContextProvider, ConversationHistoryProvider, ProjectMemoryContextProvider, TaskWhiteboardContextProvider
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.event_store import EventStore
from ergon_studio.memory_store import MemoryStore
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.whiteboard_store import WhiteboardStore


def build_agent(
    registry: RuntimeRegistry,
    agent_id: str,
    *,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
    conversation_store: ConversationStore | None = None,
    memory_store: MemoryStore | None = None,
    artifact_store: ArtifactStore | None = None,
    whiteboard_store: WhiteboardStore | None = None,
    event_store: EventStore | None = None,
) -> Agent[Any]:
    definition = registry.agent_definitions[agent_id]
    role = str(definition.metadata.get("role", definition.id))
    provider_name = _resolve_provider_name(registry.config, role, definition.id)
    provider_config = registry.config["providers"][provider_name]
    client = _build_client(provider_config)
    tools = _resolve_tools(definition, tool_registry or {})
    context_providers = _build_context_providers(
        definition=definition,
        conversation_store=conversation_store,
        memory_store=memory_store,
        artifact_store=artifact_store,
        whiteboard_store=whiteboard_store,
        event_store=event_store,
    )

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


def _build_context_providers(
    *,
    definition: DefinitionDocument,
    conversation_store: ConversationStore | None,
    memory_store: MemoryStore | None,
    artifact_store: ArtifactStore | None,
    whiteboard_store: WhiteboardStore | None,
    event_store: EventStore | None,
) -> list[object]:
    providers: list[object] = [AgentProfileContextProvider(definition)]
    if conversation_store is not None and event_store is not None:
        providers.append(ConversationHistoryProvider(conversation_store, event_store))
    if whiteboard_store is not None and event_store is not None:
        providers.append(TaskWhiteboardContextProvider(whiteboard_store, event_store))
    if memory_store is not None and event_store is not None:
        providers.append(ProjectMemoryContextProvider(memory_store, event_store))
    if artifact_store is not None and event_store is not None:
        providers.append(ArtifactContextProvider(artifact_store, event_store))
    return providers
