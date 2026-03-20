from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from typing import Any

from agent_framework import Agent, MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.context_providers import AgentProfileContextProvider, ArtifactContextProvider, ConversationHistoryProvider, ProjectMemoryContextProvider, RetrievalContextProvider, TaskWhiteboardContextProvider
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.event_store import EventStore
from ergon_studio.memory_store import MemoryStore
from ergon_studio.retrieval import RetrievalIndex
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.tool_call_logging import build_tool_call_middleware
from ergon_studio.tool_call_store import ToolCallStore
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
    tool_call_store: ToolCallStore | None = None,
    retrieval_index: RetrievalIndex | None = None,
    ignore_missing_tools: bool = False,
    include_mcp_servers: bool = True,
) -> Agent[Any]:
    definition = registry.agent_definitions[agent_id]
    role = str(definition.metadata.get("role", definition.id))
    provider_name = _resolve_provider_name(registry.config, role, definition.id)
    provider_config = registry.config["providers"][provider_name]
    provider_capabilities = provider_config.get("capabilities", {})
    if not isinstance(provider_capabilities, dict):
        provider_capabilities = {}
    client = _build_client(
        provider_config,
        event_store=event_store,
        tool_call_store=tool_call_store,
    )
    tool_calling_enabled = provider_capabilities.get("tool_calling", True) is not False
    tools = (
        _resolve_tools(
            definition,
            tool_registry or {},
            ignore_missing_tools=ignore_missing_tools,
            include_mcp_servers=include_mcp_servers,
        )
        if tool_calling_enabled
        else []
    )
    context_providers = _build_context_providers(
        registry=registry,
        definition=definition,
        provider_name=provider_name,
        provider_capabilities=provider_capabilities,
        conversation_store=conversation_store,
        memory_store=memory_store,
        artifact_store=artifact_store,
        whiteboard_store=whiteboard_store,
        event_store=event_store,
        retrieval_index=retrieval_index,
    )

    default_options: dict[str, Any] = {}
    for key in ("temperature", "max_tokens"):
        if key in definition.metadata:
            default_options[key] = definition.metadata[key]
        elif key in provider_config:
            default_options[key] = provider_config[key]

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


def _build_client(
    provider_config: dict[str, Any],
    *,
    event_store: EventStore | None,
    tool_call_store: ToolCallStore | None,
) -> OpenAIChatClient:
    provider_type = provider_config.get("type", "openai_chat")
    if provider_type != "openai_chat":
        raise ValueError(f"unsupported provider type: {provider_type}")

    model_id = provider_config.get("model")
    if not model_id:
        raise ValueError("provider config must define a model")

    middleware: list[object] = []
    if event_store is not None and tool_call_store is not None:
        middleware.append(
            build_tool_call_middleware(
                tool_call_store=tool_call_store,
                event_store=event_store,
                now=lambda: int(time.time()),
            )
        )

    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url")
    if not api_key and base_url:
        api_key = "not-needed"

    return OpenAIChatClient(
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        instruction_role=provider_config.get("instruction_role"),
        middleware=middleware or None,
    )


def _resolve_tools(
    definition: DefinitionDocument,
    tool_registry: Mapping[str, Callable[..., Any]],
    *,
    ignore_missing_tools: bool = False,
    include_mcp_servers: bool = True,
) -> list[object]:
    tool_names = definition.metadata.get("tools", [])
    if tool_names and not isinstance(tool_names, list):
        raise ValueError(f"tools for '{definition.id}' must be a list")

    resolved: list[Callable[..., Any] | object] = []
    for tool_name in tool_names or []:
        if tool_name not in tool_registry:
            if ignore_missing_tools:
                continue
            raise ValueError(f"unknown tool: {tool_name}")
        resolved.append(tool_registry[tool_name])
    if include_mcp_servers:
        resolved.extend(_resolve_mcp_tools(definition))
    return resolved


def _resolve_mcp_tools(definition: DefinitionDocument) -> list[object]:
    servers = definition.metadata.get("mcp_servers", [])
    if not servers:
        return []
    if not isinstance(servers, list):
        raise ValueError(f"mcp_servers for '{definition.id}' must be a list")

    tools: list[object] = []
    for server in servers:
        if not isinstance(server, dict):
            raise ValueError(f"mcp server entries for '{definition.id}' must be mappings")
        tools.append(_build_mcp_tool(server))
    return tools


def _build_mcp_tool(config: dict[str, Any]) -> object:
    name = config.get("name")
    transport = config.get("transport")
    if not isinstance(name, str) or not name:
        raise ValueError("mcp server entries must define a non-empty name")
    if transport not in {"stdio", "streamable_http", "websocket"}:
        raise ValueError(f"unsupported mcp transport: {transport}")

    common_kwargs: dict[str, Any] = {
        "description": config.get("description"),
        "approval_mode": config.get("approval_mode"),
        "allowed_tools": config.get("allowed_tools"),
    }
    if transport == "stdio":
        command = config.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError(f"mcp stdio server '{name}' must define command")
        args = config.get("args")
        env = config.get("env")
        return MCPStdioTool(
            name=name,
            command=command,
            args=args if isinstance(args, list) else None,
            env=env if isinstance(env, dict) else None,
            **common_kwargs,
        )

    url = config.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"mcp server '{name}' must define url")
    if transport == "streamable_http":
        return MCPStreamableHTTPTool(name=name, url=url, **common_kwargs)
    return MCPWebsocketTool(name=name, url=url, **common_kwargs)


def _build_context_providers(
    *,
    registry: RuntimeRegistry,
    definition: DefinitionDocument,
    provider_name: str,
    provider_capabilities: dict[str, object],
    conversation_store: ConversationStore | None,
    memory_store: MemoryStore | None,
    artifact_store: ArtifactStore | None,
    whiteboard_store: WhiteboardStore | None,
    event_store: EventStore | None,
    retrieval_index: RetrievalIndex | None,
) -> list[object]:
    providers: list[object] = [
        AgentProfileContextProvider(
            definition,
            registry=registry,
            provider_name=provider_name,
            provider_capabilities=provider_capabilities,
        )
    ]
    if conversation_store is not None and event_store is not None:
        providers.append(ConversationHistoryProvider(conversation_store, event_store))
    if whiteboard_store is not None and event_store is not None:
        providers.append(TaskWhiteboardContextProvider(whiteboard_store, event_store))
    if memory_store is not None and event_store is not None:
        providers.append(ProjectMemoryContextProvider(memory_store, event_store))
    if artifact_store is not None and event_store is not None:
        providers.append(ArtifactContextProvider(artifact_store, event_store))
    if retrieval_index is not None and event_store is not None:
        providers.append(
            RetrievalContextProvider(
                retrieval_index,
                event_store,
                conversation_store=conversation_store,
                whiteboard_store=whiteboard_store,
            )
        )
    return providers
