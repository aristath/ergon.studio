from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.upstream import UpstreamSettings


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    channel_presets: dict[str, tuple[str, ...]]


def load_registry(
    definitions_root: Path, *, upstream: UpstreamSettings
) -> RuntimeRegistry:
    root_dir = Path(definitions_root)
    agents_dir = root_dir / "agents"
    channels_dir = root_dir / "channels"
    if not agents_dir.exists():
        raise ValueError(f"missing agents directory: {agents_dir}")
    if not channels_dir.exists():
        raise ValueError(f"missing channels directory: {channels_dir}")
    agent_definitions = load_definitions_from_dir(agents_dir)
    if "orchestrator" not in agent_definitions:
        raise ValueError(
            f"missing required agent definition: {agents_dir / 'orchestrator.md'}"
        )
    raw_channel_presets = load_definitions_from_dir(channels_dir)
    channel_presets = _load_channel_presets(
        agent_definitions=agent_definitions,
        channel_presets=raw_channel_presets,
    )
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        channel_presets=channel_presets,
    )


def _load_channel_presets(
    *,
    agent_definitions: dict[str, DefinitionDocument],
    channel_presets: dict[str, DefinitionDocument],
) -> dict[str, tuple[str, ...]]:
    known_agents = set(agent_definitions)
    parsed_definitions: dict[str, tuple[str, ...]] = {}
    for channel_id, definition in channel_presets.items():
        participants = _channel_preset_participants_for_definition(definition)
        referenced_agents = set(participants)
        missing_agents = sorted(
            agent_id for agent_id in referenced_agents if agent_id not in known_agents
        )
        if missing_agents:
            joined = ", ".join(missing_agents)
            raise ValueError(
                f"channel preset '{channel_id}' references unknown agents: {joined}"
            )
        parsed_definitions[channel_id] = participants
    return parsed_definitions


def _channel_preset_participants_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    configured_participants = definition.metadata.get("participants")
    if configured_participants is None:
        raise ValueError(
            f"channel preset '{definition.id}' must declare `participants`"
        )
    if not isinstance(configured_participants, list):
        raise ValueError(
            f"channel preset '{definition.id}' participants must be a list"
        )
    validated: list[str] = []
    for item in configured_participants:
        if not isinstance(item, str):
            raise ValueError(
                f"channel preset '{definition.id}' participants "
                "must be non-empty strings"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                f"channel preset '{definition.id}' participants "
                "must be non-empty strings"
            )
        validated.append(stripped)
    return tuple(validated)
