from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.upstream import UpstreamSettings


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    workroom_definitions: dict[str, tuple[str, ...]]


def load_registry(
    definitions_root: Path, *, upstream: UpstreamSettings
) -> RuntimeRegistry:
    root_dir = Path(definitions_root)
    agents_dir = root_dir / "agents"
    workrooms_dir = root_dir / "workrooms"
    if not agents_dir.exists():
        raise ValueError(f"missing agents directory: {agents_dir}")
    if not workrooms_dir.exists():
        raise ValueError(f"missing workrooms directory: {workrooms_dir}")
    agent_definitions = load_definitions_from_dir(agents_dir)
    if "orchestrator" not in agent_definitions:
        raise ValueError(
            f"missing required agent definition: {agents_dir / 'orchestrator.md'}"
        )
    raw_workroom_definitions = load_definitions_from_dir(workrooms_dir)
    workroom_definitions = _load_workroom_definitions(
        agent_definitions=agent_definitions,
        workroom_definitions=raw_workroom_definitions,
    )
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        workroom_definitions=workroom_definitions,
    )


def _load_workroom_definitions(
    *,
    agent_definitions: dict[str, DefinitionDocument],
    workroom_definitions: dict[str, DefinitionDocument],
) -> dict[str, tuple[str, ...]]:
    known_agents = set(agent_definitions)
    parsed_definitions: dict[str, tuple[str, ...]] = {}
    for workroom_id, definition in workroom_definitions.items():
        participants = _workroom_participants_for_definition(definition)
        referenced_agents = set(participants)
        missing_agents = sorted(
            agent_id for agent_id in referenced_agents if agent_id not in known_agents
        )
        if missing_agents:
            joined = ", ".join(missing_agents)
            raise ValueError(
                f"workroom preset '{workroom_id}' references unknown agents: {joined}"
            )
        parsed_definitions[workroom_id] = participants
    return parsed_definitions


def _workroom_participants_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    configured_participants = definition.metadata.get("participants")
    if configured_participants is None:
        raise ValueError(
            f"workroom preset '{definition.id}' must declare `participants`"
        )
    if not isinstance(configured_participants, list):
        raise ValueError(
            f"workroom preset '{definition.id}' participants must be a list"
        )
    validated: list[str] = []
    for item in configured_participants:
        if not isinstance(item, str):
            raise ValueError(
                f"workroom preset '{definition.id}' participants "
                "must be non-empty strings"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                f"workroom preset '{definition.id}' participants "
                "must be non-empty strings"
            )
        validated.append(stripped)
    return tuple(validated)
