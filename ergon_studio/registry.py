from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.proxy.workroom_metadata import (
    workroom_finalizers_for_definition,
    workroom_handoffs_for_definition,
    workroom_selection_sequence_for_definition,
    workroom_start_agent_for_definition,
)
from ergon_studio.upstream import UpstreamSettings
from ergon_studio.workroom_compiler import workroom_step_groups_for_definition


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    workroom_definitions: dict[str, DefinitionDocument]


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
    workroom_definitions = load_definitions_from_dir(workrooms_dir)
    _validate_workroom_references(
        agent_definitions=agent_definitions,
        workroom_definitions=workroom_definitions,
    )
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        workroom_definitions=workroom_definitions,
    )


def _validate_workroom_references(
    *,
    agent_definitions: dict[str, DefinitionDocument],
    workroom_definitions: dict[str, DefinitionDocument],
) -> None:
    known_agents = set(agent_definitions)
    for workroom_id, definition in workroom_definitions.items():
        referenced_agents: set[str] = set()

        for group in workroom_step_groups_for_definition(definition):
            referenced_agents.update(group)
        referenced_agents.update(workroom_selection_sequence_for_definition(definition))
        referenced_agents.update(workroom_finalizers_for_definition(definition))

        start_agent = workroom_start_agent_for_definition(definition)
        if start_agent is not None:
            referenced_agents.add(start_agent)

        for source_agent, target_agents in workroom_handoffs_for_definition(
            definition
        ).items():
            referenced_agents.add(source_agent)
            referenced_agents.update(target_agents)

        missing_agents = sorted(
            agent_id for agent_id in referenced_agents if agent_id not in known_agents
        )
        if missing_agents:
            joined = ", ".join(missing_agents)
            raise ValueError(
                f"workroom template '{workroom_id}' references unknown agents: {joined}"
            )
