from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.proxy.workflow_metadata import (
    workflow_finalizers_for_definition,
    workflow_handoffs_for_definition,
    workflow_selection_sequence_for_definition,
    workflow_start_agent_for_definition,
)
from ergon_studio.upstream import UpstreamSettings
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    workflow_definitions: dict[str, DefinitionDocument]


def load_registry(
    definitions_root: Path, *, upstream: UpstreamSettings
) -> RuntimeRegistry:
    root_dir = Path(definitions_root)
    agents_dir = root_dir / "agents"
    workflows_dir = root_dir / "workflows"
    if not agents_dir.exists():
        raise ValueError(f"missing agents directory: {agents_dir}")
    if not workflows_dir.exists():
        raise ValueError(f"missing workflows directory: {workflows_dir}")
    agent_definitions = load_definitions_from_dir(agents_dir)
    if "orchestrator" not in agent_definitions:
        raise ValueError(
            f"missing required agent definition: {agents_dir / 'orchestrator.md'}"
        )
    workflow_definitions = load_definitions_from_dir(workflows_dir)
    _validate_workflow_references(
        agent_definitions=agent_definitions,
        workflow_definitions=workflow_definitions,
    )
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        workflow_definitions=workflow_definitions,
    )


def _validate_workflow_references(
    *,
    agent_definitions: dict[str, DefinitionDocument],
    workflow_definitions: dict[str, DefinitionDocument],
) -> None:
    known_agents = set(agent_definitions)
    for workflow_id, definition in workflow_definitions.items():
        referenced_agents: set[str] = set()

        for group in workflow_step_groups_for_definition(definition):
            referenced_agents.update(group)
        referenced_agents.update(workflow_selection_sequence_for_definition(definition))
        referenced_agents.update(workflow_finalizers_for_definition(definition))

        start_agent = workflow_start_agent_for_definition(definition)
        if start_agent is not None:
            referenced_agents.add(start_agent)

        for source_agent, target_agents in workflow_handoffs_for_definition(
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
                f"workflow '{workflow_id}' references unknown agents: {joined}"
            )
