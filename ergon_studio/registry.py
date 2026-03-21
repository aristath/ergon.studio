from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.upstream import UpstreamSettings


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
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        workflow_definitions=load_definitions_from_dir(workflows_dir),
    )
