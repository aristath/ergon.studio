from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.paths import DefinitionPaths
from ergon_studio.upstream import UpstreamSettings


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    workflow_definitions: dict[str, DefinitionDocument]


def load_registry(definitions_root: Path | DefinitionPaths, *, upstream: UpstreamSettings) -> RuntimeRegistry:
    paths = definitions_root if isinstance(definitions_root, DefinitionPaths) else DefinitionPaths(Path(definitions_root))
    if not paths.agents_dir.exists():
        raise ValueError(f"missing agents directory: {paths.agents_dir}")
    if not paths.workflows_dir.exists():
        raise ValueError(f"missing workflows directory: {paths.workflows_dir}")
    agent_definitions = load_definitions_from_dir(paths.agents_dir)
    if "orchestrator" not in agent_definitions:
        raise ValueError(f"missing required agent definition: {paths.agents_dir / 'orchestrator.md'}")
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=agent_definitions,
        workflow_definitions=load_definitions_from_dir(paths.workflows_dir),
    )
