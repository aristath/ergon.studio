from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.agent_factory import build_agent
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.tool_registry import build_workspace_tool_registry


@dataclass(frozen=True)
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]

    def build_agent(self, agent_id: str):
        return build_agent(self.registry, agent_id, tool_registry=self.tool_registry)


def load_runtime(project_root: Path, home_dir: Path) -> RuntimeContext:
    paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
    registry = load_registry(paths)
    tool_registry = build_workspace_tool_registry(paths.project_root)
    return RuntimeContext(
        paths=paths,
        registry=registry,
        tool_registry=tool_registry,
    )
