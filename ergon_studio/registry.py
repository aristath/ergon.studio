from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ergon_studio.config import load_or_create_global_config
from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.paths import GlobalStudioPaths


@dataclass(frozen=True)
class RuntimeRegistry:
    config: dict[str, Any]
    agent_definitions: dict[str, DefinitionDocument]
    workflow_definitions: dict[str, DefinitionDocument]


def load_registry(paths: GlobalStudioPaths) -> RuntimeRegistry:
    return load_registry_from_layout(
        config_path=paths.config_path,
        agents_dir=paths.agents_dir,
        workflows_dir=paths.workflows_dir,
    )


def load_registry_from_layout(*, config_path: Path, agents_dir: Path, workflows_dir: Path) -> RuntimeRegistry:
    return RuntimeRegistry(
        config=load_or_create_global_config(config_path),
        agent_definitions=load_definitions_from_dir(agents_dir),
        workflow_definitions=load_definitions_from_dir(workflows_dir),
    )
