from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ergon_studio.config import load_or_create_global_config
from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.paths import StudioPaths


@dataclass(frozen=True)
class RuntimeRegistry:
    config: dict[str, Any]
    agent_definitions: dict[str, DefinitionDocument]
    workflow_definitions: dict[str, DefinitionDocument]


def load_registry(paths: StudioPaths) -> RuntimeRegistry:
    return RuntimeRegistry(
        config=load_or_create_global_config(paths.config_path),
        agent_definitions=load_definitions_from_dir(paths.agents_dir),
        workflow_definitions=load_definitions_from_dir(paths.workflows_dir),
    )
