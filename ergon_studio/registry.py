from __future__ import annotations

from dataclasses import dataclass

from ergon_studio.definitions import DefinitionDocument, load_definitions_from_dir
from ergon_studio.paths import GlobalStudioPaths
from ergon_studio.upstream import UpstreamSettings


@dataclass(frozen=True)
class RuntimeRegistry:
    upstream: UpstreamSettings
    agent_definitions: dict[str, DefinitionDocument]
    workflow_definitions: dict[str, DefinitionDocument]


def load_registry(paths: GlobalStudioPaths, *, upstream: UpstreamSettings) -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=upstream,
        agent_definitions=load_definitions_from_dir(paths.agents_dir),
        workflow_definitions=load_definitions_from_dir(paths.workflows_dir),
    )
