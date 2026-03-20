from __future__ import annotations

from pathlib import Path

from ergon_studio.paths import GlobalStudioPaths
from ergon_studio.seed import seed_default_definitions


def bootstrap_definition_home(home_dir: Path) -> GlobalStudioPaths:
    paths = GlobalStudioPaths(home_dir=home_dir)
    paths.ensure_layout()
    seed_default_definitions(paths.agents_dir, paths.workflows_dir)
    return paths
