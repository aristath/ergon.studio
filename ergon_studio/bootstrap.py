from __future__ import annotations

from pathlib import Path

from ergon_studio.config import load_or_create_global_config
from ergon_studio.paths import GlobalStudioPaths
from ergon_studio.seed import seed_default_definitions


def bootstrap_proxy_home(home_dir: Path) -> GlobalStudioPaths:
    paths = GlobalStudioPaths(home_dir=home_dir)
    paths.ensure_layout()
    load_or_create_global_config(paths.config_path)
    seed_default_definitions(paths.agents_dir, paths.workflows_dir)
    return paths
