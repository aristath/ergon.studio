from __future__ import annotations

from pathlib import Path

from ergon_studio.config import load_or_create_global_config
from ergon_studio.paths import StudioPaths
from ergon_studio.project import initialize_project


def bootstrap_workspace(project_root: Path, home_dir: Path) -> StudioPaths:
    identity = initialize_project(project_root)
    paths = StudioPaths(
        home_dir=home_dir,
        project_root=project_root,
        project_uuid=identity.project_uuid,
    )
    paths.ensure_global_layout()
    paths.ensure_project_layout()
    load_or_create_global_config(paths.config_path)
    return paths
