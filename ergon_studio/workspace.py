from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from ergon_studio.app_config import config_path, definitions_dir
from ergon_studio.file_ops import atomic_write_text


@dataclass(frozen=True)
class WorkspacePaths:
    app_dir: Path
    config_path: Path
    definitions_dir: Path
    agents_dir: Path
    workflows_dir: Path


def ensure_workspace(app_dir: Path) -> WorkspacePaths:
    agents_dir = definitions_dir(app_dir) / "agents"
    workflows_dir = definitions_dir(app_dir) / "workflows"
    agents_dir.mkdir(parents=True, exist_ok=True)
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _seed_templates(agents_dir, _bundled_templates("agents"))
    _seed_templates(workflows_dir, _bundled_templates("workflows"))
    return WorkspacePaths(
        app_dir=app_dir,
        config_path=config_path(app_dir),
        definitions_dir=definitions_dir(app_dir),
        agents_dir=agents_dir,
        workflows_dir=workflows_dir,
    )


def _seed_templates(
    directory: Path,
    templates: list[tuple[str, str]],
) -> None:
    for filename, content in templates:
        path = directory / filename
        if path.exists():
            continue
        atomic_write_text(path, content)


def _bundled_templates(kind: str) -> list[tuple[str, str]]:
    template_dir = _bundled_definition_root().joinpath(kind)
    if not template_dir.is_dir():
        raise ValueError(f"missing bundled definitions directory: {template_dir}")
    return [
        (entry.name, entry.read_text(encoding="utf-8"))
        for entry in sorted(template_dir.iterdir(), key=lambda item: item.name)
        if entry.is_file() and entry.name.endswith(".md")
    ]


def _bundled_definition_root() -> Traversable:
    return files("ergon_studio").joinpath("default_definitions")
