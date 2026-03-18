from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from ergon_studio.paths import STUDIO_DIRNAME


@dataclass(frozen=True)
class ProjectIdentity:
    project_uuid: UUID

    def to_dict(self) -> dict[str, str]:
        return {"project_uuid": str(self.project_uuid)}


def project_identity_path(project_root: Path) -> Path:
    return project_root / STUDIO_DIRNAME / "project.json"


def load_project_identity(project_root: Path) -> ProjectIdentity:
    path = project_identity_path(project_root)
    raw = json.loads(path.read_text(encoding="utf-8"))

    if set(raw.keys()) != {"project_uuid"}:
        raise ValueError("project.json must contain only the project_uuid field")

    return ProjectIdentity(project_uuid=UUID(raw["project_uuid"]))


def initialize_project(project_root: Path, project_uuid: UUID | None = None) -> ProjectIdentity:
    path = project_identity_path(project_root)
    if path.exists():
        return load_project_identity(project_root)

    identity = ProjectIdentity(project_uuid=project_uuid or uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return identity
