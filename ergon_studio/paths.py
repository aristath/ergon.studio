from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


STUDIO_DIRNAME = ".ergon.studio"


@dataclass(frozen=True)
class StudioPaths:
    home_dir: Path
    project_root: Path
    project_uuid: UUID

    @property
    def studio_home(self) -> Path:
        return self.home_dir / STUDIO_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.studio_home / "config.json"

    @property
    def agents_dir(self) -> Path:
        return self.studio_home / "agents"

    @property
    def workflows_dir(self) -> Path:
        return self.studio_home / "workflows"

    @property
    def project_identity_path(self) -> Path:
        return self.project_root / STUDIO_DIRNAME / "project.json"

    @property
    def project_data_dir(self) -> Path:
        return self.studio_home / str(self.project_uuid)

    @property
    def state_db_path(self) -> Path:
        return self.project_data_dir / "state.db"

    @property
    def sessions_dir(self) -> Path:
        return self.project_data_dir / "sessions"

    @property
    def threads_dir(self) -> Path:
        return self.project_data_dir / "threads"

    @property
    def tasks_dir(self) -> Path:
        return self.project_data_dir / "tasks"

    @property
    def memory_dir(self) -> Path:
        return self.project_data_dir / "memory"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_data_dir / "artifacts"

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_data_dir / "checkpoints"

    @property
    def indexes_dir(self) -> Path:
        return self.project_data_dir / "indexes"

    @property
    def logs_dir(self) -> Path:
        return self.project_data_dir / "logs"

    @property
    def diffs_dir(self) -> Path:
        return self.project_data_dir / "diffs"

    @property
    def exports_dir(self) -> Path:
        return self.project_data_dir / "exports"

    def ensure_global_layout(self) -> None:
        self.studio_home.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

    def ensure_project_layout(self) -> None:
        self.project_data_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.project_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def project_directories(self) -> tuple[Path, ...]:
        return (
            self.sessions_dir,
            self.threads_dir,
            self.tasks_dir,
            self.memory_dir,
            self.artifacts_dir,
            self.checkpoints_dir,
            self.indexes_dir,
            self.logs_dir,
            self.diffs_dir,
            self.exports_dir,
        )
