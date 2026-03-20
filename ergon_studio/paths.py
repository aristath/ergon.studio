from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STUDIO_DIRNAME = ".ergon.studio"


@dataclass(frozen=True)
class GlobalStudioPaths:
    home_dir: Path

    @property
    def studio_home(self) -> Path:
        return self.home_dir / STUDIO_DIRNAME

    @property
    def agents_dir(self) -> Path:
        return self.studio_home / "agents"

    @property
    def workflows_dir(self) -> Path:
        return self.studio_home / "workflows"

    def ensure_layout(self) -> None:
        self.studio_home.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
