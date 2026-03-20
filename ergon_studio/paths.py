from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DefinitionPaths:
    root_dir: Path

    @property
    def agents_dir(self) -> Path:
        return self.root_dir / "agents"

    @property
    def workflows_dir(self) -> Path:
        return self.root_dir / "workflows"
