from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SessionOverlay:
    """Copy-on-write file overlay for an agent sub-session.

    Reads check the primary overlay first, then each read layer in order,
    then fall back to the real filesystem.  Writes always go to the primary
    overlay (``root``), leaving everything else untouched.

    ``read_layers`` allows a later sub-session (e.g. a reviewer) to see files
    written by earlier sub-sessions without sharing a writable workspace with
    them.
    """

    root: Path
    read_layers: tuple[Path, ...] = field(default_factory=tuple)

    def read_file(self, path: str) -> str:
        overlay_path = self._to_overlay(self.root, Path(path))
        if overlay_path.exists():
            return overlay_path.read_text(encoding="utf-8")
        for layer in self.read_layers:
            layer_path = self._to_overlay(layer, Path(path))
            if layer_path.exists():
                return layer_path.read_text(encoding="utf-8")
        real = Path(path)
        if real.exists():
            return real.read_text(encoding="utf-8")
        raise FileNotFoundError(path)

    def write_file(self, path: str, content: str) -> None:
        overlay_path = self._to_overlay(self.root, Path(path))
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(content, encoding="utf-8")

    def list_files(self, directory: str) -> list[str]:
        abs_dir = Path(directory)
        names: set[str] = set()
        if abs_dir.is_dir():
            names.update(p.name for p in abs_dir.iterdir())
        overlay_dir = self._to_overlay(self.root, abs_dir)
        if overlay_dir.is_dir():
            names.update(p.name for p in overlay_dir.iterdir())
        for layer in self.read_layers:
            layer_dir = self._to_overlay(layer, abs_dir)
            if layer_dir.is_dir():
                names.update(p.name for p in layer_dir.iterdir())
        return sorted(str(abs_dir / name) for name in names)

    @staticmethod
    def _to_overlay(overlay_root: Path, abs_path: Path) -> Path:
        # Strip the leading "/" to make the path relative before joining.
        relative = Path(*abs_path.parts[1:])
        return overlay_root / relative


def make_session_overlay(session_id: str) -> SessionOverlay:
    """Return a SessionOverlay rooted at ~/.ergon-workspace/<session_id>/."""
    root = Path.home() / ".ergon-workspace" / session_id
    return SessionOverlay(root=root)
