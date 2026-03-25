from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionOverlay:
    """Copy-on-write file overlay for an agent sub-session.

    Reads check the overlay first, then fall back to the real filesystem.
    Writes always go to the overlay, leaving the real filesystem untouched.
    The overlay mirrors absolute paths under ``root``.
    """

    root: Path

    def read_file(self, path: str) -> str:
        overlay_path = self._overlay_path(Path(path))
        if overlay_path.exists():
            return overlay_path.read_text(encoding="utf-8")
        real = Path(path)
        if real.exists():
            return real.read_text(encoding="utf-8")
        raise FileNotFoundError(path)

    def write_file(self, path: str, content: str) -> None:
        overlay_path = self._overlay_path(Path(path))
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.write_text(content, encoding="utf-8")

    def list_files(self, directory: str) -> list[str]:
        abs_dir = Path(directory)
        overlay_dir = self._overlay_path(abs_dir)
        names: set[str] = set()
        if abs_dir.is_dir():
            names.update(p.name for p in abs_dir.iterdir())
        if overlay_dir.is_dir():
            names.update(p.name for p in overlay_dir.iterdir())
        return sorted(str(abs_dir / name) for name in names)

    def _overlay_path(self, abs_path: Path) -> Path:
        # Strip the leading "/" to make the path relative before joining.
        relative = Path(*abs_path.parts[1:])
        return self.root / relative


def make_session_overlay(session_id: str) -> SessionOverlay:
    """Return a SessionOverlay rooted at ~/.ergon-workspace/<session_id>/."""
    root = Path.home() / ".ergon-workspace" / session_id
    return SessionOverlay(root=root)
