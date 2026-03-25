from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.proxy.session_overlay import SessionOverlay, make_session_overlay


class SessionOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _overlay(self) -> SessionOverlay:
        return SessionOverlay(root=self.tmp / "overlay")

    # --- read ---

    def test_read_falls_back_to_real_filesystem(self) -> None:
        real = self.tmp / "proj" / "foo.py"
        real.parent.mkdir(parents=True)
        real.write_text("real content", encoding="utf-8")
        overlay = self._overlay()
        self.assertEqual(overlay.read_file(str(real)), "real content")

    def test_read_returns_overlay_file_when_present(self) -> None:
        real = self.tmp / "proj" / "foo.py"
        real.parent.mkdir(parents=True)
        real.write_text("real content", encoding="utf-8")
        overlay = self._overlay()
        overlay.write_file(str(real), "overlay content")
        self.assertEqual(overlay.read_file(str(real)), "overlay content")

    def test_read_raises_file_not_found_when_absent_everywhere(self) -> None:
        overlay = self._overlay()
        with self.assertRaises(FileNotFoundError):
            overlay.read_file(str(self.tmp / "nonexistent" / "missing.py"))

    # --- write ---

    def test_write_always_writes_to_overlay_not_real_file(self) -> None:
        real = self.tmp / "proj" / "foo.py"
        real.parent.mkdir(parents=True)
        real.write_text("original", encoding="utf-8")
        overlay = self._overlay()
        overlay.write_file(str(real), "new content")
        self.assertEqual(real.read_text(encoding="utf-8"), "original")

    def test_write_creates_intermediate_directories(self) -> None:
        overlay = self._overlay()
        deep = str(self.tmp / "a" / "b" / "c" / "deep.py")
        overlay.write_file(deep, "content")
        self.assertEqual(overlay.read_file(deep), "content")

    def test_write_new_file_readable_via_overlay(self) -> None:
        overlay = self._overlay()
        path = str(self.tmp / "new_file.py")
        overlay.write_file(path, "hello")
        self.assertEqual(overlay.read_file(path), "hello")

    # --- list_files ---

    def test_list_files_merges_overlay_and_real(self) -> None:
        real_dir = self.tmp / "proj"
        real_dir.mkdir()
        (real_dir / "a.py").write_text("", encoding="utf-8")
        (real_dir / "b.py").write_text("", encoding="utf-8")
        overlay = self._overlay()
        overlay.write_file(str(real_dir / "b.py"), "overridden")
        overlay.write_file(str(real_dir / "c.py"), "only in overlay")
        result = overlay.list_files(str(real_dir))
        self.assertIn(str(real_dir / "a.py"), result)
        self.assertIn(str(real_dir / "b.py"), result)
        self.assertIn(str(real_dir / "c.py"), result)
        self.assertEqual(len(result), 3)

    def test_list_files_shows_overlay_entries_when_real_dir_missing(self) -> None:
        overlay = self._overlay()
        fake_dir = str(self.tmp / "no_such_dir")
        overlay.write_file(fake_dir + "/x.py", "content")
        result = overlay.list_files(fake_dir)
        self.assertIn(fake_dir + "/x.py", result)

    def test_list_files_returns_empty_when_neither_exists(self) -> None:
        overlay = self._overlay()
        result = overlay.list_files(str(self.tmp / "ghost_dir"))
        self.assertEqual(result, [])

    def test_list_files_no_duplicates(self) -> None:
        real_dir = self.tmp / "proj"
        real_dir.mkdir()
        (real_dir / "a.py").write_text("", encoding="utf-8")
        overlay = self._overlay()
        overlay.write_file(str(real_dir / "a.py"), "overridden")
        result = overlay.list_files(str(real_dir))
        self.assertEqual(result.count(str(real_dir / "a.py")), 1)

    # --- path mirroring ---

    def test_overlay_path_mirrors_absolute_path_structure(self) -> None:
        root = self.tmp / "overlay"
        overlay = SessionOverlay(root=root)
        real_path = str(self.tmp / "proj" / "src" / "foo.py")
        overlay.write_file(real_path, "x")
        # The file must be stored under root, mirroring the full absolute path
        overlay_file = root / Path(real_path).relative_to("/")
        self.assertTrue(overlay_file.exists())

    # --- factory ---

    def test_make_session_overlay_uses_home_ergon_workspace(self) -> None:
        overlay = make_session_overlay("test-session-abc")
        expected_root = Path.home() / ".ergon-workspace" / "test-session-abc"
        self.assertEqual(overlay.root, expected_root)
