from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ergon_studio.cli import main


class CliTests(unittest.TestCase):
    def test_bootstrap_command_initializes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "bootstrap",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((project_root / ".ergon.studio" / "project.json").exists())
            self.assertIn("project_uuid", stdout.getvalue())
