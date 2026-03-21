from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.workspace import ensure_workspace


class WorkspaceTests(unittest.TestCase):
    def test_ensure_workspace_creates_config_and_definition_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = ensure_workspace(Path(temp_dir))

            self.assertEqual(paths.app_dir, Path(temp_dir))
            self.assertTrue(paths.agents_dir.exists())
            self.assertTrue(paths.workflows_dir.exists())
            self.assertTrue((paths.agents_dir / "orchestrator.md").exists())
            self.assertTrue((paths.workflows_dir / "standard-build.md").exists())

    def test_ensure_workspace_preserves_existing_definition_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            agents_dir = Path(temp_dir) / "definitions" / "agents"
            agents_dir.mkdir(parents=True)
            custom_path = agents_dir / "orchestrator.md"
            custom_path.write_text("custom", encoding="utf-8")

            ensure_workspace(Path(temp_dir))

            self.assertEqual(custom_path.read_text(encoding="utf-8"), "custom")
