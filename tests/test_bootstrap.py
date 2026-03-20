from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_proxy_home, bootstrap_workspace


class BootstrapWorkspaceTests(unittest.TestCase):
    def test_bootstrap_workspace_creates_project_and_storage_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)

            self.assertTrue(paths.project_identity_path.exists())
            self.assertTrue(paths.config_path.exists())
            self.assertTrue(paths.agents_dir.exists())
            self.assertTrue(paths.workflows_dir.exists())
            self.assertTrue((paths.agents_dir / "orchestrator.md").exists())
            self.assertTrue((paths.agents_dir / "coder.md").exists())
            self.assertTrue((paths.workflows_dir / "standard-build.md").exists())
            self.assertTrue((paths.workflows_dir / "best-of-n.md").exists())
            self.assertTrue(paths.project_data_dir.exists())
            self.assertTrue(paths.state_db_path.exists())
            self.assertTrue(paths.sessions_dir.exists())
            self.assertTrue(paths.threads_dir.exists())
            self.assertTrue(paths.tasks_dir.exists())
            self.assertTrue(paths.memory_dir.exists())
            self.assertTrue(paths.artifacts_dir.exists())
            self.assertTrue(paths.checkpoints_dir.exists())
            self.assertTrue(paths.indexes_dir.exists())
            self.assertTrue(paths.logs_dir.exists())
            self.assertTrue(paths.diffs_dir.exists())
            self.assertTrue(paths.exports_dir.exists())

    def test_bootstrap_workspace_reuses_existing_project_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            second = bootstrap_workspace(project_root=project_root, home_dir=home_dir)

            self.assertEqual(first.project_uuid, second.project_uuid)
            self.assertEqual(first.project_data_dir, second.project_data_dir)

    def test_bootstrap_workspace_does_not_overwrite_existing_definition_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            agents_dir = home_dir / ".ergon.studio" / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            orchestrator_path = agents_dir / "orchestrator.md"
            orchestrator_path.write_text("custom orchestrator\n", encoding="utf-8")

            bootstrap_workspace(project_root=project_root, home_dir=home_dir)

            self.assertEqual(
                orchestrator_path.read_text(encoding="utf-8"),
                "custom orchestrator\n",
            )

    def test_bootstrap_proxy_home_creates_global_layout_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            paths = bootstrap_proxy_home(home_dir)

            self.assertTrue(paths.config_path.exists())
            self.assertTrue(paths.agents_dir.exists())
            self.assertTrue(paths.workflows_dir.exists())
            self.assertFalse((project_root / ".ergon.studio" / "project.json").exists())
