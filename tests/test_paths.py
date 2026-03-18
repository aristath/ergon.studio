from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from ergon_studio.paths import StudioPaths


class StudioPathsTests(unittest.TestCase):
    def test_paths_follow_the_home_and_project_uuid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            home_dir = base / "home"
            project_root = base / "repo"
            project_uuid = UUID("12345678-1234-5678-1234-567812345678")

            paths = StudioPaths(
                home_dir=home_dir,
                project_root=project_root,
                project_uuid=project_uuid,
            )

            self.assertEqual(paths.config_path, home_dir / ".ergon.studio" / "config.json")
            self.assertEqual(paths.agents_dir, home_dir / ".ergon.studio" / "agents")
            self.assertEqual(paths.workflows_dir, home_dir / ".ergon.studio" / "workflows")
            self.assertEqual(
                paths.project_identity_path,
                project_root / ".ergon.studio" / "project.json",
            )
            self.assertEqual(
                paths.project_data_dir,
                home_dir / ".ergon.studio" / str(project_uuid),
            )
            self.assertEqual(paths.state_db_path, paths.project_data_dir / "state.db")
            self.assertEqual(paths.sessions_dir, paths.project_data_dir / "sessions")
            self.assertEqual(paths.threads_dir, paths.project_data_dir / "threads")
            self.assertEqual(paths.tasks_dir, paths.project_data_dir / "tasks")
            self.assertEqual(paths.memory_dir, paths.project_data_dir / "memory")
            self.assertEqual(paths.whiteboards_dir, paths.project_data_dir / "memory" / "whiteboards")
            self.assertEqual(paths.artifacts_dir, paths.project_data_dir / "artifacts")
            self.assertEqual(paths.checkpoints_dir, paths.project_data_dir / "checkpoints")
            self.assertEqual(paths.indexes_dir, paths.project_data_dir / "indexes")
            self.assertEqual(paths.logs_dir, paths.project_data_dir / "logs")
            self.assertEqual(paths.diffs_dir, paths.project_data_dir / "diffs")
            self.assertEqual(paths.exports_dir, paths.project_data_dir / "exports")
