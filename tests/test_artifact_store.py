from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.storage.sqlite import initialize_database


class ArtifactStoreTests(unittest.TestCase):
    def test_create_and_list_artifacts_persist_metadata_and_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = ArtifactStore(paths)

            artifact = store.create_artifact(
                session_id="session-main",
                artifact_id="artifact-1",
                kind="design-note",
                title="Architecture Notes",
                content="Use Textual with a runtime-first architecture.",
                created_at=10,
            )

            self.assertEqual(artifact.id, "artifact-1")
            self.assertTrue(artifact.file_path.exists())
            self.assertEqual(
                artifact.file_path.read_text(encoding="utf-8"),
                "Use Textual with a runtime-first architecture.\n",
            )
            self.assertEqual(
                [item.id for item in store.list_artifacts("session-main")],
                ["artifact-1"],
            )
