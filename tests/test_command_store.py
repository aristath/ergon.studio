from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.command_store import CommandStore


class CommandStoreTests(unittest.TestCase):
    def test_command_store_persists_metadata_and_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = CommandStore(paths)

            record = store.create_command_run(
                session_id="session-main",
                command_run_id="command-run-1",
                command="pwd",
                cwd=str(project_root),
                exit_code=0,
                status="completed",
                output_content="# Command Run\n\nok\n",
                created_at=1_710_755_200,
                thread_id="thread-main",
                task_id="task-1",
                agent_id="orchestrator",
            )

            self.assertEqual(record.id, "command-run-1")
            self.assertEqual(record.exit_code, 0)
            self.assertTrue(record.output_path.exists())
            self.assertEqual(store.list_command_runs("session-main"), [record])
            self.assertEqual(store.read_command_output(record), "# Command Run\n\nok\n")
