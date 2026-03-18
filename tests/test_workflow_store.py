from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.storage.sqlite import initialize_database
from ergon_studio.workflow_store import WorkflowStore


class WorkflowStoreTests(unittest.TestCase):
    def test_create_and_list_workflow_runs_persist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = WorkflowStore(paths)

            first = store.create_workflow_run(
                session_id="session-main",
                workflow_run_id="workflow-run-1",
                workflow_id="standard-build",
                state="running",
                created_at=10,
                root_task_id="task-1",
            )
            second = store.create_workflow_run(
                session_id="session-main",
                workflow_run_id="workflow-run-2",
                workflow_id="test-driven-repair",
                state="running",
                created_at=20,
                root_task_id="task-2",
            )

            self.assertEqual(first.workflow_id, "standard-build")
            self.assertEqual(second.root_task_id, "task-2")
            self.assertEqual(first.current_step_index, 0)
            self.assertEqual(
                [run.id for run in store.list_workflow_runs("session-main")],
                ["workflow-run-1", "workflow-run-2"],
            )
