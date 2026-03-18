from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.task_store import TaskStore
from ergon_studio.storage.sqlite import initialize_database


class TaskStoreTests(unittest.TestCase):
    def test_create_and_list_tasks_persist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = TaskStore(paths)

            first = store.create_task(
                session_id="session-main",
                task_id="task-1",
                title="Build shell",
                state="in_progress",
                created_at=10,
            )
            second = store.create_task(
                session_id="session-main",
                task_id="task-2",
                title="Add orchestration",
                state="planned",
                created_at=20,
            )

            self.assertEqual(first.id, "task-1")
            self.assertEqual(second.title, "Add orchestration")
            self.assertEqual(
                [task.id for task in store.list_tasks("session-main")],
                ["task-1", "task-2"],
            )

    def test_list_tasks_returns_created_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = TaskStore(paths)

            store.create_task(
                session_id="session-main",
                task_id="task-2",
                title="Second",
                state="planned",
                created_at=20,
            )
            store.create_task(
                session_id="session-main",
                task_id="task-1",
                title="First",
                state="in_progress",
                created_at=10,
            )

            tasks = store.list_tasks("session-main")

            self.assertEqual([task.id for task in tasks], ["task-1", "task-2"])
