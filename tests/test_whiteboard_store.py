from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.whiteboard_store import WhiteboardStore


class WhiteboardStoreTests(unittest.TestCase):
    def test_ensure_and_update_task_whiteboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = WhiteboardStore(paths)

            created = store.ensure_task_whiteboard(
                task_id="task-1",
                title="Build memory system",
                updated_at=10,
                goal="Implement the framework-native memory layer.",
            )
            updated = store.update_task_whiteboard(
                task_id="task-1",
                updated_at=20,
                section_updates={
                    "Decisions": "Use task whiteboards plus durable project memory facts.",
                    "Acceptance Criteria": "Agents receive whiteboard context before each run.",
                },
            )

            self.assertEqual(created.task_id, "task-1")
            self.assertEqual(updated.updated_at, 20)
            self.assertEqual(
                updated.sections["Acceptance Criteria"],
                "Agents receive whiteboard context before each run.",
            )
            self.assertIn("## Goal", store.read_task_whiteboard_text("task-1"))

    def test_save_task_whiteboard_text_requires_matching_task_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = WhiteboardStore(paths)
            store.ensure_task_whiteboard(
                task_id="task-1",
                title="Build memory system",
                updated_at=10,
            )

            with self.assertRaisesRegex(ValueError, "must match the selected task"):
                store.save_task_whiteboard_text(
                    "task-1",
                    """---
task_id: task-2
title: Wrong
updated_at: 10
---
## Goal
Nope.
""",
                )
