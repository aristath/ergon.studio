from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.event_store import EventStore
from ergon_studio.storage.sqlite import initialize_database


class EventStoreTests(unittest.TestCase):
    def test_append_and_list_events_persist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = EventStore(paths)

            first = store.append_event(
                session_id="session-main",
                event_id="event-1",
                kind="message_created",
                summary="User sent a message",
                created_at=10,
            )
            second = store.append_event(
                session_id="session-main",
                event_id="event-2",
                kind="task_created",
                summary="Created task task-1",
                created_at=20,
            )

            self.assertEqual(first.id, "event-1")
            self.assertEqual(second.kind, "task_created")
            self.assertEqual(
                [event.id for event in store.list_events("session-main")],
                ["event-1", "event-2"],
            )
