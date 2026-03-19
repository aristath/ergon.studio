from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_create_list_rename_and_archive_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = SessionStore(paths)

            first = store.create_session(session_id="session-main", created_at=10)
            second = store.create_session(created_at=20, title="Bugfix lane")
            renamed = store.rename_session(
                session_id=second.id,
                title="Bugfix session",
                updated_at=30,
            )
            archived = store.archive_session(session_id=first.id, archived_at=40)

            visible = store.list_sessions()
            all_sessions = store.list_sessions(include_archived=True)

            self.assertEqual(first.title, "Main Session")
            self.assertEqual(renamed.title, "Bugfix session")
            self.assertEqual([session.id for session in visible], [second.id])
            self.assertEqual([session.id for session in all_sessions], [first.id, second.id])
            self.assertEqual(archived.archived_at, 40)

    def test_default_titles_use_session_suffix_for_generated_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = SessionStore(paths)

            created = store.create_session(session_id="session-abc12345", created_at=10)

            self.assertEqual(created.title, "Session abc12345")

    def test_latest_session_prefers_most_recent_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = SessionStore(paths)

            first = store.create_session(session_id="session-main", created_at=10)
            second = store.create_session(created_at=20, title="Feature branch")
            store.touch_session(session_id=first.id, updated_at=50)

            latest = store.latest_session()

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.id, first.id)
            self.assertEqual(latest.updated_at, 50)
