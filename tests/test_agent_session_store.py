from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from agent_framework import AgentSession

from ergon_studio.agent_session_store import AgentSessionStore
from ergon_studio.paths import StudioPaths


class AgentSessionStoreTests(unittest.TestCase):
    def test_save_and_load_session_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            home_dir = base / "home"
            project_root = base / "repo"
            home_dir.mkdir()
            project_root.mkdir()
            paths = StudioPaths(
                home_dir=home_dir,
                project_root=project_root,
                project_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            )
            paths.ensure_project_layout()
            store = AgentSessionStore(paths)
            session = AgentSession(session_id="thread-main:orchestrator")

            store.save_session(thread_id="thread-main", agent_id="orchestrator", session=session)
            loaded = store.load_session(thread_id="thread-main", agent_id="orchestrator")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.session_id, "thread-main:orchestrator")

    def test_load_or_create_session_uses_stable_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            home_dir = base / "home"
            project_root = base / "repo"
            home_dir.mkdir()
            project_root.mkdir()
            paths = StudioPaths(
                home_dir=home_dir,
                project_root=project_root,
                project_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            )
            paths.ensure_project_layout()
            store = AgentSessionStore(paths)
            created_ids: list[str] = []

            session = store.load_or_create_session(
                thread_id="thread-main",
                agent_id="orchestrator",
                session_factory=lambda session_id: created_ids.append(session_id) or AgentSession(session_id=session_id),
            )

            self.assertEqual(session.session_id, "thread-main:orchestrator")
            self.assertEqual(created_ids, ["thread-main:orchestrator"])
