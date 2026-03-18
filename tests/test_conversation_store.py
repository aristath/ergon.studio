from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.storage.sqlite import initialize_database


class ConversationStoreTests(unittest.TestCase):
    def test_create_thread_and_append_message_persists_metadata_and_markdown_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = ConversationStore(paths)

            session = store.create_session(
                session_id="session-1",
                created_at=1_710_755_200,
            )
            thread = store.create_thread(
                session_id=session.id,
                thread_id="thread-1",
                kind="main",
                created_at=1_710_755_201,
            )
            message = store.append_message(
                thread_id=thread.id,
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Hello from the main thread.",
                created_at=1_710_755_202,
            )

            self.assertEqual(message.id, "message-1")
            self.assertTrue(message.body_path.exists())
            self.assertEqual(
                message.body_path.read_text(encoding="utf-8"),
                "Hello from the main thread.\n",
            )

            stored_messages = store.list_messages(thread.id)
            self.assertEqual([item.id for item in stored_messages], ["message-1"])
            self.assertEqual(store.read_message_body(message), "Hello from the main thread.\n")

    def test_list_messages_returns_created_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = ConversationStore(paths)

            store.create_session(session_id="session-1", created_at=1)
            store.create_thread(
                session_id="session-1",
                thread_id="thread-1",
                kind="main",
                created_at=1,
            )
            store.append_message(
                thread_id="thread-1",
                message_id="message-2",
                sender="orchestrator",
                kind="chat",
                body="second",
                created_at=2,
            )
            store.append_message(
                thread_id="thread-1",
                message_id="message-1",
                sender="user",
                kind="chat",
                body="first",
                created_at=1,
            )

            self.assertEqual(
                [message.id for message in store.list_messages("thread-1")],
                ["message-1", "message-2"],
            )
