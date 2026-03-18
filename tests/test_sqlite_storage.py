from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from ergon_studio.storage.models import CommandRunRecord, MessageRecord, SessionRecord, TaskRecord, ThreadRecord, WorkflowRunRecord
from ergon_studio.storage.sqlite import MetadataStore, initialize_database


class InitializeDatabaseTests(unittest.TestCase):
    def test_initialize_database_creates_expected_metadata_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"

            initialize_database(db_path)

            with sqlite3.connect(db_path) as connection:
                table_names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

            self.assertTrue({"sessions", "threads", "tasks", "messages", "command_runs"}.issubset(table_names))


class MetadataStoreTests(unittest.TestCase):
    def test_can_insert_and_fetch_session_thread_task_and_message_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            initialize_database(db_path)
            store = MetadataStore(db_path)

            session = SessionRecord(
                id="session-1",
                project_uuid="12345678-1234-5678-1234-567812345678",
                created_at=1_710_755_200,
            )
            thread = ThreadRecord(
                id="thread-1",
                session_id="session-1",
                kind="main",
                created_at=1_710_755_200,
                updated_at=1_710_755_260,
                assigned_agent_id="orchestrator",
                summary="Main thread",
            )
            task = TaskRecord(
                id="task-1",
                session_id="session-1",
                title="Build foundation",
                state="in_progress",
                created_at=1_710_755_300,
                updated_at=1_710_755_360,
            )
            message = MessageRecord(
                id="message-1",
                thread_id="thread-1",
                sender="orchestrator",
                kind="chat",
                body_path=Path("threads/thread-1/messages/message-1.md"),
                created_at=1_710_755_400,
                task_id="task-1",
            )

            store.insert_session(session)
            store.insert_thread(thread)
            store.insert_task(task)
            store.insert_message(message)
            workflow_run = WorkflowRunRecord(
                id="workflow-run-1",
                session_id="session-1",
                workflow_id="standard-build",
                state="running",
                created_at=1_710_755_500,
                updated_at=1_710_755_500,
                root_task_id="task-1",
                current_step_index=1,
                last_thread_id="thread-1",
            )
            store.insert_workflow_run(workflow_run)
            command_run = CommandRunRecord(
                id="command-run-1",
                session_id="session-1",
                command="pwd",
                cwd="/workspace",
                exit_code=0,
                status="completed",
                output_path=Path("logs/commands/command-run-1.md"),
                created_at=1_710_755_600,
                thread_id="thread-1",
                task_id="task-1",
                agent_id="orchestrator",
            )
            store.insert_command_run(command_run)

            self.assertEqual(store.get_session("session-1"), session)
            self.assertEqual(store.get_thread("thread-1"), thread)
            self.assertEqual(store.get_task("task-1"), task)
            self.assertEqual(store.get_message("message-1"), message)
            self.assertEqual(store.get_workflow_run("workflow-run-1"), workflow_run)
            self.assertEqual(store.list_command_runs("session-1"), [command_run])

    def test_message_metadata_stores_body_path_not_message_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.db"
            initialize_database(db_path)
            store = MetadataStore(db_path)

            session = SessionRecord(
                id="session-1",
                project_uuid="12345678-1234-5678-1234-567812345678",
                created_at=1,
            )
            thread = ThreadRecord(
                id="thread-1",
                session_id="session-1",
                kind="main",
                created_at=1,
                updated_at=1,
                assigned_agent_id=None,
            )
            message = MessageRecord(
                id="message-1",
                thread_id="thread-1",
                sender="orchestrator",
                kind="chat",
                body_path=Path("threads/thread-1/messages/message-1.md"),
                created_at=2,
            )

            store.insert_session(session)
            store.insert_thread(thread)
            store.insert_message(message)

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    "SELECT body_path, created_at FROM messages WHERE id = ?",
                    ("message-1",),
                ).fetchone()

            self.assertEqual(row, ("threads/thread-1/messages/message-1.md", 2))
