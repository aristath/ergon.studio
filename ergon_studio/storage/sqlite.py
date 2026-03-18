from __future__ import annotations

import sqlite3
from pathlib import Path

from ergon_studio.storage.models import ApprovalRecord, ArtifactRecord, EventRecord, MemoryFactRecord, MessageRecord, SessionRecord, TaskRecord, ThreadRecord


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      project_uuid TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      summary TEXT,
      parent_task_id TEXT,
      parent_thread_id TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      title TEXT NOT NULL,
      state TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      parent_task_id TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      sender TEXT NOT NULL,
      kind TEXT NOT NULL,
      body_path TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      task_id TEXT,
      artifact_id TEXT,
      tool_call_id TEXT,
      FOREIGN KEY(thread_id) REFERENCES threads(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      summary TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      thread_id TEXT,
      task_id TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approvals (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      requester TEXT NOT NULL,
      action TEXT NOT NULL,
      risk_class TEXT NOT NULL,
      reason TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_facts (
      id TEXT PRIMARY KEY,
      scope TEXT NOT NULL,
      kind TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS artifacts (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      title TEXT NOT NULL,
      file_path TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      thread_id TEXT,
      task_id TEXT,
      FOREIGN KEY(session_id) REFERENCES sessions(id)
    )
    """,
)


def initialize_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_session(self, record: SessionRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, project_uuid, created_at)
                VALUES (?, ?, ?)
                """,
                (record.id, record.project_uuid, record.created_at),
            )
            connection.commit()

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, project_uuid, created_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(id=row[0], project_uuid=row[1], created_at=row[2])

    def insert_thread(self, record: ThreadRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO threads (
                  id, session_id, kind, created_at, updated_at, summary, parent_task_id, parent_thread_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.kind,
                    record.created_at,
                    record.updated_at,
                    record.summary,
                    record.parent_task_id,
                    record.parent_thread_id,
                ),
            )
            connection.commit()

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, session_id, kind, created_at, updated_at, summary, parent_task_id, parent_thread_id
                FROM threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return ThreadRecord(
            id=row[0],
            session_id=row[1],
            kind=row[2],
            created_at=row[3],
            updated_at=row[4],
            summary=row[5],
            parent_task_id=row[6],
            parent_thread_id=row[7],
        )

    def list_threads(self, session_id: str) -> list[ThreadRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, kind, created_at, updated_at, summary, parent_task_id, parent_thread_id
                FROM threads
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ThreadRecord(
                id=row[0],
                session_id=row[1],
                kind=row[2],
                created_at=row[3],
                updated_at=row[4],
                summary=row[5],
                parent_task_id=row[6],
                parent_thread_id=row[7],
            )
            for row in rows
        ]

    def insert_task(self, record: TaskRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                  id, session_id, title, state, created_at, updated_at, parent_task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.title,
                    record.state,
                    record.created_at,
                    record.updated_at,
                    record.parent_task_id,
                ),
            )
            connection.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, session_id, title, state, created_at, updated_at, parent_task_id
                FROM tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return TaskRecord(
            id=row[0],
            session_id=row[1],
            title=row[2],
            state=row[3],
            created_at=row[4],
            updated_at=row[5],
            parent_task_id=row[6],
        )

    def list_tasks(self, session_id: str) -> list[TaskRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, title, state, created_at, updated_at, parent_task_id
                FROM tasks
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            TaskRecord(
                id=row[0],
                session_id=row[1],
                title=row[2],
                state=row[3],
                created_at=row[4],
                updated_at=row[5],
                parent_task_id=row[6],
            )
            for row in rows
        ]

    def insert_message(self, record: MessageRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                  id, thread_id, sender, kind, body_path, created_at, task_id, artifact_id, tool_call_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.thread_id,
                    record.sender,
                    record.kind,
                    str(record.body_path),
                    record.created_at,
                    record.task_id,
                    record.artifact_id,
                    record.tool_call_id,
                ),
            )
            connection.commit()

    def get_message(self, message_id: str) -> MessageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, thread_id, sender, kind, body_path, created_at, task_id, artifact_id, tool_call_id
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return MessageRecord(
            id=row[0],
            thread_id=row[1],
            sender=row[2],
            kind=row[3],
            body_path=Path(row[4]),
            created_at=row[5],
            task_id=row[6],
            artifact_id=row[7],
            tool_call_id=row[8],
        )

    def list_messages(self, thread_id: str) -> list[MessageRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, thread_id, sender, kind, body_path, created_at, task_id, artifact_id, tool_call_id
                FROM messages
                WHERE thread_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (thread_id,),
            ).fetchall()
        return [
            MessageRecord(
                id=row[0],
                thread_id=row[1],
                sender=row[2],
                kind=row[3],
                body_path=Path(row[4]),
                created_at=row[5],
                task_id=row[6],
                artifact_id=row[7],
                tool_call_id=row[8],
            )
            for row in rows
        ]

    def insert_event(self, record: EventRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events (
                  id, session_id, kind, summary, created_at, thread_id, task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.kind,
                    record.summary,
                    record.created_at,
                    record.thread_id,
                    record.task_id,
                ),
            )
            connection.commit()

    def list_events(self, session_id: str) -> list[EventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, kind, summary, created_at, thread_id, task_id
                FROM events
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            EventRecord(
                id=row[0],
                session_id=row[1],
                kind=row[2],
                summary=row[3],
                created_at=row[4],
                thread_id=row[5],
                task_id=row[6],
            )
            for row in rows
        ]

    def insert_approval(self, record: ApprovalRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO approvals (
                  id, session_id, requester, action, risk_class, reason, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.requester,
                    record.action,
                    record.risk_class,
                    record.reason,
                    record.status,
                    record.created_at,
                ),
            )
            connection.commit()

    def list_approvals(self, session_id: str) -> list[ApprovalRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, requester, action, risk_class, reason, status, created_at
                FROM approvals
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ApprovalRecord(
                id=row[0],
                session_id=row[1],
                requester=row[2],
                action=row[3],
                risk_class=row[4],
                reason=row[5],
                status=row[6],
                created_at=row[7],
            )
            for row in rows
        ]

    def insert_memory_fact(self, record: MemoryFactRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_facts (id, scope, kind, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.scope,
                    record.kind,
                    record.content,
                    record.created_at,
                ),
            )
            connection.commit()

    def list_memory_facts(self) -> list[MemoryFactRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, scope, kind, content, created_at
                FROM memory_facts
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [
            MemoryFactRecord(
                id=row[0],
                scope=row[1],
                kind=row[2],
                content=row[3],
                created_at=row[4],
            )
            for row in rows
        ]

    def insert_artifact(self, record: ArtifactRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                  id, session_id, kind, title, file_path, created_at, thread_id, task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.session_id,
                    record.kind,
                    record.title,
                    str(record.file_path),
                    record.created_at,
                    record.thread_id,
                    record.task_id,
                ),
            )
            connection.commit()

    def list_artifacts(self, session_id: str) -> list[ArtifactRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, kind, title, file_path, created_at, thread_id, task_id
                FROM artifacts
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ArtifactRecord(
                id=row[0],
                session_id=row[1],
                kind=row[2],
                title=row[3],
                file_path=Path(row[4]),
                created_at=row[5],
                thread_id=row[6],
                task_id=row[7],
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
