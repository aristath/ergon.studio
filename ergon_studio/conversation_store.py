from __future__ import annotations

import time
from pathlib import Path

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import MessageRecord, SessionRecord, ThreadRecord
from ergon_studio.storage.sqlite import MetadataStore


class ConversationStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_session(self, session_id: str, created_at: int, title: str | None = None) -> SessionRecord:
        record = SessionRecord(
            id=session_id,
            project_uuid=str(self.paths.project_uuid),
            title=title or session_id,
            created_at=created_at,
            updated_at=created_at,
            archived_at=None,
        )
        self.metadata.insert_session(record)
        return record

    def ensure_session(self, session_id: str, created_at: int | None = None, title: str | None = None) -> SessionRecord:
        existing = self.metadata.get_session(session_id)
        if existing is not None:
            return existing
        return self.create_session(
            session_id=session_id,
            created_at=created_at if created_at is not None else _now(),
            title=title,
        )

    def create_thread(
        self,
        *,
        session_id: str,
        thread_id: str,
        kind: str,
        created_at: int,
        assigned_agent_id: str | None = None,
        summary: str | None = None,
        parent_task_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> ThreadRecord:
        thread_dir = self.paths.session_threads_dir(session_id) / thread_id / "messages"
        thread_dir.mkdir(parents=True, exist_ok=True)
        record = ThreadRecord(
            id=thread_id,
            session_id=session_id,
            kind=kind,
            created_at=created_at,
            updated_at=created_at,
            assigned_agent_id=assigned_agent_id,
            summary=summary,
            parent_task_id=parent_task_id,
            parent_thread_id=parent_thread_id,
        )
        self.metadata.insert_thread(record)
        self.metadata.touch_session(session_id, updated_at=created_at)
        return record

    def ensure_thread(
        self,
        *,
        session_id: str,
        thread_id: str,
        kind: str,
        created_at: int | None = None,
        assigned_agent_id: str | None = None,
        summary: str | None = None,
        parent_task_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> ThreadRecord:
        existing = self.metadata.get_thread(thread_id)
        if existing is not None:
            return existing
        return self.create_thread(
            session_id=session_id,
            thread_id=thread_id,
            kind=kind,
            created_at=created_at if created_at is not None else _now(),
            assigned_agent_id=assigned_agent_id,
            summary=summary,
            parent_task_id=parent_task_id,
            parent_thread_id=parent_thread_id,
        )

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        return self.metadata.get_thread(thread_id)

    def append_message(
        self,
        *,
        thread_id: str,
        message_id: str,
        sender: str,
        kind: str,
        body: str,
        created_at: int,
        task_id: str | None = None,
        artifact_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> MessageRecord:
        thread = self.metadata.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        message_dir = self.paths.session_threads_dir(thread.session_id) / thread_id / "messages"
        message_dir.mkdir(parents=True, exist_ok=True)
        body_path = message_dir / f"{message_id}.md"
        body_path.write_text(_ensure_trailing_newline(body), encoding="utf-8")

        record = MessageRecord(
            id=message_id,
            thread_id=thread_id,
            sender=sender,
            kind=kind,
            body_path=body_path,
            created_at=created_at,
            task_id=task_id,
            artifact_id=artifact_id,
            tool_call_id=tool_call_id,
        )
        self.metadata.insert_message(record)
        self.metadata.touch_session(thread.session_id, updated_at=created_at)
        return record

    def list_messages(self, thread_id: str) -> list[MessageRecord]:
        return self.metadata.list_messages(thread_id)

    def list_threads(self, session_id: str) -> list[ThreadRecord]:
        return self.metadata.list_threads(session_id)

    def list_sessions(self, *, include_archived: bool = False) -> list[SessionRecord]:
        return self.metadata.list_sessions(
            str(self.paths.project_uuid),
            include_archived=include_archived,
        )

    def read_message_body(self, message: MessageRecord) -> str:
        return Path(message.body_path).read_text(encoding="utf-8")


def _ensure_trailing_newline(body: str) -> str:
    return body if body.endswith("\n") else f"{body}\n"


def _now() -> int:
    return int(time.time())
