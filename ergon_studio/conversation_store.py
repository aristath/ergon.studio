from __future__ import annotations

from pathlib import Path

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import MessageRecord, SessionRecord, ThreadRecord
from ergon_studio.storage.sqlite import MetadataStore


class ConversationStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_session(self, session_id: str, created_at: int) -> SessionRecord:
        record = SessionRecord(
            id=session_id,
            project_uuid=str(self.paths.project_uuid),
            created_at=created_at,
        )
        self.metadata.insert_session(record)
        return record

    def create_thread(
        self,
        *,
        session_id: str,
        thread_id: str,
        kind: str,
        created_at: int,
        summary: str | None = None,
        parent_task_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> ThreadRecord:
        thread_dir = self.paths.threads_dir / thread_id / "messages"
        thread_dir.mkdir(parents=True, exist_ok=True)
        record = ThreadRecord(
            id=thread_id,
            session_id=session_id,
            kind=kind,
            created_at=created_at,
            updated_at=created_at,
            summary=summary,
            parent_task_id=parent_task_id,
            parent_thread_id=parent_thread_id,
        )
        self.metadata.insert_thread(record)
        return record

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
        message_dir = self.paths.threads_dir / thread_id / "messages"
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
        return record

    def list_messages(self, thread_id: str) -> list[MessageRecord]:
        return self.metadata.list_messages(thread_id)

    def read_message_body(self, message: MessageRecord) -> str:
        return Path(message.body_path).read_text(encoding="utf-8")


def _ensure_trailing_newline(body: str) -> str:
    return body if body.endswith("\n") else f"{body}\n"
