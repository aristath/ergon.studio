from __future__ import annotations

from uuid import uuid4

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import SessionRecord
from ergon_studio.storage.sqlite import MetadataStore


class SessionStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str | None = None,
        created_at: int,
    ) -> SessionRecord:
        resolved_session_id = session_id or f"session-{uuid4().hex[:8]}"
        record = SessionRecord(
            id=resolved_session_id,
            project_uuid=str(self.paths.project_uuid),
            title=(title or default_session_title(session_id=resolved_session_id, created_at=created_at)).strip(),
            created_at=created_at,
            updated_at=created_at,
            archived_at=None,
        )
        self.metadata.insert_session(record)
        return record

    def ensure_session(
        self,
        *,
        session_id: str,
        created_at: int,
        title: str | None = None,
    ) -> SessionRecord:
        existing = self.metadata.get_session(session_id)
        if existing is not None:
            return existing
        return self.create_session(
            session_id=session_id,
            title=title,
            created_at=created_at,
        )

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.metadata.get_session(session_id)

    def list_sessions(self, *, include_archived: bool = False) -> list[SessionRecord]:
        return self.metadata.list_sessions(
            str(self.paths.project_uuid),
            include_archived=include_archived,
        )

    def latest_session(self, *, include_archived: bool = False) -> SessionRecord | None:
        sessions = self.list_sessions(include_archived=include_archived)
        return sessions[0] if sessions else None

    def rename_session(self, *, session_id: str, title: str, updated_at: int) -> SessionRecord:
        existing = self.metadata.get_session(session_id)
        if existing is None:
            raise ValueError(f"unknown session: {session_id}")
        updated = SessionRecord(
            id=existing.id,
            project_uuid=existing.project_uuid,
            title=title.strip() or existing.title,
            created_at=existing.created_at,
            updated_at=updated_at,
            archived_at=existing.archived_at,
        )
        self.metadata.update_session(updated)
        return updated

    def archive_session(self, *, session_id: str, archived_at: int) -> SessionRecord:
        existing = self.metadata.get_session(session_id)
        if existing is None:
            raise ValueError(f"unknown session: {session_id}")
        updated = SessionRecord(
            id=existing.id,
            project_uuid=existing.project_uuid,
            title=existing.title,
            created_at=existing.created_at,
            updated_at=archived_at,
            archived_at=archived_at,
        )
        self.metadata.update_session(updated)
        return updated

    def touch_session(self, *, session_id: str, updated_at: int) -> None:
        self.metadata.touch_session(session_id, updated_at=updated_at)


def default_session_title(*, session_id: str, created_at: int) -> str:
    if session_id.startswith("session-"):
        suffix = session_id.removeprefix("session-")
    else:
        suffix = session_id
    suffix = suffix.strip() or str(created_at)
    return f"Session {suffix}"
