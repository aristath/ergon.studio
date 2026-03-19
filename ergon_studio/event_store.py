from __future__ import annotations

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import EventRecord, SessionRecord
from ergon_studio.storage.sqlite import MetadataStore


class EventStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def append_event(
        self,
        *,
        session_id: str,
        event_id: str,
        kind: str,
        summary: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> EventRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    title=session_id,
                    created_at=created_at,
                    updated_at=created_at,
                    archived_at=None,
                )
            )
        record = EventRecord(
            id=event_id,
            session_id=session_id,
            kind=kind,
            summary=summary,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        self.metadata.insert_event(record)
        self.metadata.touch_session(session_id, updated_at=created_at)
        return record

    def list_events(self, session_id: str) -> list[EventRecord]:
        return self.metadata.list_events(session_id)
