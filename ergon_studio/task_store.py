from __future__ import annotations

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import SessionRecord, TaskRecord
from ergon_studio.storage.sqlite import MetadataStore


class TaskStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_task(
        self,
        *,
        session_id: str,
        task_id: str,
        title: str,
        state: str,
        created_at: int,
        parent_task_id: str | None = None,
    ) -> TaskRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    created_at=created_at,
                )
            )
        record = TaskRecord(
            id=task_id,
            session_id=session_id,
            title=title,
            state=state,
            created_at=created_at,
            updated_at=created_at,
            parent_task_id=parent_task_id,
        )
        self.metadata.insert_task(record)
        return record

    def list_tasks(self, session_id: str) -> list[TaskRecord]:
        return self.metadata.list_tasks(session_id)
