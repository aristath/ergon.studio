from __future__ import annotations

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import SessionRecord, WorkflowRunRecord
from ergon_studio.storage.sqlite import MetadataStore


class WorkflowStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_workflow_run(
        self,
        *,
        session_id: str,
        workflow_run_id: str,
        workflow_id: str,
        state: str,
        created_at: int,
        root_task_id: str | None = None,
    ) -> WorkflowRunRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    created_at=created_at,
                )
            )
        record = WorkflowRunRecord(
            id=workflow_run_id,
            session_id=session_id,
            workflow_id=workflow_id,
            state=state,
            created_at=created_at,
            updated_at=created_at,
            root_task_id=root_task_id,
        )
        self.metadata.insert_workflow_run(record)
        return record

    def list_workflow_runs(self, session_id: str) -> list[WorkflowRunRecord]:
        return self.metadata.list_workflow_runs(session_id)
