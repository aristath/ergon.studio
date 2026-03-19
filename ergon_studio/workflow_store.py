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
        current_step_index: int = 0,
        last_thread_id: str | None = None,
    ) -> WorkflowRunRecord:
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
        record = WorkflowRunRecord(
            id=workflow_run_id,
            session_id=session_id,
            workflow_id=workflow_id,
            state=state,
            created_at=created_at,
            updated_at=created_at,
            root_task_id=root_task_id,
            current_step_index=current_step_index,
            last_thread_id=last_thread_id,
        )
        self.metadata.insert_workflow_run(record)
        self.metadata.touch_session(session_id, updated_at=created_at)
        return record

    def get_workflow_run(self, workflow_run_id: str) -> WorkflowRunRecord | None:
        return self.metadata.get_workflow_run(workflow_run_id)

    def update_workflow_run(self, record: WorkflowRunRecord) -> None:
        self.metadata.update_workflow_run(record)
        self.metadata.touch_session(record.session_id, updated_at=record.updated_at)

    def list_workflow_runs(self, session_id: str) -> list[WorkflowRunRecord]:
        return self.metadata.list_workflow_runs(session_id)
