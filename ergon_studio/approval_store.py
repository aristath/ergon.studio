from __future__ import annotations

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import ApprovalRecord, SessionRecord
from ergon_studio.storage.sqlite import MetadataStore


class ApprovalStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def request_approval(
        self,
        *,
        session_id: str,
        approval_id: str,
        requester: str,
        action: str,
        risk_class: str,
        reason: str,
        created_at: int,
    ) -> ApprovalRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    created_at=created_at,
                )
            )
        record = ApprovalRecord(
            id=approval_id,
            session_id=session_id,
            requester=requester,
            action=action,
            risk_class=risk_class,
            reason=reason,
            status="pending",
            created_at=created_at,
        )
        self.metadata.insert_approval(record)
        return record

    def list_approvals(self, session_id: str) -> list[ApprovalRecord]:
        return self.metadata.list_approvals(session_id)
