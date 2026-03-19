from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        thread_id: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ApprovalRecord:
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
        payload_path = None
        if payload is not None:
            payload_path = self._write_payload(
                session_id=session_id,
                approval_id=approval_id,
                payload=payload,
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
            thread_id=thread_id,
            task_id=task_id,
            payload_path=payload_path,
        )
        self.metadata.insert_approval(record)
        self.metadata.touch_session(session_id, updated_at=created_at)
        return record

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        return self.metadata.get_approval(approval_id)

    def list_approvals(self, session_id: str) -> list[ApprovalRecord]:
        return self.metadata.list_approvals(session_id)

    def read_payload(self, approval: ApprovalRecord) -> dict[str, Any] | None:
        if approval.payload_path is None:
            return None
        return json.loads(Path(approval.payload_path).read_text(encoding="utf-8"))

    def update_approval_status(self, *, approval_id: str, status: str) -> ApprovalRecord:
        if status not in {"approved", "rejected"}:
            raise ValueError(f"unsupported approval status: {status}")
        approval = self.metadata.get_approval(approval_id)
        if approval is None:
            raise ValueError(f"unknown approval: {approval_id}")
        updated = ApprovalRecord(
            id=approval.id,
            session_id=approval.session_id,
            requester=approval.requester,
            action=approval.action,
            risk_class=approval.risk_class,
            reason=approval.reason,
            status=status,
            created_at=approval.created_at,
            thread_id=approval.thread_id,
            task_id=approval.task_id,
            payload_path=approval.payload_path,
        )
        self.metadata.update_approval(updated)
        return updated

    def _write_payload(self, *, session_id: str, approval_id: str, payload: dict[str, Any]) -> Path:
        base_dir = self.paths.session_logs_dir(session_id)
        payload_path = base_dir / "approvals" / f"{approval_id}.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload_path
