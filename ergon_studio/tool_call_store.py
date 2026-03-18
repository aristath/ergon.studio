from __future__ import annotations

import json
from pathlib import Path

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import SessionRecord, ToolCallRecord
from ergon_studio.storage.sqlite import MetadataStore


class ToolCallStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def record_tool_call(
        self,
        *,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: object,
        result: object | None,
        status: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        error_message: str | None = None,
    ) -> ToolCallRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    created_at=created_at,
                )
            )

        tool_dir = self.paths.logs_dir / "tool_calls"
        tool_dir.mkdir(parents=True, exist_ok=True)
        request_path = tool_dir / f"{tool_call_id}-request.json"
        request_path.write_text(
            json.dumps(arguments, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        response_path: Path | None = None
        if result is not None:
            response_path = tool_dir / f"{tool_call_id}-response.json"
            response_path.write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

        record = ToolCallRecord(
            id=tool_call_id,
            session_id=session_id,
            tool_name=tool_name,
            status=status,
            request_path=request_path,
            response_path=response_path,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
            agent_id=agent_id,
            error_message=error_message,
        )
        self.metadata.insert_tool_call(record)
        return record

    def list_tool_calls(self, session_id: str) -> list[ToolCallRecord]:
        return self.metadata.list_tool_calls(session_id)

    def read_request(self, record: ToolCallRecord) -> str:
        return record.request_path.read_text(encoding="utf-8")

    def read_response(self, record: ToolCallRecord) -> str:
        if record.response_path is None:
            return ""
        return record.response_path.read_text(encoding="utf-8")
