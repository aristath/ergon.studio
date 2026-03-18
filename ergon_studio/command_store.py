from __future__ import annotations

from pathlib import Path

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import CommandRunRecord, SessionRecord
from ergon_studio.storage.sqlite import MetadataStore


class CommandStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_command_run(
        self,
        *,
        session_id: str,
        command_run_id: str,
        command: str,
        cwd: str,
        exit_code: int,
        status: str,
        output_content: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
    ) -> CommandRunRecord:
        if self.metadata.get_session(session_id) is None:
            self.metadata.insert_session(
                SessionRecord(
                    id=session_id,
                    project_uuid=str(self.paths.project_uuid),
                    created_at=created_at,
                )
            )

        output_path = self.paths.logs_dir / "commands" / f"{command_run_id}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_ensure_trailing_newline(output_content), encoding="utf-8")

        record = CommandRunRecord(
            id=command_run_id,
            session_id=session_id,
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            status=status,
            output_path=output_path,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
            agent_id=agent_id,
        )
        self.metadata.insert_command_run(record)
        return record

    def list_command_runs(self, session_id: str) -> list[CommandRunRecord]:
        return self.metadata.list_command_runs(session_id)

    def read_command_output(self, command_run: CommandRunRecord) -> str:
        return Path(command_run.output_path).read_text(encoding="utf-8")


def _ensure_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else f"{content}\n"
