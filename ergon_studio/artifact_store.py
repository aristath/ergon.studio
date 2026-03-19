from __future__ import annotations

from pathlib import Path

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import ArtifactRecord, SessionRecord
from ergon_studio.storage.sqlite import MetadataStore


class ArtifactStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def create_artifact(
        self,
        *,
        session_id: str,
        artifact_id: str,
        kind: str,
        title: str,
        content: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> ArtifactRecord:
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

        file_path = self.paths.session_artifacts_dir(session_id) / f"{artifact_id}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(_ensure_trailing_newline(content), encoding="utf-8")

        record = ArtifactRecord(
            id=artifact_id,
            session_id=session_id,
            kind=kind,
            title=title,
            file_path=file_path,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        self.metadata.insert_artifact(record)
        self.metadata.touch_session(session_id, updated_at=created_at)
        return record

    def list_artifacts(self, session_id: str) -> list[ArtifactRecord]:
        return self.metadata.list_artifacts(session_id)

    def list_all_artifacts(self) -> list[ArtifactRecord]:
        return self.metadata.list_all_artifacts()

    def read_artifact_body(self, artifact: ArtifactRecord) -> str:
        return Path(artifact.file_path).read_text(encoding="utf-8")


def _ensure_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else f"{content}\n"
