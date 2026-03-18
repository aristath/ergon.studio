from __future__ import annotations

from ergon_studio.paths import StudioPaths
from ergon_studio.storage.models import MemoryFactRecord
from ergon_studio.storage.sqlite import MetadataStore


class MemoryStore:
    def __init__(self, paths: StudioPaths) -> None:
        self.paths = paths
        self.metadata = MetadataStore(paths.state_db_path)

    def add_fact(
        self,
        *,
        fact_id: str,
        scope: str,
        kind: str,
        content: str,
        created_at: int,
    ) -> MemoryFactRecord:
        record = MemoryFactRecord(
            id=fact_id,
            scope=scope,
            kind=kind,
            content=content,
            created_at=created_at,
        )
        self.metadata.insert_memory_fact(record)
        return record

    def list_facts(self) -> list[MemoryFactRecord]:
        return self.metadata.list_memory_facts()
