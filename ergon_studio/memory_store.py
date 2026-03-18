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
        source: str | None = None,
        confidence: float | None = None,
        tags: tuple[str, ...] = (),
        last_used_at: int | None = None,
    ) -> MemoryFactRecord:
        record = MemoryFactRecord(
            id=fact_id,
            scope=scope,
            kind=kind,
            content=content,
            created_at=created_at,
            source=source,
            confidence=confidence,
            tags=tags,
            last_used_at=last_used_at,
        )
        self.metadata.insert_memory_fact(record)
        return record

    def list_facts(self, *, scopes: tuple[str, ...] | None = None) -> list[MemoryFactRecord]:
        facts = self.metadata.list_memory_facts()
        if scopes is None:
            return facts
        return [fact for fact in facts if fact.scope in scopes]

    def touch_facts(self, fact_ids: tuple[str, ...], *, last_used_at: int) -> None:
        facts_by_id = {fact.id: fact for fact in self.metadata.list_memory_facts()}
        for fact_id in fact_ids:
            record = facts_by_id.get(fact_id)
            if record is None:
                continue
            self.metadata.update_memory_fact(
                MemoryFactRecord(
                    id=record.id,
                    scope=record.scope,
                    kind=record.kind,
                    content=record.content,
                    created_at=record.created_at,
                    source=record.source,
                    confidence=record.confidence,
                    tags=record.tags,
                    last_used_at=last_used_at,
                )
            )
