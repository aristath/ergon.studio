from __future__ import annotations

import time
import unittest
from dataclasses import replace

from ergon_studio.proxy.pending_store import PendingStore


class PendingStoreTests(unittest.TestCase):
    def test_record_has_created_at(self) -> None:
        store = PendingStore()
        record = store.create(session_id="s1", actor="orchestrator", tool_calls=())
        self.assertIsInstance(record.created_at, float)

    def test_sweep_discards_records_older_than_ttl(self) -> None:
        store = PendingStore()
        record = store.create(session_id="s1", actor="orchestrator", tool_calls=())
        store._records[record.pending_id] = replace(
            record, created_at=time.monotonic() - 700
        )
        store.sweep(max_age_seconds=600.0)
        self.assertIsNone(store.get(record.pending_id))

    def test_sweep_keeps_recent_records(self) -> None:
        store = PendingStore()
        record = store.create(session_id="s1", actor="orchestrator", tool_calls=())
        store.sweep(max_age_seconds=600.0)
        self.assertIsNotNone(store.get(record.pending_id))

    def test_create_triggers_sweep_of_old_records(self) -> None:
        store = PendingStore()
        record = store.create(session_id="s1", actor="orchestrator", tool_calls=())
        store._records[record.pending_id] = replace(
            record, created_at=time.monotonic() - 700
        )
        store.create(session_id="s1", actor="orchestrator", tool_calls=())
        self.assertIsNone(store.get(record.pending_id))
