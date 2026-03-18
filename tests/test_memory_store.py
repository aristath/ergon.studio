from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.memory_store import MemoryStore
from ergon_studio.storage.sqlite import initialize_database


class MemoryStoreTests(unittest.TestCase):
    def test_add_and_list_memory_facts_persist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = MemoryStore(paths)

            first = store.add_fact(
                fact_id="fact-1",
                scope="project",
                kind="decision",
                content="Use Textual for the TUI.",
                created_at=10,
                source="plan",
                confidence=0.9,
                tags=("ui", "decision"),
            )
            second = store.add_fact(
                fact_id="fact-2",
                scope="project",
                kind="preference",
                content="Use Unix time ints everywhere.",
                created_at=20,
            )

            self.assertEqual(first.id, "fact-1")
            self.assertEqual(second.kind, "preference")
            self.assertEqual(first.source, "plan")
            self.assertEqual(first.tags, ("ui", "decision"))
            self.assertEqual(
                [fact.id for fact in store.list_facts()],
                ["fact-1", "fact-2"],
            )
            store.touch_facts(("fact-1",), last_used_at=30)
            touched = store.list_facts()[0]
            self.assertEqual(touched.last_used_at, 30)
