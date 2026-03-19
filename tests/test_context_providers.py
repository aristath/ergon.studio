from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_framework import Message

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.context_providers import RetrievalContextProvider, WORKSPACE_STATE_KEY
from ergon_studio.event_store import EventStore
from ergon_studio.retrieval import RetrievalResult


class _FakeRetrievalIndex:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def query(self, text: str, *, limit: int = 5) -> list[RetrievalResult]:
        self.queries.append((text, limit))
        return self.results[:limit]


class _FakeContext:
    def __init__(self, input_messages: list[Message]) -> None:
        self.input_messages = input_messages
        self.instructions: list[tuple[str, str]] = []

    def extend_instructions(self, source_id: str, text: str) -> None:
        self.instructions.append((source_id, text))


class RetrievalContextProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_provider_injects_relevant_workspace_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            event_store = EventStore(paths)
            retrieval_index = _FakeRetrievalIndex(
                [
                    RetrievalResult(
                        path="index.php",
                        chunk_id="chunk-1",
                        text="<?php echo 'hello from php';",
                        score=0.9,
                        start_line=1,
                        end_line=1,
                    )
                ]
            )
            provider = RetrievalContextProvider(retrieval_index, event_store)
            context = _FakeContext(
                [
                    Message(
                        role="user",
                        text="What does the PHP file say?",
                        author_name="user",
                        message_id="message-1",
                    )
                ]
            )
            session = SimpleNamespace(
                state={
                    WORKSPACE_STATE_KEY: {
                        "session_id": "session-main",
                        "thread_id": "thread-main",
                        "agent_id": "orchestrator",
                        "created_at": 10,
                    }
                }
            )

            await provider.before_run(
                agent=None,
                session=session,
                context=context,
                state=None,
            )

            self.assertEqual(retrieval_index.queries, [("What does the PHP file say?", 4)])
            self.assertEqual(len(context.instructions), 1)
            self.assertIn("index.php:1-1", context.instructions[0][1])
            self.assertIn("hello from php", context.instructions[0][1])
            self.assertEqual(
                [event.kind for event in event_store.list_events("session-main")],
                ["retrieval_loaded"],
            )
