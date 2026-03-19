from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_framework import Message

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.context_providers import RetrievalContextProvider, WORKSPACE_STATE_KEY
from ergon_studio.event_store import EventStore
from ergon_studio.retrieval import RetrievalResult
from ergon_studio.whiteboard_store import WhiteboardStore


class _FakeRetrievalIndex:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[tuple[tuple[str, ...], int]] = []

    def query_many(self, texts: list[str], *, limit: int = 5) -> list[RetrievalResult]:
        self.queries.append((tuple(texts), limit))
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
            conversation_store = ConversationStore(paths)
            whiteboard_store = WhiteboardStore(paths)
            conversation_store.ensure_session("session-main", created_at=1)
            conversation_store.ensure_thread(
                session_id="session-main",
                thread_id="thread-main",
                kind="main",
                created_at=1,
            )
            conversation_store.append_message(
                thread_id="thread-main",
                message_id="message-0",
                sender="user",
                kind="chat",
                body="Build a PHP signup flow with invite code support.",
                created_at=2,
            )
            whiteboard_store.ensure_task_whiteboard(
                task_id="task-main",
                title="Signup flow",
                updated_at=3,
                goal="Build a signup flow that validates invite codes before creating users.",
            )
            retrieval_index = _FakeRetrievalIndex(
                [
                    RetrievalResult(
                        path="index.php",
                        chunk_id="chunk-1",
                        text="<?php echo 'hello from php';",
                        score=0.9,
                        start_line=1,
                        end_line=1,
                        source_type="workspace",
                    )
                ]
            )
            provider = RetrievalContextProvider(
                retrieval_index,
                event_store,
                conversation_store=conversation_store,
                whiteboard_store=whiteboard_store,
            )
            context = _FakeContext(
                [
                    Message(
                        role="user",
                        text="Do that thing we discussed.",
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
                        "task_id": "task-main",
                    }
                }
            )

            await provider.before_run(
                agent=None,
                session=session,
                context=context,
                state=None,
            )

            self.assertEqual(len(retrieval_index.queries), 1)
            query_texts, limit = retrieval_index.queries[0]
            self.assertIn("Do that thing we discussed.", query_texts)
            self.assertTrue(any("Build a PHP signup flow with invite code support." in item for item in query_texts))
            self.assertTrue(
                any(
                    "Build a signup flow that validates invite codes before creating users." in item
                    for item in query_texts
                )
            )
            self.assertGreater(limit, 4)
            self.assertEqual(len(context.instructions), 1)
            self.assertIn("index.php:1-1", context.instructions[0][1])
            self.assertIn("hello from php", context.instructions[0][1])
            self.assertEqual(
                [event.kind for event in event_store.list_events("session-main")],
                ["retrieval_loaded"],
            )
