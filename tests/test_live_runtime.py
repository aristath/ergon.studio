from __future__ import annotations

import asyncio
import unittest

from ergon_studio.live_runtime import LiveRuntimeState


class LiveRuntimeStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscribers_receive_draft_lifecycle_events(self) -> None:
        state = LiveRuntimeState()
        subscription = state.subscribe()

        started = state.start_draft(
            draft_id="draft-1",
            thread_id="thread-1",
            sender="orchestrator",
            kind="chat",
            created_at=10,
        )
        updated = state.append_delta(
            draft_id="draft-1",
            delta="hello",
            created_at=11,
        )
        completed = state.complete_draft(
            draft_id="draft-1",
            message_id="message-1",
            created_at=12,
        )

        events = [
            await asyncio.wait_for(subscription.__anext__(), timeout=0.1),
            await asyncio.wait_for(subscription.__anext__(), timeout=0.1),
            await asyncio.wait_for(subscription.__anext__(), timeout=0.1),
        ]

        self.assertEqual(events, [started, updated, completed])
        subscription.close()

