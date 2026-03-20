from __future__ import annotations

import asyncio
import unittest

from ergon_studio.runtime_events import RuntimeEventStream
from ergon_studio.storage.models import EventRecord


class RuntimeEventStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscribers_receive_published_runtime_events(self) -> None:
        stream = RuntimeEventStream()
        subscription = stream.subscribe()

        event = EventRecord(
            id="event-1",
            session_id="session-1",
            kind="workflow_started",
            summary="Workflow started",
            created_at=10,
            thread_id="thread-1",
            task_id="task-1",
        )
        stream.publish(event)

        received = await asyncio.wait_for(subscription.__anext__(), timeout=0.1)
        self.assertEqual(received, event)
        subscription.close()
