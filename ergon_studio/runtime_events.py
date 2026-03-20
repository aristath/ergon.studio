from __future__ import annotations

import asyncio
from uuid import uuid4

from ergon_studio.storage.models import EventRecord


class RuntimeEventSubscription:
    def __init__(
        self,
        *,
        queue: asyncio.Queue[EventRecord | None],
        unsubscribe,
    ) -> None:
        self._queue = queue
        self._unsubscribe = unsubscribe
        self._closed = False

    def __aiter__(self) -> RuntimeEventSubscription:
        return self

    async def __anext__(self) -> EventRecord:
        event = await self._queue.get()
        if event is None:
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        self._queue.put_nowait(None)


class RuntimeEventStream:
    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[EventRecord | None]] = {}

    def subscribe(self) -> RuntimeEventSubscription:
        token = f"runtime-event-subscription-{uuid4().hex}"
        queue: asyncio.Queue[EventRecord | None] = asyncio.Queue()
        self._subscribers[token] = queue
        return RuntimeEventSubscription(
            queue=queue,
            unsubscribe=lambda: self._unsubscribe(token),
        )

    def publish(self, event: EventRecord) -> None:
        for queue in tuple(self._subscribers.values()):
            queue.put_nowait(event)

    def _unsubscribe(self, token: str) -> None:
        self._subscribers.pop(token, None)
