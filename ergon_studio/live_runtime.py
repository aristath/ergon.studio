from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4


@dataclass(frozen=True)
class LiveMessageDraft:
    draft_id: str
    thread_id: str
    sender: str
    kind: str
    body: str
    created_at: int


@dataclass(frozen=True)
class LiveRuntimeEvent:
    kind: str
    draft_id: str
    thread_id: str
    sender: str
    created_at: int
    body: str = ""
    delta: str = ""
    message_id: str | None = None
    error: str | None = None


class LiveRuntimeSubscription:
    def __init__(
        self,
        *,
        queue: asyncio.Queue[LiveRuntimeEvent | None],
        unsubscribe,
    ) -> None:
        self._queue = queue
        self._unsubscribe = unsubscribe
        self._closed = False

    def __aiter__(self) -> LiveRuntimeSubscription:
        return self

    async def __anext__(self) -> LiveRuntimeEvent:
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


class LiveRuntimeState:
    def __init__(self) -> None:
        self._drafts: dict[str, LiveMessageDraft] = {}
        self._subscribers: dict[str, asyncio.Queue[LiveRuntimeEvent | None]] = {}

    def list_drafts(self) -> tuple[LiveMessageDraft, ...]:
        return tuple(sorted(self._drafts.values(), key=lambda draft: (draft.created_at, draft.draft_id)))

    def subscribe(self) -> LiveRuntimeSubscription:
        token = f"subscription-{uuid4().hex}"
        queue: asyncio.Queue[LiveRuntimeEvent | None] = asyncio.Queue()
        self._subscribers[token] = queue
        return LiveRuntimeSubscription(
            queue=queue,
            unsubscribe=lambda: self._unsubscribe(token),
        )

    def start_draft(
        self,
        *,
        draft_id: str,
        thread_id: str,
        sender: str,
        kind: str,
        created_at: int,
    ) -> LiveRuntimeEvent:
        draft = LiveMessageDraft(
            draft_id=draft_id,
            thread_id=thread_id,
            sender=sender,
            kind=kind,
            body="",
            created_at=created_at,
        )
        self._drafts[draft_id] = draft
        event = LiveRuntimeEvent(
            kind="message_started",
            draft_id=draft_id,
            thread_id=thread_id,
            sender=sender,
            created_at=created_at,
            body="",
        )
        self._publish(event)
        return event

    def append_delta(self, *, draft_id: str, delta: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.get(draft_id)
        if draft is None:
            return None
        updated = replace(draft, body=draft.body + delta)
        self._drafts[draft_id] = updated
        event = LiveRuntimeEvent(
            kind="message_delta",
            draft_id=draft_id,
            thread_id=updated.thread_id,
            sender=updated.sender,
            created_at=created_at,
            body=updated.body,
            delta=delta,
        )
        self._publish(event)
        return event

    def complete_draft(self, *, draft_id: str, message_id: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.pop(draft_id, None)
        if draft is None:
            return None
        event = LiveRuntimeEvent(
            kind="message_completed",
            draft_id=draft_id,
            thread_id=draft.thread_id,
            sender=draft.sender,
            created_at=created_at,
            body=draft.body,
            message_id=message_id,
        )
        self._publish(event)
        return event

    def fail_draft(self, *, draft_id: str, error: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.pop(draft_id, None)
        if draft is None:
            return None
        event = LiveRuntimeEvent(
            kind="message_failed",
            draft_id=draft_id,
            thread_id=draft.thread_id,
            sender=draft.sender,
            created_at=created_at,
            body=draft.body,
            error=error,
        )
        self._publish(event)
        return event

    def _unsubscribe(self, token: str) -> None:
        self._subscribers.pop(token, None)

    def _publish(self, event: LiveRuntimeEvent) -> None:
        for queue in tuple(self._subscribers.values()):
            queue.put_nowait(event)
