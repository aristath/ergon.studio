from __future__ import annotations

from dataclasses import dataclass, replace


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


class LiveRuntimeState:
    def __init__(self) -> None:
        self._drafts: dict[str, LiveMessageDraft] = {}

    def list_drafts(self) -> tuple[LiveMessageDraft, ...]:
        return tuple(sorted(self._drafts.values(), key=lambda draft: (draft.created_at, draft.draft_id)))

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
        return LiveRuntimeEvent(
            kind="message_started",
            draft_id=draft_id,
            thread_id=thread_id,
            sender=sender,
            created_at=created_at,
            body="",
        )

    def append_delta(self, *, draft_id: str, delta: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.get(draft_id)
        if draft is None:
            return None
        updated = replace(draft, body=draft.body + delta)
        self._drafts[draft_id] = updated
        return LiveRuntimeEvent(
            kind="message_delta",
            draft_id=draft_id,
            thread_id=updated.thread_id,
            sender=updated.sender,
            created_at=created_at,
            body=updated.body,
            delta=delta,
        )

    def complete_draft(self, *, draft_id: str, message_id: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.pop(draft_id, None)
        if draft is None:
            return None
        return LiveRuntimeEvent(
            kind="message_completed",
            draft_id=draft_id,
            thread_id=draft.thread_id,
            sender=draft.sender,
            created_at=created_at,
            body=draft.body,
            message_id=message_id,
        )

    def fail_draft(self, *, draft_id: str, error: str, created_at: int) -> LiveRuntimeEvent | None:
        draft = self._drafts.pop(draft_id, None)
        if draft is None:
            return None
        return LiveRuntimeEvent(
            kind="message_failed",
            draft_id=draft_id,
            thread_id=draft.thread_id,
            sender=draft.sender,
            created_at=created_at,
            body=draft.body,
            error=error,
        )
