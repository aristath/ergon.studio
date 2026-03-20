from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class TimelineThreadMessage:
    message_id: str
    sender: str
    kind: str
    body: str
    created_at: int


@dataclass(frozen=True)
class ChatTurnItem:
    item_id: str
    message_id: str
    sender: str
    kind: str
    body: str
    created_at: int


@dataclass(frozen=True)
class WorkroomSegmentItem:
    item_id: str
    thread_id: str
    thread_kind: str
    title: str
    assigned_agent_id: str | None
    parent_task_id: str | None
    parent_thread_id: str | None
    created_at: int
    messages: tuple[TimelineThreadMessage, ...]


@dataclass(frozen=True)
class ApprovalItem:
    item_id: str
    approval_id: str
    requester: str
    action: str
    risk_class: str
    reason: str
    status: str
    created_at: int


@dataclass(frozen=True)
class NoticeItem:
    item_id: str
    title: str | None
    body: str
    level: str
    created_at: int


TimelineItem: TypeAlias = ChatTurnItem | WorkroomSegmentItem | ApprovalItem | NoticeItem
