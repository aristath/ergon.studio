from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ergon_studio.live_runtime import LiveMessageDraft
from ergon_studio.runtime import RuntimeContext
from ergon_studio.storage.models import ApprovalRecord, MessageRecord, ThreadRecord
from ergon_studio.tui.timeline_models import ApprovalItem, ChatTurnItem, NoticeItem, TimelineItem, TimelineThreadMessage, WorkroomSegmentItem


@dataclass(frozen=True)
class _TimelineEntry:
    kind: str
    created_at: int
    sort_id: str
    payload: object


def build_session_timeline(
    runtime: RuntimeContext,
    *,
    notices: Sequence[NoticeItem] = (),
    hidden_main_message_ids: Iterable[str] = (),
) -> tuple[TimelineItem, ...]:
    hidden_ids = set(hidden_main_message_ids)
    entries: list[_TimelineEntry] = []

    main_messages = [
        message
        for message in runtime.list_main_messages()
        if message.id not in hidden_ids
    ]
    for message in main_messages:
        entries.append(
            _TimelineEntry(
                kind="main_message",
                created_at=message.created_at,
                sort_id=message.id,
                payload=message,
            )
        )

    for thread in runtime.list_threads():
        if thread.id == runtime.main_thread_id:
            continue
        for message in runtime.list_thread_messages(thread.id):
            entries.append(
                _TimelineEntry(
                    kind="thread_message",
                    created_at=message.created_at,
                    sort_id=message.id,
                    payload=(thread, message),
                )
            )

    for approval in runtime.list_pending_approvals():
        entries.append(
            _TimelineEntry(
                kind="approval",
                created_at=approval.created_at,
                sort_id=approval.id,
                payload=approval,
            )
        )

    for draft in runtime.list_live_message_drafts():
        entries.append(
            _TimelineEntry(
                kind="live_draft",
                created_at=draft.created_at,
                sort_id=draft.draft_id,
                payload=draft,
            )
        )

    for notice in notices:
        entries.append(
            _TimelineEntry(
                kind="notice",
                created_at=notice.created_at,
                sort_id=notice.item_id,
                payload=notice,
            )
        )

    entries.sort(key=lambda entry: (entry.created_at, _kind_priority(entry.kind), entry.sort_id))

    items: list[TimelineItem] = []
    active_thread: ThreadRecord | None = None
    active_messages: list[TimelineThreadMessage] = []

    def flush_segment() -> None:
        nonlocal active_thread, active_messages
        if active_thread is None or not active_messages:
            active_thread = None
            active_messages = []
            return
        items.append(
            WorkroomSegmentItem(
                item_id=f"segment-{active_thread.id}-{active_messages[0].message_id}",
                thread_id=active_thread.id,
                thread_kind=active_thread.kind,
                title=_thread_timeline_title(active_thread),
                assigned_agent_id=active_thread.assigned_agent_id,
                parent_task_id=active_thread.parent_task_id,
                parent_thread_id=active_thread.parent_thread_id,
                created_at=active_messages[0].created_at,
                messages=tuple(active_messages),
            )
        )
        active_thread = None
        active_messages = []

    for entry in entries:
        if entry.kind == "thread_message":
            thread, message = entry.payload
            assert isinstance(thread, ThreadRecord)
            assert isinstance(message, MessageRecord)
            if active_thread is None or active_thread.id != thread.id:
                flush_segment()
                active_thread = thread
            active_messages.append(_timeline_thread_message(runtime, message))
            continue
        if entry.kind == "live_draft":
            draft = entry.payload
            assert isinstance(draft, LiveMessageDraft)
            if draft.thread_id == runtime.main_thread_id:
                flush_segment()
                items.append(_live_chat_turn_item(draft))
                continue
            thread = runtime.get_thread(draft.thread_id)
            if thread is None:
                continue
            if active_thread is None or active_thread.id != thread.id:
                flush_segment()
                active_thread = thread
            active_messages.append(_live_timeline_thread_message(draft))
            continue

        flush_segment()
        if entry.kind == "main_message":
            message = entry.payload
            assert isinstance(message, MessageRecord)
            items.append(_chat_turn_item(runtime, message))
            continue
        if entry.kind == "approval":
            approval = entry.payload
            assert isinstance(approval, ApprovalRecord)
            items.append(_approval_item(approval))
            continue
        if entry.kind == "notice":
            notice = entry.payload
            assert isinstance(notice, NoticeItem)
            items.append(notice)
            continue
        raise ValueError(f"unknown timeline entry kind: {entry.kind}")

    flush_segment()
    return tuple(items)


def _chat_turn_item(runtime: RuntimeContext, message: MessageRecord) -> ChatTurnItem:
    body = runtime.conversation_store.read_message_body(message).rstrip("\n")
    return ChatTurnItem(
        item_id=f"chat-{message.id}",
        message_id=message.id,
        sender=message.sender,
        kind=message.kind,
        body=body,
        created_at=message.created_at,
        is_live=False,
    )


def _timeline_thread_message(runtime: RuntimeContext, message: MessageRecord) -> TimelineThreadMessage:
    body = runtime.conversation_store.read_message_body(message).rstrip("\n")
    return TimelineThreadMessage(
        message_id=message.id,
        sender=message.sender,
        kind=message.kind,
        body=body,
        created_at=message.created_at,
        is_live=False,
    )


def _live_chat_turn_item(draft: LiveMessageDraft) -> ChatTurnItem:
    return ChatTurnItem(
        item_id=f"live-chat-{draft.draft_id}",
        message_id=draft.draft_id,
        sender=draft.sender,
        kind=draft.kind,
        body=draft.body,
        created_at=draft.created_at,
        is_live=True,
    )


def _live_timeline_thread_message(draft: LiveMessageDraft) -> TimelineThreadMessage:
    return TimelineThreadMessage(
        message_id=draft.draft_id,
        sender=draft.sender,
        kind=draft.kind,
        body=draft.body,
        created_at=draft.created_at,
        is_live=True,
    )


def _approval_item(approval: ApprovalRecord) -> ApprovalItem:
    return ApprovalItem(
        item_id=f"approval-{approval.id}",
        approval_id=approval.id,
        requester=approval.requester,
        action=approval.action,
        risk_class=approval.risk_class,
        reason=approval.reason,
        status=approval.status,
        created_at=approval.created_at,
    )


def _thread_timeline_title(thread: ThreadRecord) -> str:
    summary = (thread.summary or thread.kind).strip()
    if thread.kind == "agent_direct" and thread.assigned_agent_id:
        return f"Orchestrator <-> {thread.assigned_agent_id}"
    if thread.kind == "group_workroom":
        return f"Workroom: {summary}"
    if thread.kind == "review":
        return f"Review: {summary}"
    if thread.assigned_agent_id:
        return f"{thread.assigned_agent_id}: {summary}"
    return summary


def _kind_priority(kind: str) -> int:
    order = {
        "main_message": 0,
        "thread_message": 1,
        "live_draft": 2,
        "approval": 3,
        "notice": 4,
    }
    return order.get(kind, 99)
