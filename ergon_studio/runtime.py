from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ergon_studio.approval_store import ApprovalStore
from ergon_studio.agent_factory import build_agent
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.event_store import EventStore
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.storage.models import ApprovalRecord, EventRecord, MessageRecord, TaskRecord, ThreadRecord
from ergon_studio.task_store import TaskStore
from ergon_studio.tool_registry import build_workspace_tool_registry


MAIN_SESSION_ID = "session-main"
MAIN_THREAD_ID = "thread-main"


@dataclass(frozen=True)
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]
    conversation_store: ConversationStore
    task_store: TaskStore
    event_store: EventStore
    approval_store: ApprovalStore
    main_session_id: str = MAIN_SESSION_ID
    main_thread_id: str = MAIN_THREAD_ID

    def build_agent(self, agent_id: str):
        return build_agent(self.registry, agent_id, tool_registry=self.tool_registry)

    def list_threads(self) -> list[ThreadRecord]:
        return self.conversation_store.list_threads(self.main_session_id)

    def list_tasks(self) -> list[TaskRecord]:
        return self.task_store.list_tasks(self.main_session_id)

    def list_events(self) -> list[EventRecord]:
        return self.event_store.list_events(self.main_session_id)

    def list_approvals(self) -> list[ApprovalRecord]:
        return self.approval_store.list_approvals(self.main_session_id)

    def list_agent_ids(self) -> list[str]:
        return sorted(self.registry.agent_definitions.keys())

    def list_workflow_ids(self) -> list[str]:
        return sorted(self.registry.workflow_definitions.keys())

    def list_provider_ids(self) -> list[str]:
        return sorted(self.registry.config.get("providers", {}).keys())

    def create_thread(
        self,
        *,
        thread_id: str,
        kind: str,
        created_at: int,
        summary: str | None = None,
        parent_task_id: str | None = None,
        parent_thread_id: str | None = None,
    ) -> ThreadRecord:
        self.ensure_main_conversation()
        thread = self.conversation_store.create_thread(
            session_id=self.main_session_id,
            thread_id=thread_id,
            kind=kind,
            created_at=created_at,
            summary=summary,
            parent_task_id=parent_task_id,
            parent_thread_id=parent_thread_id,
        )
        self.append_event(
            kind="thread_created",
            summary=f"Created thread {thread_id}",
            created_at=created_at,
            thread_id=thread_id,
            task_id=parent_task_id,
        )
        return thread

    def list_main_messages(self) -> list[MessageRecord]:
        return self.conversation_store.list_messages(self.main_thread_id)

    def list_thread_messages(self, thread_id: str) -> list[MessageRecord]:
        return self.conversation_store.list_messages(thread_id)

    def append_message_to_main_thread(
        self,
        *,
        message_id: str,
        sender: str,
        kind: str,
        body: str,
        created_at: int,
    ) -> MessageRecord:
        self.ensure_main_conversation()
        return self.append_message_to_thread(
            thread_id=self.main_thread_id,
            message_id=message_id,
            sender=sender,
            kind=kind,
            body=body,
            created_at=created_at,
        )

    def append_message_to_thread(
        self,
        *,
        thread_id: str,
        message_id: str,
        sender: str,
        kind: str,
        body: str,
        created_at: int,
    ) -> MessageRecord:
        self.ensure_main_conversation()
        message = self.conversation_store.append_message(
            thread_id=thread_id,
            message_id=message_id,
            sender=sender,
            kind=kind,
            body=body,
            created_at=created_at,
        )
        self.append_event(
            kind="message_created",
            summary=f"{sender} posted to {thread_id}",
            created_at=created_at,
            thread_id=thread_id,
        )
        return message

    def create_task(
        self,
        *,
        task_id: str,
        title: str,
        state: str,
        created_at: int,
        parent_task_id: str | None = None,
    ) -> TaskRecord:
        self.ensure_main_conversation()
        task = self.task_store.create_task(
            session_id=self.main_session_id,
            task_id=task_id,
            title=title,
            state=state,
            created_at=created_at,
            parent_task_id=parent_task_id,
        )
        self.append_event(
            kind="task_created",
            summary=f"Created task {task_id}",
            created_at=created_at,
            task_id=task_id,
        )
        return task

    def append_event(
        self,
        *,
        kind: str,
        summary: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> EventRecord:
        return self.event_store.append_event(
            session_id=self.main_session_id,
            event_id=f"event-{uuid4().hex}",
            kind=kind,
            summary=summary,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )

    def request_approval(
        self,
        *,
        approval_id: str,
        requester: str,
        action: str,
        risk_class: str,
        reason: str,
        created_at: int,
    ) -> ApprovalRecord:
        return self.approval_store.request_approval(
            session_id=self.main_session_id,
            approval_id=approval_id,
            requester=requester,
            action=action,
            risk_class=risk_class,
            reason=reason,
            created_at=created_at,
        )

    def ensure_main_conversation(self) -> None:
        self.conversation_store.ensure_session(self.main_session_id, created_at=0)
        self.conversation_store.ensure_thread(
            session_id=self.main_session_id,
            thread_id=self.main_thread_id,
            kind="main",
            created_at=0,
        )


def load_runtime(project_root: Path, home_dir: Path) -> RuntimeContext:
    paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
    registry = load_registry(paths)
    tool_registry = build_workspace_tool_registry(paths.project_root)
    conversation_store = ConversationStore(paths)
    task_store = TaskStore(paths)
    event_store = EventStore(paths)
    approval_store = ApprovalStore(paths)
    runtime = RuntimeContext(
        paths=paths,
        registry=registry,
        tool_registry=tool_registry,
        conversation_store=conversation_store,
        task_store=task_store,
        event_store=event_store,
        approval_store=approval_store,
    )
    runtime.ensure_main_conversation()
    return runtime
