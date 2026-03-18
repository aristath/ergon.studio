from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from uuid import uuid4

from agent_framework import Message

from ergon_studio.agent_session_store import AgentSessionStore
from ergon_studio.approval_store import ApprovalStore
from ergon_studio.agent_factory import build_agent
from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.config import save_global_config_text
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.definitions import DefinitionDocument, save_definition_text
from ergon_studio.event_store import EventStore
from ergon_studio.memory_store import MemoryStore
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.storage.models import ApprovalRecord, ArtifactRecord, EventRecord, MemoryFactRecord, MessageRecord, TaskRecord, ThreadRecord
from ergon_studio.task_store import TaskStore
from ergon_studio.tool_registry import build_workspace_tool_registry


MAIN_SESSION_ID = "session-main"
MAIN_THREAD_ID = "thread-main"


@dataclass(frozen=True)
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]
    agent_session_store: AgentSessionStore
    conversation_store: ConversationStore
    task_store: TaskStore
    event_store: EventStore
    approval_store: ApprovalStore
    memory_store: MemoryStore
    artifact_store: ArtifactStore
    main_session_id: str = MAIN_SESSION_ID
    main_thread_id: str = MAIN_THREAD_ID

    def build_agent(self, agent_id: str):
        return build_agent(self.registry, agent_id, tool_registry=self.tool_registry)

    def reload_registry(self) -> None:
        object.__setattr__(self, "registry", load_registry(self.paths))

    def can_build_agent(self, agent_id: str) -> bool:
        try:
            self.build_agent(agent_id)
        except (KeyError, ValueError):
            return False
        return True

    def assigned_provider_name(self, agent_id: str) -> str | None:
        definition = self.registry.agent_definitions.get(agent_id)
        if definition is None:
            return None

        role = str(definition.metadata.get("role", definition.id))
        role_assignments = self.registry.config.get("role_assignments", {})
        provider_name = role_assignments.get(role) or role_assignments.get(agent_id)
        if not provider_name:
            return None
        if provider_name not in self.registry.config.get("providers", {}):
            return None
        return provider_name

    def agent_status_summary(self, agent_id: str) -> str:
        provider_name = self.assigned_provider_name(agent_id)
        if provider_name is None:
            return "not configured"

        provider = self.registry.config["providers"][provider_name]
        model_name = provider.get("model", "unknown-model")
        return f"ready via {provider_name} ({model_name})"

    def read_agent_definition_text(self, agent_id: str) -> str:
        definition = self.registry.agent_definitions[agent_id]
        return definition.path.read_text(encoding="utf-8")

    def read_global_config_text(self) -> str:
        return self.paths.config_path.read_text(encoding="utf-8")

    def save_agent_definition_text(self, *, agent_id: str, text: str, created_at: int | None = None) -> DefinitionDocument:
        definition = self.registry.agent_definitions[agent_id]
        saved = save_definition_text(definition.path, text)
        self.reload_registry()
        if created_at is None:
            created_at = int(time.time())
        self.append_event(
            kind="definition_saved",
            summary=f"Saved agent definition {agent_id}",
            created_at=created_at,
        )
        return saved

    def save_global_config_text(self, *, text: str, created_at: int | None = None) -> dict[str, object]:
        saved = save_global_config_text(self.paths.config_path, text)
        self.reload_registry()
        if created_at is None:
            created_at = int(time.time())
        self.append_event(
            kind="config_saved",
            summary="Saved global config",
            created_at=created_at,
        )
        return saved

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

    def list_memory_facts(self) -> list[MemoryFactRecord]:
        return self.memory_store.list_facts()

    def list_artifacts(self) -> list[ArtifactRecord]:
        return self.artifact_store.list_artifacts(self.main_session_id)

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

    async def send_user_message_to_orchestrator(
        self,
        *,
        body: str,
        created_at: int,
    ) -> tuple[MessageRecord, MessageRecord | None]:
        user_message = self.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="user",
            kind="chat",
            body=body,
            created_at=created_at,
        )

        try:
            agent = self.build_agent("orchestrator")
        except (KeyError, ValueError) as exc:
            self.append_event(
                kind="agent_unavailable",
                summary=f"Orchestrator unavailable: {exc}",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
            return user_message, None

        session = self.agent_session_store.load_or_create_session(
            thread_id=self.main_thread_id,
            agent_id="orchestrator",
            session_factory=lambda session_id: agent.create_session(session_id=session_id),
        )

        try:
            response = await agent.run(
                [
                    Message(
                        role="user",
                        text=body,
                        author_name="user",
                    )
                ],
                session=session,
            )
        except Exception as exc:
            self.append_event(
                kind="agent_failed",
                summary=f"Orchestrator run failed: {type(exc).__name__}: {exc}",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
            self.agent_session_store.save_session(
                thread_id=self.main_thread_id,
                agent_id="orchestrator",
                session=session,
            )
            return user_message, None

        self.agent_session_store.save_session(
            thread_id=self.main_thread_id,
            agent_id="orchestrator",
            session=session,
        )
        response_text = response.text.strip()
        if not response_text:
            return user_message, None

        reply_message = self.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="orchestrator",
            kind="chat",
            body=response_text,
            created_at=created_at + 1,
        )
        return user_message, reply_message

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
        approval = self.approval_store.request_approval(
            session_id=self.main_session_id,
            approval_id=approval_id,
            requester=requester,
            action=action,
            risk_class=risk_class,
            reason=reason,
            created_at=created_at,
        )
        self.append_event(
            kind="approval_requested",
            summary=f"{requester} requested approval for {action}",
            created_at=created_at,
        )
        return approval

    def add_memory_fact(
        self,
        *,
        fact_id: str,
        scope: str,
        kind: str,
        content: str,
        created_at: int,
    ) -> MemoryFactRecord:
        fact = self.memory_store.add_fact(
            fact_id=fact_id,
            scope=scope,
            kind=kind,
            content=content,
            created_at=created_at,
        )
        self.append_event(
            kind="memory_fact_added",
            summary=f"Added memory fact {fact_id}",
            created_at=created_at,
        )
        return fact

    def create_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        title: str,
        content: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> ArtifactRecord:
        artifact = self.artifact_store.create_artifact(
            session_id=self.main_session_id,
            artifact_id=artifact_id,
            kind=kind,
            title=title,
            content=content,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        self.append_event(
            kind="artifact_created",
            summary=f"Created artifact {artifact_id}",
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        return artifact

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
    agent_session_store = AgentSessionStore(paths)
    conversation_store = ConversationStore(paths)
    task_store = TaskStore(paths)
    event_store = EventStore(paths)
    approval_store = ApprovalStore(paths)
    memory_store = MemoryStore(paths)
    artifact_store = ArtifactStore(paths)
    runtime = RuntimeContext(
        paths=paths,
        registry=registry,
        tool_registry=tool_registry,
        agent_session_store=agent_session_store,
        conversation_store=conversation_store,
        task_store=task_store,
        event_store=event_store,
        approval_store=approval_store,
        memory_store=memory_store,
        artifact_store=artifact_store,
    )
    runtime.ensure_main_conversation()
    return runtime
