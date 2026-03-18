from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4

from agent_framework import BaseContextProvider, Message

from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.event_store import EventStore
from ergon_studio.memory_store import MemoryStore
from ergon_studio.whiteboard_store import WhiteboardStore


WORKSPACE_STATE_KEY = "ergon_workspace"


@dataclass(frozen=True)
class WorkspaceState:
    session_id: str
    thread_id: str
    agent_id: str
    created_at: int
    task_id: str | None = None


class ConversationHistoryProvider(BaseContextProvider):
    def __init__(self, conversation_store: ConversationStore, event_store: EventStore) -> None:
        super().__init__("thread_history")
        self.conversation_store = conversation_store
        self.event_store = event_store

    async def before_run(self, *, agent, session, context, state) -> None:
        workspace_state = _workspace_state_from_session(session)
        if workspace_state is None:
            return

        records = self.conversation_store.list_messages(workspace_state.thread_id)
        messages: list[Message] = []
        for record in records:
            body = self.conversation_store.read_message_body(record).rstrip("\n")
            if not body:
                continue
            role = "assistant" if record.sender == workspace_state.agent_id else "user"
            messages.append(
                Message(
                    role=role,
                    text=body,
                    author_name=record.sender,
                    message_id=record.id,
                )
            )

        trimmed = _trim_duplicate_input(messages, context.input_messages)
        if not trimmed:
            return

        context.extend_messages(self, trimmed)
        _append_provider_event(
            event_store=self.event_store,
            workspace_state=workspace_state,
            kind="history_loaded",
            summary=f"Loaded {len(trimmed)} history messages for {workspace_state.thread_id}",
        )


class TaskWhiteboardContextProvider(BaseContextProvider):
    def __init__(self, whiteboard_store: WhiteboardStore, event_store: EventStore) -> None:
        super().__init__("task_whiteboard")
        self.whiteboard_store = whiteboard_store
        self.event_store = event_store

    async def before_run(self, *, agent, session, context, state) -> None:
        workspace_state = _workspace_state_from_session(session)
        if workspace_state is None or workspace_state.task_id is None:
            return

        whiteboard = self.whiteboard_store.read_task_whiteboard(workspace_state.task_id)
        if whiteboard is None:
            return

        context.extend_instructions(
            self.source_id,
            f"Current task whiteboard for {whiteboard.task_id}:\n\n{whiteboard.body}",
        )
        _append_provider_event(
            event_store=self.event_store,
            workspace_state=workspace_state,
            kind="whiteboard_loaded",
            summary=f"Loaded task whiteboard {whiteboard.task_id}",
        )


class ProjectMemoryContextProvider(BaseContextProvider):
    def __init__(self, memory_store: MemoryStore, event_store: EventStore) -> None:
        super().__init__("project_memory")
        self.memory_store = memory_store
        self.event_store = event_store

    async def before_run(self, *, agent, session, context, state) -> None:
        workspace_state = _workspace_state_from_session(session)
        if workspace_state is None:
            return

        facts = self.memory_store.list_facts(scopes=("project", "user"))
        if not facts:
            return

        selected = facts[-12:]
        lines = ["Relevant project memory:"]
        for fact in selected:
            tag_text = f" tags={', '.join(fact.tags)}" if fact.tags else ""
            source_text = f" source={fact.source}" if fact.source else ""
            lines.append(f"- [{fact.kind}] {fact.content}{source_text}{tag_text}")
        context.extend_instructions(self.source_id, "\n".join(lines))
        self.memory_store.touch_facts(
            tuple(fact.id for fact in selected),
            last_used_at=workspace_state.created_at,
        )
        _append_provider_event(
            event_store=self.event_store,
            workspace_state=workspace_state,
            kind="memory_retrieved",
            summary=f"Loaded {len(selected)} durable memory facts",
        )


class ArtifactContextProvider(BaseContextProvider):
    def __init__(self, artifact_store: ArtifactStore, event_store: EventStore) -> None:
        super().__init__("artifact_context")
        self.artifact_store = artifact_store
        self.event_store = event_store

    async def before_run(self, *, agent, session, context, state) -> None:
        workspace_state = _workspace_state_from_session(session)
        if workspace_state is None:
            return

        relevant = [
            artifact
            for artifact in self.artifact_store.list_artifacts(workspace_state.session_id)
            if artifact.task_id == workspace_state.task_id or artifact.thread_id == workspace_state.thread_id
        ]
        if not relevant:
            return

        selected = relevant[-3:]
        lines = ["Relevant artifacts:"]
        for artifact in selected:
            body = self.artifact_store.read_artifact_body(artifact).rstrip("\n")
            lines.extend(
                [
                    f"### {artifact.title} [{artifact.kind}]",
                    body if body else "(empty)",
                    "",
                ]
            )
        context.extend_instructions(self.source_id, "\n".join(lines).strip())
        _append_provider_event(
            event_store=self.event_store,
            workspace_state=workspace_state,
            kind="artifact_context_loaded",
            summary=f"Loaded {len(selected)} relevant artifacts",
        )


class AgentProfileContextProvider(BaseContextProvider):
    def __init__(self, definition: DefinitionDocument) -> None:
        super().__init__("agent_profile")
        self.definition = definition

    async def before_run(self, *, agent, session, context, state) -> None:
        role = str(self.definition.metadata.get("role", self.definition.id))
        tools = self.definition.metadata.get("tools", [])
        flags = []
        for key in ("can_speak_unprompted", "can_interrupt_on_risk", "can_propose_replan", "can_request_user_input"):
            if key in self.definition.metadata:
                flags.append(f"{key}={self.definition.metadata[key]}")
        lines = [
            f"Agent profile: {self.definition.id}",
            f"Role: {role}",
            f"Tools: {', '.join(str(tool) for tool in tools) if isinstance(tools, list) and tools else 'none'}",
        ]
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
        context.extend_instructions(self.source_id, "\n".join(lines))


def _workspace_state_from_session(session) -> WorkspaceState | None:
    raw = session.state.get(WORKSPACE_STATE_KEY)
    if not isinstance(raw, dict):
        return None
    session_id = raw.get("session_id")
    thread_id = raw.get("thread_id")
    agent_id = raw.get("agent_id")
    created_at = raw.get("created_at")
    task_id = raw.get("task_id")
    if not isinstance(session_id, str) or not isinstance(thread_id, str) or not isinstance(agent_id, str) or type(created_at) is not int:
        return None
    if task_id is not None and not isinstance(task_id, str):
        task_id = None
    return WorkspaceState(
        session_id=session_id,
        thread_id=thread_id,
        agent_id=agent_id,
        created_at=created_at,
        task_id=task_id,
    )


def _trim_duplicate_input(messages: list[Message], input_messages: list[Message]) -> list[Message]:
    if not messages or not input_messages:
        return messages

    last_message = messages[-1]
    last_input = input_messages[-1]
    if last_message.role != last_input.role:
        return messages
    if last_message.text != last_input.text:
        return messages
    return messages[:-1]


def _append_provider_event(
    *,
    event_store: EventStore,
    workspace_state: WorkspaceState,
    kind: str,
    summary: str,
) -> None:
    event_store.append_event(
        session_id=workspace_state.session_id,
        event_id=f"event-{uuid4().hex}",
        kind=kind,
        summary=summary,
        created_at=workspace_state.created_at,
        thread_id=workspace_state.thread_id,
        task_id=workspace_state.task_id,
    )
