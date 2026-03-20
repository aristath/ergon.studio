from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from agent_framework import Agent, Message, ResponseStream

from ergon_studio.agent_session_store import AgentSessionStore
from ergon_studio.approval_store import ApprovalStore
from ergon_studio.agent_factory import build_agent, compose_instructions
from ergon_studio.artifact_store import ArtifactStore
from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.command_store import CommandStore
from ergon_studio.config import save_global_config_text
from ergon_studio.conversation_store import ConversationStore
from ergon_studio.context_providers import WORKSPACE_STATE_KEY
from ergon_studio.definitions import DefinitionDocument, parse_definition_text, save_definition_text
from ergon_studio.event_store import EventStore
from ergon_studio.live_runtime import LiveMessageDraft, LiveRuntimeEvent, LiveRuntimeState
from ergon_studio.memory_store import MemoryStore
from ergon_studio.paths import StudioPaths
from ergon_studio.retrieval import RetrievalIndex
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.session_store import SessionStore, default_session_title
from ergon_studio.storage.models import ApprovalRecord, ArtifactRecord, CommandRunRecord, EventRecord, MemoryFactRecord, MessageRecord, SessionRecord, TaskRecord, ThreadRecord, ToolCallRecord, WorkflowRunRecord
from ergon_studio.task_store import TaskStore
from ergon_studio.tool_call_store import ToolCallStore
from ergon_studio.tool_context import ToolExecutionContext, current_tool_execution_context, use_tool_execution_context
from ergon_studio.tool_registry import build_workspace_tool_registry
from ergon_studio.whiteboard_store import TaskWhiteboardRecord, WhiteboardStore
from ergon_studio.workflow_compiler import compile_workflow_definition, validate_workflow_group, workflow_step_groups_for_definition
from ergon_studio.workflow_policy import acceptance_criteria_for_mode, acceptance_mode_for_metadata, delivery_candidate_for_metadata, selection_hints_for_metadata, step_groups_for_metadata
from ergon_studio.workflow_runtime import execute_defined_workflow
from ergon_studio.workflow_store import WorkflowStore


MAIN_THREAD_PREFIX = "thread-main"


@dataclass(frozen=True)
class WorkflowRunStepView:
    task: TaskRecord
    threads: tuple[ThreadRecord, ...]


@dataclass(frozen=True)
class WorkflowRunView:
    workflow_run: WorkflowRunRecord
    root_task: TaskRecord | None
    steps: tuple[WorkflowRunStepView, ...]


@dataclass(frozen=True)
class OrchestratorTurnDecision:
    mode: str
    workflow_id: str | None = None
    agent_id: str | None = None
    title: str | None = None
    request: str | None = None
    goal: str | None = None
    deliverable_expected: bool = False


@dataclass(frozen=True)
class DeliveryAuditDecision:
    deliverable_expected: bool
    reconsider: bool
    reason: str = ""


@dataclass(frozen=True)
class DelegationReviewVerdict:
    accepted: bool
    summary: str
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedOrchestratorTurn:
    user_message: MessageRecord
    decision: OrchestratorTurnDecision
    resolved_goal: str
    resolved_request: str


_DEFAULT_CONTEXT_LENGTH = 131072  # 128k — safe default for modern coding models
_COMPACTION_THRESHOLD = 0.95
_COMPACTION_MAX_FAILURES = 3

_COMPACTION_PROMPT = (
    "Summarize this conversation for continuity. You are compacting a long session "
    "so it can continue within the model's context window.\n\n"
    "PRESERVE (these must survive compaction):\n"
    "- Current goals, active tasks, and their states\n"
    "- Decisions made and their rationale\n"
    "- Key technical facts: file paths, architecture choices, naming conventions\n"
    "- Code snippets that were written or modified (include the actual code)\n"
    "- Outstanding issues, blockers, and next steps\n"
    "- User preferences and corrections expressed during the session\n"
    "- Tool call results that informed decisions\n\n"
    "DISCARD (safe to drop):\n"
    "- Verbose intermediate reasoning and failed attempts\n"
    "- Repeated explanations of the same concept\n"
    "- Full command outputs when only the conclusion matters\n"
    "- Progress messages and status updates\n\n"
    "Format the summary as a structured briefing that another instance of yourself "
    "could read and seamlessly continue the work."
)


@dataclass(frozen=True)
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]
    session_store: SessionStore
    agent_session_store: AgentSessionStore
    conversation_store: ConversationStore
    task_store: TaskStore
    workflow_store: WorkflowStore
    event_store: EventStore
    approval_store: ApprovalStore
    memory_store: MemoryStore
    whiteboard_store: WhiteboardStore
    artifact_store: ArtifactStore
    command_store: CommandStore
    tool_call_store: ToolCallStore
    retrieval_index: RetrievalIndex
    live_state: LiveRuntimeState
    main_session_id: str
    main_thread_id: str
    _accumulated_tokens: int
    _compaction_failure_count: int

    def build_agent(self, agent_id: str):
        return build_agent(
            self.registry,
            agent_id,
            tool_registry=self.tool_registry,
            conversation_store=self.conversation_store,
            memory_store=self.memory_store,
            artifact_store=self.artifact_store,
            whiteboard_store=self.whiteboard_store,
            event_store=self.event_store,
            tool_call_store=self.tool_call_store,
            retrieval_index=self.retrieval_index,
        )

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

    def provider_details(self, provider_name: str) -> dict[str, object] | None:
        providers = self.registry.config.get("providers", {})
        if not isinstance(providers, dict):
            return None
        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            return None
        return provider

    def provider_capabilities(self, provider_name: str) -> dict[str, object]:
        provider = self.provider_details(provider_name)
        if provider is None:
            return {}
        capabilities = provider.get("capabilities", {})
        if not isinstance(capabilities, dict):
            return {}
        return capabilities

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

    def read_workflow_definition_text(self, workflow_id: str) -> str:
        definition = self.registry.workflow_definitions[workflow_id]
        return definition.path.read_text(encoding="utf-8")

    def read_global_config_text(self) -> str:
        return self.paths.config_path.read_text(encoding="utf-8")

    def current_session(self) -> SessionRecord | None:
        return self.session_store.get_session(self.main_session_id)

    def list_sessions(self, *, include_archived: bool = False) -> list[SessionRecord]:
        return self.session_store.list_sessions(include_archived=include_archived)

    def session_preview(self, session_id: str, *, limit: int = 72) -> str:
        thread_id = _main_thread_id_for_session(session_id)
        messages = self.conversation_store.list_messages(thread_id)
        for message in reversed(messages):
            body = self.conversation_store.read_message_body(message).strip()
            if not body:
                continue
            return _truncate_preview(" ".join(body.split()), limit=limit)
        return "No messages yet."

    def rename_session(self, *, session_id: str, title: str, created_at: int) -> SessionRecord:
        session = self.session_store.rename_session(
            session_id=session_id,
            title=title,
            updated_at=created_at,
        )
        self.append_event(
            kind="session_renamed",
            summary=f"Renamed session {session.id} to {session.title}",
            created_at=created_at,
        )
        return session

    def archive_session(self, *, session_id: str, created_at: int) -> SessionRecord:
        session = self.session_store.archive_session(
            session_id=session_id,
            archived_at=created_at,
        )
        self.append_event(
            kind="session_archived",
            summary=f"Archived session {session.id}",
            created_at=created_at,
        )
        return session

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

    def save_workflow_definition_text(
        self,
        *,
        workflow_id: str,
        text: str,
        created_at: int | None = None,
    ) -> DefinitionDocument:
        definition = self.registry.workflow_definitions[workflow_id]
        saved_preview = parse_definition_text(text, path=definition.path)
        compile_workflow_definition(saved_preview)
        saved = save_definition_text(definition.path, text)
        self.reload_registry()
        if created_at is None:
            created_at = int(time.time())
        self.append_event(
            kind="definition_saved",
            summary=f"Saved workflow definition {workflow_id}",
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

    def list_workflow_runs(self) -> list[WorkflowRunRecord]:
        return self.workflow_store.list_workflow_runs(self.main_session_id)

    def get_workflow_run(self, workflow_run_id: str) -> WorkflowRunRecord | None:
        return self.workflow_store.get_workflow_run(workflow_run_id)

    def describe_workflow_run(self, workflow_run_id: str) -> WorkflowRunView | None:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            return None

        root_task = None
        if workflow_run.root_task_id is not None:
            root_task = self.get_task(workflow_run.root_task_id)

        child_tasks = [
            task
            for task in self.list_tasks()
            if task.parent_task_id == workflow_run.root_task_id
        ]
        child_tasks = sorted(
            child_tasks,
            key=lambda task: (task.created_at, task.id),
        )
        threads_by_task_id: dict[str, list[ThreadRecord]] = {}
        for thread in self.list_threads():
            if thread.parent_task_id is None:
                continue
            threads_by_task_id.setdefault(thread.parent_task_id, []).append(thread)

        steps = tuple(
            WorkflowRunStepView(
                task=task,
                threads=tuple(
                    sorted(
                        threads_by_task_id.get(task.id, []),
                        key=lambda thread: (thread.created_at, thread.id),
                    )
                ),
            )
            for task in child_tasks
        )
        return WorkflowRunView(
            workflow_run=workflow_run,
            root_task=root_task,
            steps=steps,
        )

    def preferred_thread_id_for_workflow_run(self, workflow_run_id: str) -> str | None:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            return None
        if workflow_run.last_thread_id is not None and self.get_thread(workflow_run.last_thread_id) is not None:
            return workflow_run.last_thread_id

        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return None
        for step in run_view.steps:
            if step.threads:
                return step.threads[0].id
        return None

    def list_artifacts_for_workflow_run(self, workflow_run_id: str) -> list[ArtifactRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids = {thread.id for thread in self.list_threads_for_workflow_run(workflow_run_id)}
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)

        artifacts = [
            artifact
            for artifact in self.list_artifacts()
            if artifact.task_id in task_ids or artifact.thread_id in thread_ids
        ]
        return sorted(
            artifacts,
            key=lambda artifact: (_workflow_artifact_priority(artifact.kind), artifact.created_at, artifact.id),
        )

    def list_threads_for_workflow_run(self, workflow_run_id: str) -> list[ThreadRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []
        threads: list[ThreadRecord] = []
        if run_view.root_task is not None:
            threads.extend(
                thread
                for thread in self.list_threads()
                if thread.parent_task_id == run_view.root_task.id
            )
        for step in run_view.steps:
            threads.extend(step.threads)
        unique_threads = {thread.id: thread for thread in threads}
        return sorted(unique_threads.values(), key=lambda thread: (thread.created_at, thread.id))

    def list_events_for_workflow_run(self, workflow_run_id: str) -> list[EventRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids = {thread.id for thread in self.list_threads_for_workflow_run(workflow_run_id)}
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)

        events = [
            event
            for event in self.list_events()
            if event.task_id in task_ids or event.thread_id in thread_ids
        ]
        return sorted(events, key=lambda event: (event.created_at, event.id))

    def list_events(self) -> list[EventRecord]:
        return self.event_store.list_events(self.main_session_id)

    def list_approvals(self) -> list[ApprovalRecord]:
        return self.approval_store.list_approvals(self.main_session_id)

    def list_pending_approvals(self) -> list[ApprovalRecord]:
        return [
            approval
            for approval in self.list_approvals()
            if approval.status == "pending"
        ]

    def list_approvals_for_workflow_run(self, workflow_run_id: str) -> list[ApprovalRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids = {thread.id for thread in self.list_threads_for_workflow_run(workflow_run_id)}
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)

        approvals = [
            approval
            for approval in self.list_approvals()
            if approval.task_id in task_ids or approval.thread_id in thread_ids
        ]
        return sorted(approvals, key=lambda approval: (approval.created_at, approval.id))

    def list_pending_approvals_for_workflow_run(self, workflow_run_id: str) -> list[ApprovalRecord]:
        return [
            approval
            for approval in self.list_approvals_for_workflow_run(workflow_run_id)
            if approval.status == "pending"
        ]

    def read_approval_payload(self, approval_id: str) -> dict[str, object] | None:
        approval = self.approval_store.get_approval(approval_id)
        if approval is None:
            raise ValueError(f"unknown approval: {approval_id}")
        return self.approval_store.read_payload(approval)

    def list_agent_ids(self) -> list[str]:
        return sorted(self.registry.agent_definitions.keys())

    def list_agent_summaries(self) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for agent_id in self.list_agent_ids():
            definition = self.registry.agent_definitions[agent_id]
            summaries.append(
                {
                    "id": agent_id,
                    "name": str(definition.metadata.get("name", agent_id)),
                    "role": str(definition.metadata.get("role", agent_id)),
                    "tools": tuple(str(tool) for tool in definition.metadata.get("tools", []) or []),
                    "status": self.agent_status_summary(agent_id),
                }
            )
        return summaries

    def describe_agent_definition(self, agent_id: str) -> dict[str, object]:
        definition = self.registry.agent_definitions[agent_id]
        return {
            "id": definition.id,
            "name": str(definition.metadata.get("name", definition.id)),
            "role": str(definition.metadata.get("role", definition.id)),
            "metadata": dict(definition.metadata),
            "sections": dict(definition.sections),
        }

    def list_workflow_ids(self) -> list[str]:
        return sorted(self.registry.workflow_definitions.keys())

    def list_workflow_summaries(self) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for workflow_id in self.list_workflow_ids():
            definition = self.registry.workflow_definitions[workflow_id]
            purpose = definition.sections.get("Purpose", "")
            summaries.append(
                {
                    "id": workflow_id,
                    "name": str(definition.metadata.get("name", workflow_id)),
                    "orchestration": str(definition.metadata.get("orchestration", "unknown")),
                    "acceptance_mode": acceptance_mode_for_metadata(definition.metadata),
                    "delivery_candidate": delivery_candidate_for_metadata(definition.metadata),
                    "selection_hints": selection_hints_for_metadata(definition.metadata),
                    "step_groups": self.workflow_step_groups(workflow_id),
                    "purpose": purpose,
                }
            )
        return summaries

    def describe_workflow_definition(self, workflow_id: str) -> dict[str, object]:
        definition = self.registry.workflow_definitions[workflow_id]
        return {
            "id": definition.id,
            "name": str(definition.metadata.get("name", definition.id)),
            "orchestration": str(definition.metadata.get("orchestration", "unknown")),
            "metadata": dict(definition.metadata),
            "sections": dict(definition.sections),
            "step_groups": self.workflow_step_groups(workflow_id),
        }

    def workflow_orchestration(self, workflow_id: str) -> str:
        definition = self.registry.workflow_definitions[workflow_id]
        return str(definition.metadata.get("orchestration", "unknown"))

    def workflow_steps(self, workflow_id: str) -> tuple[str, ...]:
        groups = self.workflow_step_groups(workflow_id)
        if any(len(group) != 1 for group in groups):
            raise ValueError(f"workflow '{workflow_id}' includes grouped steps")
        return tuple(group[0] for group in groups)

    def workflow_step_groups(self, workflow_id: str) -> tuple[tuple[str, ...], ...]:
        definition = self.registry.workflow_definitions.get(workflow_id)
        if definition is None:
            raise ValueError(f"unknown workflow: {workflow_id}")
        return workflow_step_groups_for_definition(definition)

    def list_provider_ids(self) -> list[str]:
        return sorted(self.registry.config.get("providers", {}).keys())

    def list_memory_facts(self) -> list[MemoryFactRecord]:
        return self.memory_store.list_facts()

    def get_task_whiteboard(self, task_id: str) -> TaskWhiteboardRecord | None:
        return self.whiteboard_store.read_task_whiteboard(task_id)

    def read_task_whiteboard_text(self, task_id: str) -> str:
        return self.whiteboard_store.read_task_whiteboard_text(task_id)

    def save_task_whiteboard_text(self, *, task_id: str, text: str, created_at: int | None = None) -> TaskWhiteboardRecord:
        record = self.whiteboard_store.save_task_whiteboard_text(task_id, text)
        if created_at is None:
            created_at = int(time.time())
        self.append_event(
            kind="whiteboard_saved",
            summary=f"Saved task whiteboard {task_id}",
            created_at=created_at,
            task_id=task_id,
        )
        return record

    def list_artifacts(self) -> list[ArtifactRecord]:
        return self.artifact_store.list_artifacts(self.main_session_id)

    def list_command_runs(self) -> list[CommandRunRecord]:
        return self.command_store.list_command_runs(self.main_session_id)

    def list_tool_calls(self) -> list[ToolCallRecord]:
        return self.tool_call_store.list_tool_calls(self.main_session_id)

    def list_command_runs_for_workflow_run(self, workflow_run_id: str) -> list[CommandRunRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids = {thread.id for thread in self.list_threads_for_workflow_run(workflow_run_id)}
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)

        command_runs = [
            command_run
            for command_run in self.list_command_runs()
            if command_run.task_id in task_ids or command_run.thread_id in thread_ids
        ]
        return sorted(command_runs, key=lambda command_run: (command_run.created_at, command_run.id))

    def list_tool_calls_for_workflow_run(self, workflow_run_id: str) -> list[ToolCallRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids = {thread.id for thread in self.list_threads_for_workflow_run(workflow_run_id)}
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)

        tool_calls = [
            tool_call
            for tool_call in self.list_tool_calls()
            if tool_call.task_id in task_ids or tool_call.thread_id in thread_ids
        ]
        return sorted(tool_calls, key=lambda tool_call: (tool_call.created_at, tool_call.id))

    def read_command_output(self, command_run_id: str) -> str:
        for command_run in self.list_command_runs():
            if command_run.id == command_run_id:
                return self.command_store.read_command_output(command_run)
        raise ValueError(f"unknown command run: {command_run_id}")

    def read_tool_call_request(self, tool_call_id: str) -> str:
        for tool_call in self.list_tool_calls():
            if tool_call.id == tool_call_id:
                return self.tool_call_store.read_request(tool_call)
        raise ValueError(f"unknown tool call: {tool_call_id}")

    def read_tool_call_response(self, tool_call_id: str) -> str:
        for tool_call in self.list_tool_calls():
            if tool_call.id == tool_call_id:
                return self.tool_call_store.read_response(tool_call)
        raise ValueError(f"unknown tool call: {tool_call_id}")

    def read_artifact_body(self, artifact_id: str) -> str:
        for artifact in self.list_artifacts():
            if artifact.id == artifact_id:
                return self.artifact_store.read_artifact_body(artifact)
        raise ValueError(f"unknown artifact: {artifact_id}")

    def write_workspace_file(
        self,
        path: str,
        content: str,
        *,
        created_at: int | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        require_approval: bool = True,
    ) -> dict[str, str]:
        if created_at is None:
            created_at = int(time.time())

        context = current_tool_execution_context()
        resolved_thread_id = thread_id if thread_id is not None else (context.thread_id if context is not None else None)
        resolved_task_id = task_id if task_id is not None else (context.task_id if context is not None else None)
        resolved_agent_id = agent_id if agent_id is not None else (context.agent_id if context is not None else None)

        if require_approval and self._approval_mode_for_action("write_file") in {"ask", "block"}:
            approval = self.request_approval(
                approval_id=f"approval-{uuid4().hex[:8]}",
                requester=resolved_agent_id or "user",
                action="write_file",
                risk_class="moderate",
                reason=f"Write file in workspace: {path}",
                created_at=created_at,
                thread_id=resolved_thread_id,
                task_id=resolved_task_id,
                payload={
                    "path": path,
                    "content": content,
                    "thread_id": resolved_thread_id,
                    "task_id": resolved_task_id,
                    "agent_id": resolved_agent_id or "user",
                },
            )
            return {"path": path, "status": "awaiting_approval", "approval_id": approval.id}

        target = self._resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        relative_path = str(target.relative_to(self.paths.project_root.resolve()))
        self.append_event(
            kind="file_written",
            summary=f"{resolved_agent_id or 'user'} wrote {relative_path}",
            created_at=created_at,
            thread_id=resolved_thread_id,
            task_id=resolved_task_id,
        )
        return {"path": relative_path, "status": "written"}

    def patch_workspace_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        created_at: int | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        require_approval: bool = True,
    ) -> dict[str, int | str]:
        if created_at is None:
            created_at = int(time.time())

        context = current_tool_execution_context()
        resolved_thread_id = thread_id if thread_id is not None else (context.thread_id if context is not None else None)
        resolved_task_id = task_id if task_id is not None else (context.task_id if context is not None else None)
        resolved_agent_id = agent_id if agent_id is not None else (context.agent_id if context is not None else None)

        if require_approval and self._approval_mode_for_action("patch_file") in {"ask", "block"}:
            approval = self.request_approval(
                approval_id=f"approval-{uuid4().hex[:8]}",
                requester=resolved_agent_id or "user",
                action="patch_file",
                risk_class="moderate",
                reason=f"Patch file in workspace: {path}",
                created_at=created_at,
                thread_id=resolved_thread_id,
                task_id=resolved_task_id,
                payload={
                    "path": path,
                    "old_text": old_text,
                    "new_text": new_text,
                    "thread_id": resolved_thread_id,
                    "task_id": resolved_task_id,
                    "agent_id": resolved_agent_id or "user",
                },
            )
            return {"path": path, "replacements": 0, "status": "awaiting_approval", "approval_id": approval.id}

        target = self._resolve_workspace_path(path)
        file_content = target.read_text(encoding="utf-8")
        replacements = file_content.count(old_text)
        if replacements == 0:
            raise ValueError("old_text was not found in the file")
        updated = file_content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        relative_path = str(target.relative_to(self.paths.project_root.resolve()))
        self.append_event(
            kind="file_patched",
            summary=f"{resolved_agent_id or 'user'} patched {relative_path}",
            created_at=created_at,
            thread_id=resolved_thread_id,
            task_id=resolved_task_id,
        )
        return {"path": relative_path, "replacements": 1, "status": "patched"}

    def create_thread(
        self,
        *,
        thread_id: str,
        kind: str,
        created_at: int,
        assigned_agent_id: str | None = None,
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
            assigned_agent_id=assigned_agent_id,
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

    def create_agent_thread(
        self,
        *,
        agent_id: str,
        created_at: int,
        parent_task_id: str | None = None,
    ) -> ThreadRecord:
        return self.create_thread(
            thread_id=f"thread-agent-{agent_id}-{uuid4().hex[:8]}",
            kind="agent_direct",
            created_at=created_at,
            assigned_agent_id=agent_id,
            summary=f"Direct thread with {agent_id}",
            parent_task_id=parent_task_id,
        )

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        return self.conversation_store.get_thread(thread_id)

    def list_main_messages(self) -> list[MessageRecord]:
        return self.conversation_store.list_messages(self.main_thread_id)

    def latest_main_user_message_body(self) -> str | None:
        for message in reversed(self.list_main_messages()):
            if message.sender != "user":
                continue
            return self.conversation_store.read_message_body(message).rstrip("\n")
        return None

    def recent_main_user_context(self, *, limit: int = 4) -> str:
        user_messages = [message for message in self.list_main_messages() if message.sender == "user"]
        if not user_messages:
            return ""
        selected = user_messages[-limit:]
        parts = []
        for message in selected:
            body = self.conversation_store.read_message_body(message).rstrip("\n")
            if body:
                parts.append(body)
        return "\n\n".join(parts).strip()

    def list_thread_messages(self, thread_id: str) -> list[MessageRecord]:
        return self.conversation_store.list_messages(thread_id)

    def list_live_message_drafts(self) -> tuple[LiveMessageDraft, ...]:
        return self.live_state.list_drafts()

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

    def record_user_message_to_main_thread(
        self,
        *,
        body: str,
        created_at: int,
    ) -> MessageRecord:
        user_message = self.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="user",
            kind="chat",
            body=body,
            created_at=created_at,
        )
        self._refresh_session_title_from_user_turn(
            body=body,
            created_at=created_at,
        )
        return user_message

    async def send_user_message_to_orchestrator(
        self,
        *,
        body: str,
        created_at: int,
    ) -> tuple[MessageRecord, MessageRecord | None]:
        stream = self.stream_user_message_to_orchestrator(
            body=body,
            created_at=created_at,
        )
        async for _ in stream:
            pass
        return await stream.get_final_response()

    def stream_user_message_to_orchestrator(
        self,
        *,
        body: str,
        created_at: int,
        user_message: MessageRecord | None = None,
    ) -> ResponseStream[LiveRuntimeEvent, tuple[MessageRecord, MessageRecord | None]]:
        resolved_user_message = user_message or self.record_user_message_to_main_thread(
            body=body,
            created_at=created_at,
        )
        state: dict[str, MessageRecord | None] = {"user": resolved_user_message, "reply": None}

        async def _events():
            prepared = await self._prepare_orchestrator_turn(
                body=body,
                created_at=created_at,
                user_message=resolved_user_message,
            )
            state["user"] = prepared.user_message
            decision = prepared.decision
            if decision.mode == "workflow" and decision.workflow_id is not None:
                result = await self.run_workflow(
                    workflow_id=decision.workflow_id,
                    goal=prepared.resolved_goal,
                    created_at=created_at + 2,
                    parent_thread_id=self.main_thread_id,
                )
                summary = self._format_workflow_summary(
                    workflow_id=decision.workflow_id,
                    result=result,
                )
                state["reply"] = self.append_message_to_main_thread(
                    message_id=f"message-{uuid4().hex}",
                    sender="orchestrator",
                    kind="status_update",
                    body=summary,
                    created_at=created_at + 3,
                )
                return
            if decision.mode == "delegate" and decision.agent_id is not None:
                result = await self.delegate_to_agent(
                    agent_id=decision.agent_id,
                    request=prepared.resolved_request,
                    title=decision.title,
                    created_at=created_at + 2,
                    parent_thread_id=self.main_thread_id,
                )
                summary = self._format_delegation_summary(
                    agent_id=decision.agent_id,
                    result=result,
                )
                state["reply"] = self.append_message_to_main_thread(
                    message_id=f"message-{uuid4().hex}",
                    sender="orchestrator",
                    kind="status_update",
                    body=summary,
                    created_at=created_at + 3,
                )
                return
            turn_stream = self._stream_agent_turn(
                thread_id=self.main_thread_id,
                agent_id="orchestrator",
                prompt_sender="user",
                reply_sender="orchestrator",
                body=prepared.resolved_request if decision.request else body,
                created_at=created_at + 2,
                record_prompt=False,
            )
            async for event in turn_stream:
                yield event
            _, state["reply"] = await turn_stream.get_final_response()

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: (
                state["user"],
                state["reply"],
            ),
        )

    async def _prepare_orchestrator_turn(
        self,
        *,
        body: str,
        created_at: int,
        user_message: MessageRecord,
    ) -> PreparedOrchestratorTurn:
        decision = await self._decide_orchestrator_turn(body=body, created_at=created_at + 1)
        delivery_audit = await self._audit_orchestrator_delivery_turn(
            body=body,
            decision=decision,
            created_at=created_at + 1,
        )
        if delivery_audit.deliverable_expected and not decision.deliverable_expected:
            decision = OrchestratorTurnDecision(
                mode=decision.mode,
                workflow_id=decision.workflow_id,
                agent_id=decision.agent_id,
                title=decision.title,
                request=decision.request,
                goal=decision.goal,
                deliverable_expected=True,
            )
            self.append_event(
                kind="orchestrator_deliverable_detected",
                summary="Detected delivery intent for the current turn",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
        delivery_context = self.recent_main_user_context(limit=4) or body
        resolved_goal = decision.goal or delivery_context
        resolved_request = decision.request or resolved_goal
        if decision.deliverable_expected and decision.mode in {"workflow", "delegate"}:
            resolved_goal = delivery_context
            resolved_request = delivery_context
        if (
            delivery_audit.reconsider
            and decision.mode == "workflow"
            and decision.workflow_id is not None
            and self._workflow_is_non_delivery(decision.workflow_id)
        ):
            self.append_event(
                kind="orchestrator_non_delivery_rejected",
                summary="Planner selected a non-delivery workflow for a delivery-sensitive turn",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
        if delivery_audit.reconsider and delivery_audit.reason:
            self.append_event(
                kind="orchestrator_delivery_reconsidered",
                summary=delivery_audit.reason,
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
        if delivery_audit.reconsider:
            reconsidered = await self._reconsider_delivery_turn(
                body=delivery_context,
                created_at=created_at + 1,
                reason=delivery_audit.reason,
            )
            if reconsidered is not None:
                decision = reconsidered
            else:
                decision = OrchestratorTurnDecision(
                    mode="act",
                    goal=delivery_context,
                    request=delivery_context,
                    deliverable_expected=True,
                )
        self.append_event(
            kind="orchestrator_turn_planned",
            summary=f"Orchestrator selected {decision.mode}",
            created_at=created_at + 1,
            thread_id=self.main_thread_id,
        )
        return PreparedOrchestratorTurn(
            user_message=user_message,
            decision=decision,
            resolved_goal=resolved_goal,
            resolved_request=resolved_request,
        )

    def _stream_agent_turn(
        self,
        *,
        thread_id: str,
        agent_id: str,
        prompt_sender: str,
        reply_sender: str,
        body: str,
        created_at: int,
        record_prompt: bool = True,
    ) -> ResponseStream[LiveRuntimeEvent, tuple[MessageRecord | None, MessageRecord | None]]:
        state: dict[str, MessageRecord | None] = {"prompt": None, "reply": None}

        async def _events():
            async for event in self._stream_agent_turn_events(
                thread_id=thread_id,
                agent_id=agent_id,
                prompt_sender=prompt_sender,
                reply_sender=reply_sender,
                body=body,
                created_at=created_at,
                state=state,
                record_prompt=record_prompt,
            ):
                yield event

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: (
                state["prompt"],
                state["reply"],
            ),
        )

    async def _stream_agent_turn_events(
        self,
        *,
        thread_id: str,
        agent_id: str,
        prompt_sender: str,
        reply_sender: str,
        body: str,
        created_at: int,
        state: dict[str, MessageRecord | None],
        record_prompt: bool = True,
    ):
        prompt_message = None
        if record_prompt:
            prompt_message = self.append_message_to_thread(
                thread_id=thread_id,
                message_id=f"message-{uuid4().hex}",
                sender=prompt_sender,
                kind="chat",
                body=body,
                created_at=created_at,
            )
        state["prompt"] = prompt_message

        try:
            agent = self.build_agent(agent_id)
        except (KeyError, ValueError) as exc:
            self.append_event(
                kind="agent_unavailable",
                summary=f"{agent_id} unavailable: {exc}",
                created_at=created_at + 1,
                thread_id=thread_id,
            )
            return

        thread = self.get_thread(thread_id)
        runtime_session_id = thread.session_id if thread is not None else self.main_session_id
        session = self.agent_session_store.load_or_create_session(
            session_id=runtime_session_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session_factory=lambda session_id: agent.create_session(session_id=session_id),
        )
        session.state[WORKSPACE_STATE_KEY] = {
            "session_id": runtime_session_id,
            "thread_id": thread_id,
            "task_id": thread.parent_task_id if thread is not None else None,
            "agent_id": agent_id,
            "created_at": created_at,
        }
        tool_context = ToolExecutionContext(
            session_id=runtime_session_id,
            thread_id=thread_id,
            task_id=thread.parent_task_id if thread is not None else None,
            agent_id=agent_id,
        )
        draft_id = f"draft-{uuid4().hex}"
        started = self.live_state.start_draft(
            draft_id=draft_id,
            thread_id=thread_id,
            sender=reply_sender,
            kind="chat",
            created_at=created_at + 1,
        )
        yield started

        response_text = ""
        try:
            with use_tool_execution_context(tool_context):
                stream_or_response = agent.run(
                    [
                        Message(
                            role="user",
                            text=body,
                            author_name=prompt_sender,
                        )
                    ],
                    session=session,
                    stream=True,
                )
                if hasattr(stream_or_response, "__aiter__") and hasattr(stream_or_response, "get_final_response"):
                    async for update in stream_or_response:
                        text_delta = getattr(update, "text", "")
                        if not text_delta:
                            continue
                        response_text += text_delta
                        event = self.live_state.append_delta(
                            draft_id=draft_id,
                            delta=text_delta,
                            created_at=created_at + 1,
                        )
                        if event is not None:
                            yield event
                    response = await stream_or_response.get_final_response()
                else:
                    response = await stream_or_response
        except Exception as exc:
            self.append_event(
                kind="agent_failed",
                summary=f"{agent_id} run failed: {type(exc).__name__}: {exc}",
                created_at=created_at + 1,
                thread_id=thread_id,
            )
            self.agent_session_store.save_session(
                session_id=runtime_session_id,
                thread_id=thread_id,
                agent_id=agent_id,
                session=session,
            )
            failed = self.live_state.fail_draft(
                draft_id=draft_id,
                error=str(exc),
                created_at=created_at + 1,
            )
            if failed is not None:
                yield failed
            return

        self.agent_session_store.save_session(
            session_id=runtime_session_id,
            thread_id=thread_id,
            agent_id=agent_id,
            session=session,
        )
        self.track_token_usage(response)
        final_text = getattr(response, "text", "").strip()
        if final_text and not response_text:
            response_text = final_text
            event = self.live_state.append_delta(
                draft_id=draft_id,
                delta=final_text,
                created_at=created_at + 1,
            )
            if event is not None:
                yield event
        if not final_text:
            final_text = response_text.strip()
        if not final_text:
            failed = self.live_state.fail_draft(
                draft_id=draft_id,
                error="empty response",
                created_at=created_at + 1,
            )
            if failed is not None:
                yield failed
            return

        reply_message = self.append_message_to_thread(
            thread_id=thread_id,
            message_id=f"message-{uuid4().hex}",
            sender=reply_sender,
            kind="chat",
            body=final_text,
            created_at=created_at + 1,
        )
        state["reply"] = reply_message
        completed = self.live_state.complete_draft(
            draft_id=draft_id,
            message_id=reply_message.id,
            created_at=created_at + 1,
        )
        if completed is not None:
            yield completed

    def _refresh_session_title_from_user_turn(self, *, body: str, created_at: int) -> None:
        session = self.current_session()
        if session is None:
            return
        fallback_title = default_session_title(
            session_id=session.id,
            created_at=session.created_at,
        )
        if session.title != fallback_title:
            return
        user_messages = [message for message in self.list_main_messages() if message.sender == "user"]
        if len(user_messages) != 1:
            return
        suggested = _session_title_from_message(body)
        if suggested == fallback_title:
            return
        updated = self.session_store.rename_session(
            session_id=session.id,
            title=suggested,
            updated_at=created_at,
        )
        self.append_event(
            kind="session_titled",
            summary=f"Updated session title to {updated.title}",
            created_at=created_at,
            thread_id=self.main_thread_id,
        )

    async def send_message_to_agent_thread(
        self,
        *,
        thread_id: str,
        body: str,
        created_at: int,
    ) -> tuple[MessageRecord, MessageRecord | None]:
        thread = self.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        if thread.assigned_agent_id is None:
            note = self.append_message_to_thread(
                thread_id=thread_id,
                message_id=f"message-{uuid4().hex}",
                sender="orchestrator",
                kind="chat",
                body=body,
                created_at=created_at,
            )
            return note, None
        return await self._run_agent_turn(
            thread_id=thread_id,
            agent_id=thread.assigned_agent_id,
            prompt_sender="orchestrator",
            reply_sender=thread.assigned_agent_id,
            body=body,
            created_at=created_at,
        )

    async def delegate_to_agent(
        self,
        *,
        agent_id: str,
        request: str,
        title: str | None = None,
        created_at: int | None = None,
        parent_thread_id: str | None = None,
    ) -> dict[str, object]:
        if created_at is None:
            created_at = int(time.time())
        context = current_tool_execution_context()
        resolved_parent_thread_id = parent_thread_id if parent_thread_id is not None else (context.thread_id if context is not None else None)
        task = self.create_task(
            task_id=f"task-{uuid4().hex[:8]}",
            title=title or f"Delegation: {agent_id}",
            state="in_progress",
            created_at=created_at,
        )
        thread = self.create_thread(
            thread_id=f"thread-agent-{agent_id}-{uuid4().hex[:8]}",
            kind="agent_direct",
            created_at=created_at + 1,
            assigned_agent_id=agent_id,
            summary=title or f"Delegated thread for {agent_id}",
            parent_task_id=task.id,
            parent_thread_id=resolved_parent_thread_id,
        )
        _, reply = await self.send_message_to_agent_thread(
            thread_id=thread.id,
            body=request,
            created_at=created_at + 2,
        )
        if reply is None:
            self.update_task_state(
                task_id=task.id,
                state="blocked",
                updated_at=created_at + 3,
            )
            self.append_event(
                kind="delegation_blocked",
                summary=f"Delegation to {agent_id} blocked",
                created_at=created_at + 3,
                thread_id=thread.id,
                task_id=task.id,
            )
            return {
                "status": "blocked",
                "agent_id": agent_id,
                "task_id": task.id,
                "thread_id": thread.id,
                "result": "",
            }

        reply_body = self.conversation_store.read_message_body(reply).rstrip("\n")
        review_thread = self.create_thread(
            thread_id=f"thread-review-orchestrator-{uuid4().hex[:8]}",
            kind="review",
            created_at=created_at + 3,
            assigned_agent_id="orchestrator",
            summary=f"Delegation review for {agent_id}",
            parent_task_id=task.id,
            parent_thread_id=resolved_parent_thread_id,
        )
        self.append_event(
            kind="delegation_review_requested",
            summary=f"Requested orchestrator review for delegated {agent_id} work",
            created_at=created_at + 3,
            thread_id=review_thread.id,
            task_id=task.id,
        )
        verdict = await self._review_delegation_result(
            agent_id=agent_id,
            request=request,
            result=reply_body,
            thread=thread,
            review_thread=review_thread,
            created_at=created_at + 4,
        )
        review_summary = _format_delegation_review_summary(verdict)
        self.append_message_to_thread(
            thread_id=review_thread.id,
            message_id=f"message-{uuid4().hex}",
            sender="orchestrator",
            kind="review",
            body=review_summary,
            created_at=created_at + 5,
        )
        status = "completed" if verdict.accepted else "blocked"
        self.update_task_state(
            task_id=task.id,
            state=status,
            updated_at=created_at + 5,
        )
        self.append_event(
            kind="delegation_completed" if verdict.accepted else "delegation_rejected",
            summary=(
                f"Delegation to {agent_id} completed"
                if verdict.accepted
                else f"Delegation to {agent_id} was rejected by orchestrator review"
            ),
            created_at=created_at + 5,
            thread_id=review_thread.id,
            task_id=task.id,
        )
        return {
            "status": status,
            "agent_id": agent_id,
            "task_id": task.id,
            "thread_id": thread.id,
            "result": reply_body,
            "review_thread_id": review_thread.id,
            "review_summary": review_summary,
        }

    async def _review_delegation_result(
        self,
        *,
        agent_id: str,
        request: str,
        result: str,
        thread: ThreadRecord,
        review_thread: ThreadRecord,
        created_at: int,
    ) -> DelegationReviewVerdict:
        prompt = _render_delegation_review_prompt(
            agent_id=agent_id,
            request=request,
            result=result,
            evidence_lines=self._delegation_evidence_lines(thread.id),
        )
        self.append_message_to_thread(
            thread_id=review_thread.id,
            message_id=f"message-{uuid4().hex}",
            sender="workflow",
            kind="chat",
            body=prompt,
            created_at=created_at,
        )
        parsed = await self._run_structured_delegation_review(
            prompt=prompt,
            review_thread=review_thread,
            created_at=created_at + 1,
        )
        if parsed is None:
            self.append_event(
                kind="delegation_review_unavailable",
                summary=f"Delegation review for {agent_id} fell back because no structured reviewer was available",
                created_at=created_at + 1,
                thread_id=review_thread.id,
                task_id=thread.parent_task_id,
            )
            return DelegationReviewVerdict(
                accepted=False,
                summary="I could not complete a reliable acceptance review for the delegated result.",
            )
        try:
            return _delegation_review_verdict_from_payload(parsed)
        except ValueError:
            self.append_event(
                kind="delegation_review_unavailable",
                summary=f"Delegation review for {agent_id} returned invalid structured output",
                created_at=created_at + 1,
                thread_id=review_thread.id,
                task_id=thread.parent_task_id,
            )
            return DelegationReviewVerdict(
                accepted=False,
                summary="I could not produce a structured acceptance review for the delegated result.",
            )

    async def _run_structured_delegation_review(
        self,
        *,
        prompt: str,
        review_thread: ThreadRecord,
        created_at: int,
    ) -> dict[str, object] | None:
        parsed = await self._run_orchestrator_json_agent(
            agent_id="delegation-orchestrator-review",
            name="Delegation Orchestrator Review",
            description="Structured delegation acceptance review",
            instructions=_delegation_review_instructions(),
            prompt=prompt,
            created_at=created_at,
            session_id=f"{review_thread.id}:delegation-review",
            failure_event_kind="delegation_review_failed",
            failure_summary_prefix="Delegation review failed",
            invalid_event_kind="delegation_review_invalid",
            invalid_summary_prefix="Delegation review returned invalid JSON",
        )
        if parsed is not None:
            return parsed
        try:
            orchestrator = self.build_agent("orchestrator")
        except (KeyError, ValueError):
            return None
        if getattr(orchestrator, "client", None) is not None:
            return None
        try:
            response = await orchestrator.run(
                [
                    Message(
                        role="user",
                        text=prompt,
                        author_name="workflow",
                    )
                ],
                session=orchestrator.create_session(session_id=f"{review_thread.id}:delegation-review"),
            )
        except Exception:
            return None
        self.track_token_usage(response)
        raw = response.text.strip()
        if not raw:
            return None
        try:
            return _parse_turn_decision_json(raw)
        except ValueError:
            return None

    def _delegation_evidence_lines(self, thread_id: str) -> list[str]:
        lines: list[str] = []
        file_changes: list[str] = []
        for tool_call in self.list_tool_calls():
            if tool_call.thread_id != thread_id or tool_call.status != "completed":
                continue
            if tool_call.tool_name not in {"write_file", "patch_file"}:
                continue
            try:
                request = json.loads(self.read_tool_call_request(tool_call.id))
            except json.JSONDecodeError:
                request = {}
            path = request.get("path")
            if isinstance(path, str) and path:
                file_changes.append(f"- {tool_call.tool_name}: {path}")
        if file_changes:
            lines.append("Recorded file changes:")
            lines.extend(file_changes[:8])
            lines.append("")

        command_evidence: list[str] = []
        for command_run in self.list_command_runs():
            if command_run.thread_id != thread_id or command_run.status != "completed":
                continue
            output = self.command_store.read_command_output(command_run).strip()
            command_evidence.append(
                f"- {command_run.command} -> exit {command_run.exit_code}; output: {_truncate_preview(output or '(no output)', limit=160)}"
            )
        if command_evidence:
            lines.append("Recorded command runs:")
            lines.extend(command_evidence[:8])
            lines.append("")

        return lines

    async def run_workflow(
        self,
        *,
        workflow_id: str,
        goal: str,
        created_at: int | None = None,
        parent_thread_id: str | None = None,
    ) -> dict[str, object]:
        if created_at is None:
            created_at = int(time.time())
        context = current_tool_execution_context()
        resolved_parent_thread_id = parent_thread_id if parent_thread_id is not None else (context.thread_id if context is not None else None)
        step_groups = await self._select_initial_workflow_step_groups(
            workflow_id=workflow_id,
            goal=goal,
            created_at=created_at,
        )
        workflow_run, _ = self.start_workflow_run(
            workflow_id=workflow_id,
            created_at=created_at,
            goal=goal,
            step_groups=step_groups,
        )
        review_created_at = created_at + 100
        if workflow_run.root_task_id is not None:
            self.whiteboard_store.update_task_whiteboard(
                task_id=workflow_run.root_task_id,
                updated_at=created_at + 1,
                section_updates={"Goal": goal},
            )
        review_thread = self.create_thread(
            thread_id=f"thread-review-orchestrator-{uuid4().hex[:8]}",
            kind="review",
            created_at=review_created_at,
            assigned_agent_id="orchestrator",
            summary=f"Acceptance review for {workflow_id}",
            parent_task_id=workflow_run.root_task_id,
            parent_thread_id=resolved_parent_thread_id,
        )
        self.append_event(
            kind="workflow_review_requested",
            summary=f"Requested orchestrator review for {workflow_id}",
            created_at=review_created_at,
            thread_id=review_thread.id,
            task_id=workflow_run.root_task_id,
        )
        run_view = self.describe_workflow_run(workflow_run.id)
        if run_view is None:
            raise ValueError(f"unknown workflow run: {workflow_run.id}")
        summary = await execute_defined_workflow(
            runtime=self,
            workflow_run=workflow_run,
            run_view=run_view,
            goal=goal,
            review_thread=review_thread,
            created_at=review_created_at + 1,
        )
        return {
            "workflow_run_id": summary.workflow_run_id,
            "workflow_id": summary.workflow_id,
            "status": summary.state,
            "current_step_index": summary.current_step_index,
            "last_thread_id": summary.last_thread_id,
            "review_thread_id": summary.review_thread_id,
            "review_summary": summary.review_summary or "",
            "review_accepted": summary.review_accepted,
            "artifact_id": summary.artifact_id,
            "blocked_summary": summary.blocked_summary or "",
        }

    async def _select_initial_workflow_step_groups(
        self,
        *,
        workflow_id: str,
        goal: str,
        created_at: int,
    ) -> tuple[tuple[str, ...], ...]:
        default_groups = self.workflow_step_groups(workflow_id)
        orchestration = self.workflow_orchestration(workflow_id)
        definition = self.registry.workflow_definitions[workflow_id]
        adaptive_staffing_enabled = bool(definition.metadata.get("adaptive_staffing", True))
        if not _workflow_supports_adaptive_staffing(
            adaptive_staffing_enabled=adaptive_staffing_enabled,
            orchestration=orchestration,
            default_step_groups=default_groups,
        ):
            return default_groups
        known_agents = tuple(
            agent_id
            for agent_id in self.list_agent_ids()
            if agent_id != "orchestrator"
        )
        parsed = await self._run_orchestrator_json_agent(
            agent_id=f"{workflow_id}-staffing-selector",
            name="Workflow Staffing Selector",
            description="Chooses the initial specialist sequence for a workflow run",
            instructions=_workflow_staffing_selector_instructions(
                workflow_id=workflow_id,
                orchestration=orchestration,
                available_agents=known_agents,
            ),
            prompt=_workflow_staffing_selector_prompt(
                workflow_id=workflow_id,
                orchestration=orchestration,
                goal=goal,
                default_step_groups=default_groups,
            ),
            created_at=created_at,
            session_id=f"workflow-staffing:{workflow_id}:{created_at}",
            failure_event_kind="workflow_staffing_selector_failed",
            failure_summary_prefix="Workflow staffing selector failed",
            invalid_event_kind="workflow_staffing_selector_invalid",
            invalid_summary_prefix="Workflow staffing selector returned invalid JSON",
        )
        if parsed is None:
            return default_groups
        try:
            selected_groups = _selected_workflow_step_groups_from_payload(
                workflow_id=workflow_id,
                payload=parsed,
                known_agents=known_agents,
            )
        except ValueError:
            return default_groups
        return selected_groups or default_groups

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

    async def _run_agent_turn(
        self,
        *,
        thread_id: str,
        agent_id: str,
        prompt_sender: str,
        reply_sender: str,
        body: str,
        created_at: int,
        record_prompt: bool = True,
    ) -> tuple[MessageRecord | None, MessageRecord | None]:
        stream = self._stream_agent_turn(
            thread_id=thread_id,
            agent_id=agent_id,
            prompt_sender=prompt_sender,
            reply_sender=reply_sender,
            body=body,
            created_at=created_at,
            record_prompt=record_prompt,
        )
        async for _ in stream:
            pass
        return await stream.get_final_response()

    async def _decide_orchestrator_turn(
        self,
        *,
        body: str,
        created_at: int,
    ) -> OrchestratorTurnDecision:
        parsed = await self._run_orchestrator_json_agent(
            agent_id="orchestrator-turn-planner",
            name="Orchestrator Turn Planner",
            description="Internal orchestration planner",
            instructions=_orchestrator_turn_planner_instructions(self),
            prompt=_orchestrator_turn_planner_prompt(self, body),
            created_at=created_at,
            session_id=f"main-turn-planner:{created_at}",
            failure_event_kind="orchestrator_planner_failed",
            failure_summary_prefix="Planner failed",
            invalid_event_kind="orchestrator_planner_invalid",
            invalid_summary_prefix="Planner returned invalid JSON",
        )
        if parsed is None:
            return OrchestratorTurnDecision(mode="act")
        mode = str(parsed.get("mode", "act")).strip().lower()
        if mode not in {"act", "delegate", "workflow"}:
            mode = "act"
        return OrchestratorTurnDecision(
            mode=mode,
            workflow_id=_optional_text(parsed.get("workflow_id")),
            agent_id=_optional_text(parsed.get("agent_id")),
            title=_optional_text(parsed.get("title")),
            request=_optional_text(parsed.get("request")),
            goal=_optional_text(parsed.get("goal")),
            deliverable_expected=_as_bool(parsed.get("deliverable_expected")),
        )

    async def _audit_orchestrator_delivery_turn(
        self,
        *,
        body: str,
        decision: OrchestratorTurnDecision,
        created_at: int,
    ) -> DeliveryAuditDecision:
        parsed = await self._run_orchestrator_json_agent(
            agent_id="orchestrator-delivery-auditor",
            name="Orchestrator Delivery Auditor",
            description="Internal delivery alignment audit",
            instructions=_delivery_audit_instructions(),
            prompt=_delivery_audit_prompt(self, body, decision),
            created_at=created_at,
            session_id=f"main-delivery-audit:{created_at}",
            failure_event_kind="orchestrator_delivery_audit_failed",
            failure_summary_prefix="Delivery audit failed",
        )
        if parsed is None:
            reconsider = False
            reason = ""
            deliverable_expected = decision.deliverable_expected
            if (
                decision.deliverable_expected
                and decision.mode == "workflow"
                and decision.workflow_id is not None
                and self._workflow_is_non_delivery(decision.workflow_id)
            ):
                reconsider = True
                reason = "Escalated a non-delivery workflow back to the orchestrator for delivery work"
            return DeliveryAuditDecision(
                deliverable_expected=deliverable_expected,
                reconsider=reconsider,
                reason=reason,
            )
        return DeliveryAuditDecision(
            deliverable_expected=_as_bool(parsed.get("deliverable_expected")),
            reconsider=_as_bool(parsed.get("reconsider")),
            reason=_optional_text(parsed.get("reason")) or "",
        )

    async def _reconsider_delivery_turn(
        self,
        *,
        body: str,
        created_at: int,
        reason: str,
    ) -> OrchestratorTurnDecision | None:
        parsed = await self._run_orchestrator_json_agent(
            agent_id="orchestrator-delivery-reconsideration-planner",
            name="Orchestrator Delivery Reconsideration Planner",
            description="Internal delivery replanner",
            instructions=_delivery_reconsideration_instructions(self, reason),
            prompt=_orchestrator_turn_planner_prompt(self, body),
            created_at=created_at,
            session_id=f"main-delivery-reconsideration:{created_at}",
            failure_event_kind="orchestrator_reconsideration_failed",
            failure_summary_prefix="Delivery reconsideration failed",
        )
        if parsed is None:
            return None
        mode = str(parsed.get("mode", "act")).strip().lower()
        if mode not in {"workflow", "delegate", "act"}:
            return None
        workflow_id = _optional_text(parsed.get("workflow_id"))
        if mode == "workflow":
            if workflow_id is None or self._workflow_is_non_delivery(workflow_id):
                return None
        agent_id = _optional_text(parsed.get("agent_id"))
        if mode == "delegate" and agent_id is None:
            return None
        return OrchestratorTurnDecision(
            mode=mode,
            workflow_id=workflow_id,
            agent_id=agent_id,
            title=_optional_text(parsed.get("title")),
            request=_optional_text(parsed.get("request")) or body,
            goal=_optional_text(parsed.get("goal")) or body,
            deliverable_expected=True,
        )

    async def _run_orchestrator_json_agent(
        self,
        *,
        agent_id: str,
        name: str,
        description: str,
        instructions: str,
        prompt: str,
        created_at: int,
        session_id: str,
        failure_event_kind: str,
        failure_summary_prefix: str,
        invalid_event_kind: str | None = None,
        invalid_summary_prefix: str | None = None,
    ) -> dict[str, object] | None:
        try:
            orchestrator = self.build_agent("orchestrator")
        except (KeyError, ValueError):
            return None
        client = getattr(orchestrator, "client", None)
        if client is None:
            return None
        try:
            internal_agent = Agent(
                client=client,
                id=agent_id,
                name=name,
                description=description,
                instructions=instructions,
            )
            response = await internal_agent.run(
                [
                    Message(
                        role="user",
                        text=prompt,
                        author_name="system",
                    )
                ],
                session=internal_agent.create_session(session_id=session_id),
            )
        except Exception as exc:
            self.append_event(
                kind=failure_event_kind,
                summary=f"{failure_summary_prefix}: {type(exc).__name__}: {exc}",
                created_at=created_at,
                thread_id=self.main_thread_id,
            )
            return None
        self.track_token_usage(response)
        try:
            return _parse_turn_decision_json(response.text.strip())
        except ValueError as exc:
            if invalid_event_kind is not None and invalid_summary_prefix is not None:
                self.append_event(
                    kind=invalid_event_kind,
                    summary=f"{invalid_summary_prefix}: {exc}",
                    created_at=created_at,
                    thread_id=self.main_thread_id,
                )
            return None

    def _workflow_is_non_delivery(self, workflow_id: str) -> bool:
        definition = self.registry.workflow_definitions.get(workflow_id)
        if definition is None:
            return False
        return not delivery_candidate_for_metadata(definition.metadata)

    async def generate_agent_text_without_tools(
        self,
        *,
        agent_id: str,
        body: str,
        created_at: int,
        thread_id: str | None = None,
        extra_instructions: str = "",
    ) -> str | None:
        target_thread_id = thread_id or self.main_thread_id
        try:
            base_agent = self.build_agent(agent_id)
        except (KeyError, ValueError) as exc:
            self.append_event(
                kind="agent_unavailable",
                summary=f"{agent_id} unavailable: {exc}",
                created_at=created_at + 1,
                thread_id=target_thread_id,
            )
            return None
        client = getattr(base_agent, "client", None)
        if client is None:
            _, reply_message = await self._run_agent_turn(
                thread_id=target_thread_id,
                agent_id=agent_id,
                prompt_sender="workflow",
                reply_sender=agent_id,
                body=body,
                created_at=created_at,
                record_prompt=False,
            )
            if reply_message is None:
                return None
            return self.conversation_store.read_message_body(reply_message).rstrip("\n")
        definition = self.registry.agent_definitions[agent_id]
        instructions = compose_instructions(definition)
        if extra_instructions.strip():
            instructions = f"{instructions}\n\n{extra_instructions.strip()}".strip()
        agent = Agent(
            client=client,
            id=f"{agent_id}-no-tools",
            name=str(definition.metadata.get("name", agent_id)),
            description=str(definition.metadata.get("role", agent_id)),
            instructions=instructions,
        )
        try:
            response = await agent.run(
                [
                    Message(
                        role="user",
                        text=body,
                        author_name="workflow",
                    )
                ],
                session=agent.create_session(session_id=f"{thread_id}:{agent_id}:no-tools:{created_at}"),
            )
        except Exception as exc:
            self.append_event(
                kind="agent_failed",
                summary=f"{agent_id} no-tool run failed: {type(exc).__name__}: {exc}",
                created_at=created_at + 1,
                thread_id=target_thread_id,
            )
            return None
        self.track_token_usage(response)
        response_text = response.text.strip()
        return response_text or None

    async def run_agent_turn_without_tools(
        self,
        *,
        thread_id: str,
        agent_id: str,
        prompt_sender: str,
        reply_sender: str,
        body: str,
        created_at: int,
        extra_instructions: str = "",
    ) -> tuple[MessageRecord, MessageRecord | None]:
        prompt_message = self.append_message_to_thread(
            thread_id=thread_id,
            message_id=f"message-{uuid4().hex}",
            sender=prompt_sender,
            kind="chat",
            body=body,
            created_at=created_at,
        )
        response_text = await self.generate_agent_text_without_tools(
            agent_id=agent_id,
            body=body,
            created_at=created_at,
            thread_id=thread_id,
            extra_instructions=extra_instructions,
        )
        if not response_text:
            return prompt_message, None
        reply_message = self.append_message_to_thread(
            thread_id=thread_id,
            message_id=f"message-{uuid4().hex}",
            sender=reply_sender,
            kind="chat",
            body=response_text,
            created_at=created_at + 1,
        )
        return prompt_message, reply_message

    def _format_workflow_summary(self, *, workflow_id: str, result: dict[str, object]) -> str:
        lines = [f"I used the `{workflow_id}` workflow."]
        workflow_run_id = result.get("workflow_run_id")
        if isinstance(workflow_run_id, str):
            team = self._workflow_team_summary(workflow_run_id)
            if team:
                lines.append(f"Team: {' -> '.join(team)}")
        review_summary = result.get("review_summary")
        review_accepted = result.get("review_accepted")
        if isinstance(review_summary, str) and review_summary.strip():
            lines.extend(["", review_summary.strip()])
        elif review_accepted is True:
            lines.extend(["", "ACCEPTED: The workflow completed and passed orchestrator review."])
        elif review_accepted is False:
            lines.extend(["", "REJECTED: The workflow finished without passing orchestrator review."])
        status = result.get("status")
        blocked_summary = result.get("blocked_summary")
        if isinstance(status, str) and status != "completed":
            lines.extend(["", f"Status: {status}"])
            if isinstance(blocked_summary, str) and blocked_summary.strip():
                lines.extend(["", f"Blocked: {blocked_summary.strip()}"])
        if isinstance(workflow_run_id, str):
            changed_files = self._workflow_changed_files(workflow_run_id)
            if changed_files:
                lines.extend(["", "Changed files:", *[f"- {path}" for path in changed_files]])
            checks = self._workflow_check_summaries(workflow_run_id)
            if checks:
                lines.extend(["", "Checks:", *[f"- {line}" for line in checks]])
        if len(lines) <= 3:
            created_files = self._workspace_file_list(limit=8)
            if created_files:
                lines.extend(["", "Workspace files:", *[f"- {path}" for path in created_files]])
        lines.extend(["", "If you want changes, tell me here and I’ll handle the team."])
        return "\n".join(lines)

    def _format_delegation_summary(self, *, agent_id: str, result: dict[str, object]) -> str:
        lines = [f"I delegated this to `{agent_id}`."]
        review_summary = result.get("review_summary")
        if isinstance(review_summary, str) and review_summary.strip():
            lines.extend(["", review_summary.strip()])
        response_text = result.get("result")
        if isinstance(response_text, str) and response_text.strip():
            lines.extend(["", response_text.strip()])
        status = result.get("status")
        if isinstance(status, str) and status != "completed":
            lines.extend(["", f"Status: {status}"])
        return "\n".join(lines)

    def _workspace_file_list(self, *, limit: int = 12) -> list[str]:
        root = self.paths.project_root.resolve()
        files = [
            str(path.relative_to(root))
            for path in sorted(root.rglob("*"))
            if path.is_file() and ".ergon.studio" not in path.parts
        ]
        return files[:limit]

    def _workflow_team_summary(self, workflow_run_id: str) -> list[str]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []
        labels: list[str] = []
        for step in run_view.steps:
            participants = [
                thread.assigned_agent_id or thread.summary or thread.id
                for thread in step.threads
            ]
            if participants:
                labels.append(" + ".join(participants))
        return labels

    def _workflow_changed_files(self, workflow_run_id: str) -> list[str]:
        paths: set[str] = set()
        for tool_call in self.list_tool_calls_for_workflow_run(workflow_run_id):
            if tool_call.status != "completed":
                continue
            if tool_call.tool_name not in {"write_file", "patch_file"}:
                continue
            path = self._tool_call_value(tool_call, key="path")
            if isinstance(path, str) and path.strip():
                paths.add(path.strip())
        return sorted(paths)[:8]

    def _workflow_check_summaries(self, workflow_run_id: str) -> list[str]:
        summaries: list[str] = []
        for command_run in self.list_command_runs_for_workflow_run(workflow_run_id):
            if command_run.status != "completed" or command_run.exit_code != 0:
                continue
            command = command_run.command.strip()
            if not command:
                continue
            summaries.append(command)
        return summaries[-3:]

    def _tool_call_value(self, tool_call: ToolCallRecord, *, key: str) -> object | None:
        for payload_text in (
            self.tool_call_store.read_response(tool_call) if tool_call.response_path is not None else "",
            self.tool_call_store.read_request(tool_call),
        ):
            if not payload_text.strip():
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            return payload.get(key)
        return None

    def run_workspace_command(
        self,
        command: str,
        timeout: int = 60,
        *,
        created_at: int | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        require_approval: bool = True,
    ) -> dict[str, int | str]:
        if created_at is None:
            created_at = int(time.time())

        context = current_tool_execution_context()
        session_id = context.session_id if context is not None else self.main_session_id
        resolved_thread_id = thread_id if thread_id is not None else (context.thread_id if context is not None else None)
        resolved_task_id = task_id if task_id is not None else (context.task_id if context is not None else None)
        resolved_agent_id = agent_id if agent_id is not None else (context.agent_id if context is not None else None)
        workspace_root = self.paths.project_root.resolve()
        command_budget = self._agent_thread_command_budget(
            thread_id=resolved_thread_id,
            agent_id=resolved_agent_id,
        )
        if command_budget is not None:
            prior_runs = [
                command_run
                for command_run in self.list_command_runs()
                if command_run.thread_id == resolved_thread_id
                and command_run.agent_id == resolved_agent_id
                and command_run.status not in {"awaiting_approval", "budget_exhausted"}
            ]
            if len(prior_runs) >= command_budget:
                stderr = (
                    f"Command budget exhausted for {resolved_agent_id} in {resolved_thread_id}. "
                    "Reuse the existing verification evidence instead of running more commands."
                )
                previous_budget_run = self._latest_budget_exhausted_command_run(
                    thread_id=resolved_thread_id,
                    agent_id=resolved_agent_id,
                )
                if previous_budget_run is not None:
                    return {
                        "command": command,
                        "cwd": str(workspace_root),
                        "exit_code": previous_budget_run.exit_code,
                        "stdout": "",
                        "stderr": stderr,
                        "status": "budget_exhausted",
                        "command_run_id": previous_budget_run.id,
                    }
                command_run = self.command_store.create_command_run(
                    session_id=session_id,
                    command_run_id=f"command-run-{uuid4().hex[:8]}",
                    command=command,
                    cwd=str(workspace_root),
                    exit_code=1,
                    status="budget_exhausted",
                    output_content=_render_command_output(
                        command=command,
                        cwd=str(workspace_root),
                        exit_code=1,
                        status="budget_exhausted",
                        timeout=timeout,
                        stdout="",
                        stderr=stderr,
                        thread_id=resolved_thread_id,
                        task_id=resolved_task_id,
                        agent_id=resolved_agent_id,
                    ),
                    created_at=created_at,
                    thread_id=resolved_thread_id,
                    task_id=resolved_task_id,
                    agent_id=resolved_agent_id,
                )
                runner = resolved_agent_id or "user"
                self.append_event(
                    kind="command_budget_exhausted",
                    summary=f"{runner} hit the command budget in `{resolved_thread_id}`",
                    created_at=created_at,
                    thread_id=resolved_thread_id,
                    task_id=resolved_task_id,
                )
                return {
                    "command": command,
                    "cwd": str(workspace_root),
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": stderr,
                    "status": "budget_exhausted",
                    "command_run_id": command_run.id,
                }

        if require_approval and self._approval_mode_for_action("run_command") in {"ask", "block"}:
            approval = self.request_approval(
                approval_id=f"approval-{uuid4().hex[:8]}",
                requester=resolved_agent_id or "user",
                action="run_command",
                risk_class="high",
                reason=f"Run command in workspace: {command}",
                created_at=created_at,
                thread_id=resolved_thread_id,
                task_id=resolved_task_id,
                payload={
                    "command": command,
                    "timeout": timeout,
                    "cwd": str(workspace_root),
                    "thread_id": resolved_thread_id,
                    "task_id": resolved_task_id,
                    "agent_id": resolved_agent_id or "user",
                },
            )
            return {
                "command": command,
                "cwd": str(workspace_root),
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "status": "awaiting_approval",
                "approval_id": approval.id,
            }

        try:
            completed = subprocess.run(
                command,
                cwd=workspace_root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            status = "completed"
        except subprocess.TimeoutExpired as exc:
            exit_code = -1
            stdout = exc.stdout or ""
            stderr_lines = [exc.stderr or ""]
            stderr_lines.append(f"Timed out after {timeout} seconds.")
            stderr = "\n".join(line for line in stderr_lines if line)
            status = "timeout"

        command_run = self.command_store.create_command_run(
            session_id=session_id,
            command_run_id=f"command-run-{uuid4().hex[:8]}",
            command=command,
            cwd=str(workspace_root),
            exit_code=exit_code,
            status=status,
            output_content=_render_command_output(
                command=command,
                cwd=str(workspace_root),
                exit_code=exit_code,
                status=status,
                timeout=timeout,
                stdout=stdout,
                stderr=stderr,
                thread_id=resolved_thread_id,
                task_id=resolved_task_id,
                agent_id=resolved_agent_id,
            ),
            created_at=created_at,
            thread_id=resolved_thread_id,
            task_id=resolved_task_id,
            agent_id=resolved_agent_id,
        )
        runner = resolved_agent_id or "user"
        self.append_event(
            kind="command_run",
            summary=f"{runner} ran `{command}` [{status}]",
            created_at=created_at,
            thread_id=resolved_thread_id,
            task_id=resolved_task_id,
        )
        return {
            "command": command,
            "cwd": str(workspace_root),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "status": status,
            "command_run_id": command_run.id,
        }

    def _agent_thread_command_budget(
        self,
        *,
        thread_id: str | None,
        agent_id: str | None,
    ) -> int | None:
        if thread_id is None or agent_id is None:
            return None
        thread = self.get_thread(thread_id)
        if thread is None:
            return None
        if thread.kind not in {"agent_direct", "group_workroom"}:
            return None
        budgets = {
            "coder": 4,
            "fixer": 4,
            "tester": 6,
            "reviewer": 3,
        }
        return budgets.get(agent_id)

    def _latest_budget_exhausted_command_run(
        self,
        *,
        thread_id: str | None,
        agent_id: str | None,
    ) -> CommandRunRecord | None:
        if thread_id is None or agent_id is None:
            return None
        matches = [
            command_run
            for command_run in self.list_command_runs()
            if command_run.thread_id == thread_id
            and command_run.agent_id == agent_id
            and command_run.status == "budget_exhausted"
        ]
        if not matches:
            return None
        return max(matches, key=lambda record: record.created_at)

    def _workflow_threads_for_run(self, workflow_run: WorkflowRunRecord) -> list[ThreadRecord]:
        if workflow_run.root_task_id is None:
            return []
        child_tasks = [
            task
            for task in self.list_tasks()
            if task.parent_task_id == workflow_run.root_task_id
        ]
        child_tasks_by_id = {task.id: task for task in child_tasks}
        threads = [
            thread
            for thread in self.list_threads()
            if thread.parent_task_id in child_tasks_by_id
        ]
        return sorted(
            threads,
            key=lambda thread: (
                child_tasks_by_id[thread.parent_task_id].created_at if thread.parent_task_id else 0,
                thread.created_at,
                thread.id,
            ),
        )

    def _workflow_prompt_for_step(
        self,
        *,
        workflow_run: WorkflowRunRecord,
        steps: list[WorkflowRunStepView],
        next_index: int,
    ) -> str:
        if next_index == 0:
            goal = self.latest_main_user_message_body()
            if goal:
                return f"Workflow kickoff: {workflow_run.workflow_id}\n\nGoal:\n{goal}"
            return f"Workflow kickoff: {workflow_run.workflow_id}"

        previous_step = steps[next_index - 1]
        previous_outputs: list[str] = []
        for thread in previous_step.threads:
            previous_output = self._latest_agent_message_body(thread.id)
            if not previous_output:
                continue
            previous_outputs.append(
                f"{thread.assigned_agent_id or 'unknown'}:\n{previous_output}"
            )
        if previous_outputs:
            return (
                f"Continue workflow: {workflow_run.workflow_id}\n\n"
                "Previous step outputs:\n"
                + "\n\n".join(previous_outputs)
            )
        return f"Continue workflow: {workflow_run.workflow_id}"

    def _latest_agent_message_body(self, thread_id: str) -> str | None:
        thread = self.get_thread(thread_id)
        messages = self.list_thread_messages(thread_id)
        for message in reversed(messages):
            if thread is not None and thread.assigned_agent_id is not None and message.sender != thread.assigned_agent_id:
                continue
            return self.conversation_store.read_message_body(message).rstrip("\n")
        return None

    def _append_workflow_completion_summary(
        self,
        *,
        workflow_run: WorkflowRunRecord,
        thread: ThreadRecord,
        created_at: int,
    ) -> MessageRecord:
        final_output = self._latest_agent_message_body(thread.id)
        summary_lines = [f"Workflow complete: {workflow_run.workflow_id}"]
        if thread.assigned_agent_id is not None and final_output:
            summary_lines.extend(
                [
                    "",
                    f"Final output from {thread.assigned_agent_id}:",
                    final_output,
                ]
            )
        return self.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="orchestrator",
            kind="status_update",
            body="\n".join(summary_lines),
            created_at=created_at,
        )

    def _create_workflow_completion_artifact(
        self,
        *,
        workflow_run: WorkflowRunRecord,
        thread: ThreadRecord,
        created_at: int,
    ) -> ArtifactRecord:
        final_output = self._latest_agent_message_body(thread.id) or "No final output captured."
        lines = [
            f"# Workflow Report: {workflow_run.workflow_id}",
            "",
            f"- Run ID: {workflow_run.id}",
            f"- Status: {workflow_run.state}",
        ]
        if thread.assigned_agent_id is not None:
            lines.append(f"- Final Agent: {thread.assigned_agent_id}")
        lines.extend(
            [
                "",
                "## Final Output",
                final_output,
            ]
        )
        return self.create_artifact(
            artifact_id=f"artifact-{uuid4().hex[:8]}",
            kind="workflow-report",
            title=f"Workflow Report: {workflow_run.workflow_id}",
            content="\n".join(lines),
            created_at=created_at,
            thread_id=thread.id,
            task_id=workflow_run.root_task_id,
        )

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
        self.whiteboard_store.ensure_task_whiteboard(
            task_id=task_id,
            title=title,
            updated_at=created_at,
            parent_task_id=parent_task_id,
            template_task_id=parent_task_id,
        )
        self.append_event(
            kind="task_created",
            summary=f"Created task {task_id}",
            created_at=created_at,
            task_id=task_id,
        )
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self.task_store.get_task(task_id)

    def update_task_state(self, *, task_id: str, state: str, updated_at: int) -> TaskRecord:
        task = self.task_store.update_task_state(
            task_id=task_id,
            state=state,
            updated_at=updated_at,
        )
        self.append_event(
            kind="task_updated",
            summary=f"Updated task {task_id} to {state}",
            created_at=updated_at,
            task_id=task_id,
        )
        return task

    def start_workflow_run(
        self,
        *,
        workflow_id: str,
        created_at: int,
        goal: str | None = None,
        step_groups: tuple[tuple[str, ...], ...] | None = None,
    ) -> tuple[WorkflowRunRecord, list[ThreadRecord]]:
        goal = goal or self.latest_main_user_message_body()
        compiled = compile_workflow_definition(self.registry.workflow_definitions[workflow_id])
        default_step_groups = self.workflow_step_groups(workflow_id)
        resolved_step_groups = default_step_groups if step_groups is None else _validate_runtime_step_groups(
            workflow_id=workflow_id,
            step_groups=step_groups,
            known_agents=tuple(self.registry.agent_definitions.keys()),
        )
        orchestration = self.workflow_orchestration(workflow_id)
        root_task = self.create_task(
            task_id=f"task-{uuid4().hex[:8]}",
            title=f"Workflow: {workflow_id}",
            state="in_progress",
            created_at=created_at,
        )
        self.whiteboard_store.update_task_whiteboard(
            task_id=root_task.id,
            updated_at=created_at,
            section_updates={
                "Goal": goal or f"Workflow: {workflow_id}",
                "Plan": self._render_workflow_plan(resolved_step_groups),
                "Acceptance Criteria": self._workflow_acceptance_criteria(workflow_id),
            },
        )
        workflow_run = self.workflow_store.create_workflow_run(
            session_id=self.main_session_id,
            workflow_run_id=f"workflow-run-{uuid4().hex[:8]}",
            workflow_id=workflow_id,
            state="running",
            created_at=created_at + 1,
            root_task_id=root_task.id,
        )
        self.create_artifact(
            artifact_id=f"artifact-{uuid4().hex[:8]}",
            kind="workflow-graph",
            title=f"Workflow Graph: {workflow_id}",
            content=(
                compiled.to_mermaid()
                if resolved_step_groups == default_step_groups
                else _render_runtime_workflow_graph(
                    workflow_id=workflow_id,
                    orchestration=orchestration,
                    step_groups=resolved_step_groups,
                )
            ),
            created_at=created_at + 1,
            task_id=root_task.id,
        )
        self.append_event(
            kind="workflow_started",
            summary=f"Started workflow {workflow_id}",
            created_at=created_at + 1,
            task_id=root_task.id,
        )
        if orchestration in {"group_chat", "magentic", "handoff"}:
            threads = self._append_group_workroom_step(
                workflow_id=workflow_id,
                root_task_id=root_task.id,
                participants=tuple(agent_id for group in resolved_step_groups for agent_id in group),
                created_at=created_at + 2,
            )
        else:
            threads = self._append_workflow_steps(
                workflow_id=workflow_id,
                root_task_id=root_task.id,
                step_groups=resolved_step_groups,
                created_at=created_at + 2,
            )
        return workflow_run, threads

    def _render_workflow_plan(self, step_groups: tuple[tuple[str, ...], ...]) -> str:
        if not step_groups:
            return "Handle the work directly and produce a final answer."
        lines: list[str] = []
        for index, group in enumerate(step_groups, start=1):
            participants = " + ".join(group)
            lines.append(f"{index}. {participants}")
        return "\n".join(lines)

    def _workflow_acceptance_criteria(self, workflow_id: str) -> str:
        acceptance_mode = acceptance_mode_for_metadata(
            self.registry.workflow_definitions[workflow_id].metadata
        )
        return acceptance_criteria_for_mode(acceptance_mode)

    def _workflow_followup_step_groups(
        self,
        *,
        workflow_id: str,
        metadata_key: str,
    ) -> tuple[tuple[str, ...], ...]:
        definition = self.registry.workflow_definitions[workflow_id]
        return step_groups_for_metadata(
            workflow_id=workflow_id,
            metadata=definition.metadata,
            metadata_key=metadata_key,
        )

    def request_workflow_fix_cycle(
        self,
        *,
        workflow_run_id: str,
        created_at: int,
    ) -> tuple[WorkflowRunRecord, list[ThreadRecord]]:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")
        repair_step_groups = self._workflow_followup_step_groups(
            workflow_id=workflow_run.workflow_id,
            metadata_key="repair_step_groups",
        ) or (("fixer",), ("reviewer",))
        return self.request_workflow_followup_cycle(
            workflow_run_id=workflow_run_id,
            created_at=created_at,
            step_groups=repair_step_groups,
            state="repairing",
            event_kind="workflow_fix_cycle_requested",
            event_summary="Requested fix cycle",
        )

    def request_workflow_followup_cycle(
        self,
        *,
        workflow_run_id: str,
        created_at: int,
        step_groups: tuple[tuple[str, ...], ...],
        state: str,
        event_kind: str,
        event_summary: str,
    ) -> tuple[WorkflowRunRecord, list[ThreadRecord]]:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")

        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")
        if workflow_run.current_step_index < len(run_view.steps):
            raise ValueError("workflow run still has pending steps")
        if workflow_run.root_task_id is None:
            raise ValueError("workflow run has no root task")

        threads = self._append_workflow_steps(
            workflow_id=workflow_run.workflow_id,
            root_task_id=workflow_run.root_task_id,
            step_groups=step_groups,
            created_at=created_at,
        )
        updated = WorkflowRunRecord(
            id=workflow_run.id,
            session_id=workflow_run.session_id,
            workflow_id=workflow_run.workflow_id,
            state=state,
            created_at=workflow_run.created_at,
            updated_at=created_at + 1,
            root_task_id=workflow_run.root_task_id,
            current_step_index=len(run_view.steps),
            last_thread_id=workflow_run.last_thread_id,
        )
        self.workflow_store.update_workflow_run(updated)
        self.update_task_state(
            task_id=workflow_run.root_task_id,
            state="in_progress",
            updated_at=created_at + 1,
        )
        self.append_event(
            kind=event_kind,
            summary=event_summary,
            created_at=created_at + 1,
            task_id=workflow_run.root_task_id,
        )
        return updated, threads

    async def advance_workflow_run(
        self,
        *,
        workflow_run_id: str,
        created_at: int,
    ) -> tuple[WorkflowRunRecord, ThreadRecord | None, MessageRecord | None]:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")

        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")
        steps = list(run_view.steps)
        next_index = workflow_run.current_step_index
        if next_index >= len(steps):
            completed = WorkflowRunRecord(
                id=workflow_run.id,
                session_id=workflow_run.session_id,
                workflow_id=workflow_run.workflow_id,
                state="completed",
                created_at=workflow_run.created_at,
                updated_at=created_at,
                root_task_id=workflow_run.root_task_id,
                current_step_index=workflow_run.current_step_index,
                last_thread_id=workflow_run.last_thread_id,
            )
            self.workflow_store.update_workflow_run(completed)
            self.append_event(
                kind="workflow_completed",
                summary=f"Completed workflow {workflow_run.workflow_id}",
                created_at=created_at,
                task_id=workflow_run.root_task_id,
            )
            return completed, None, None

        step = steps[next_index]
        if step.task.id is not None:
            self.update_task_state(
                task_id=step.task.id,
                state="in_progress",
                updated_at=created_at,
            )
        prompt = self._workflow_prompt_for_step(
            workflow_run=workflow_run,
            steps=steps,
            next_index=next_index,
        )
        results = await asyncio.gather(
            *[
                self.send_message_to_agent_thread(
                    thread_id=thread.id,
                    body=prompt,
                    created_at=created_at + offset,
                )
                for offset, thread in enumerate(step.threads)
            ]
        )
        blocked_thread = None
        for thread, (_, reply) in zip(step.threads, results):
            if reply is None:
                blocked_thread = thread
                break
        if blocked_thread is not None:
            blocked = WorkflowRunRecord(
                id=workflow_run.id,
                session_id=workflow_run.session_id,
                workflow_id=workflow_run.workflow_id,
                state="blocked",
                created_at=workflow_run.created_at,
                updated_at=created_at + 1,
                root_task_id=workflow_run.root_task_id,
                current_step_index=workflow_run.current_step_index,
                last_thread_id=blocked_thread.id,
            )
            self.workflow_store.update_workflow_run(blocked)
            self.update_task_state(
                task_id=step.task.id,
                state="blocked",
                updated_at=created_at + 1,
            )
            self.append_event(
                kind="workflow_blocked",
                summary=f"Workflow {workflow_run.workflow_id} blocked at {blocked_thread.assigned_agent_id}",
                created_at=created_at + 1,
                thread_id=blocked_thread.id,
                task_id=workflow_run.root_task_id,
            )
            return blocked, blocked_thread, None

        reply_message = None
        for _, reply in reversed(results):
            if reply is not None:
                reply_message = reply
                break

        updated = WorkflowRunRecord(
            id=workflow_run.id,
            session_id=workflow_run.session_id,
            workflow_id=workflow_run.workflow_id,
            state="completed" if next_index + 1 >= len(steps) else "running",
            created_at=workflow_run.created_at,
            updated_at=created_at + 1,
            root_task_id=workflow_run.root_task_id,
            current_step_index=next_index + 1,
            last_thread_id=step.threads[-1].id if step.threads else workflow_run.last_thread_id,
        )
        self.workflow_store.update_workflow_run(updated)
        self.update_task_state(
            task_id=step.task.id,
            state="completed",
            updated_at=created_at + 1,
        )
        participant_text = ", ".join(thread.assigned_agent_id or "unknown" for thread in step.threads)
        self.append_event(
            kind="workflow_advanced",
            summary=f"Advanced workflow {workflow_run.workflow_id} through {participant_text}",
            created_at=created_at + 1,
            thread_id=step.threads[-1].id if step.threads else None,
            task_id=workflow_run.root_task_id,
        )
        if updated.state == "completed":
            if workflow_run.root_task_id is not None:
                self.update_task_state(
                    task_id=workflow_run.root_task_id,
                    state="completed",
                    updated_at=created_at + 2,
                )
            final_thread = step.threads[-1] if step.threads else None
            if final_thread is None:
                return updated, None, reply_message
            self._create_workflow_completion_artifact(
                workflow_run=updated,
                thread=final_thread,
                created_at=created_at + 2,
            )
            self._append_workflow_completion_summary(
                workflow_run=updated,
                thread=final_thread,
                created_at=created_at + 3,
            )
            self.append_event(
                kind="workflow_completed",
                summary=f"Completed workflow {workflow_run.workflow_id}",
                created_at=created_at + 4,
                task_id=workflow_run.root_task_id,
            )
        return updated, (step.threads[-1] if step.threads else None), reply_message

    def _append_workflow_steps(
        self,
        *,
        workflow_id: str,
        root_task_id: str,
        step_groups: tuple[tuple[str, ...], ...],
        created_at: int,
    ) -> list[ThreadRecord]:
        threads: list[ThreadRecord] = []
        thread_offset = 0
        for offset, group in enumerate(step_groups):
            group_label = _workflow_group_label(group)
            child_task = self.create_task(
                task_id=f"task-{uuid4().hex[:8]}",
                title=f"{workflow_id}: {group_label}",
                state="planned",
                created_at=created_at + offset,
                parent_task_id=root_task_id,
            )
            for agent_id in group:
                thread = self.create_agent_thread(
                    agent_id=agent_id,
                    created_at=created_at + len(step_groups) + thread_offset,
                    parent_task_id=child_task.id,
                )
                threads.append(thread)
                thread_offset += 1
        return threads

    def _append_group_workroom_step(
        self,
        *,
        workflow_id: str,
        root_task_id: str,
        participants: tuple[str, ...],
        created_at: int,
    ) -> list[ThreadRecord]:
        label = " + ".join(participants) if participants else "workroom"
        child_task = self.create_task(
            task_id=f"task-{uuid4().hex[:8]}",
            title=f"{workflow_id}: {label}",
            state="planned",
            created_at=created_at,
            parent_task_id=root_task_id,
        )
        workroom = self.create_thread(
            thread_id=f"thread-workroom-{uuid4().hex[:8]}",
            kind="group_workroom",
            created_at=created_at + 1,
            summary=f"Workroom: {label}",
            parent_task_id=child_task.id,
        )
        return [workroom]

    def append_event(
        self,
        *,
        kind: str,
        summary: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> EventRecord:
        record = self.event_store.append_event(
            session_id=self.main_session_id,
            event_id=f"event-{uuid4().hex}",
            kind=kind,
            summary=summary,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        return record

    def request_approval(
        self,
        *,
        approval_id: str,
        requester: str,
        action: str,
        risk_class: str,
        reason: str,
        created_at: int,
        thread_id: str | None = None,
        task_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> ApprovalRecord:
        approval = self.approval_store.request_approval(
            session_id=self.main_session_id,
            approval_id=approval_id,
            requester=requester,
            action=action,
            risk_class=risk_class,
            reason=reason,
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
            payload=payload,
        )
        self.append_event(
            kind="approval_requested",
            summary=f"{requester} requested approval for {action}",
            created_at=created_at,
            thread_id=thread_id,
            task_id=task_id,
        )
        return approval

    def resolve_approval(
        self,
        *,
        approval_id: str,
        status: str,
        created_at: int,
    ) -> ApprovalRecord:
        approval = self.approval_store.update_approval_status(
            approval_id=approval_id,
            status=status,
        )
        self.append_event(
            kind=f"approval_{status}",
            summary=f"{status.capitalize()} approval {approval_id} for {approval.action}",
            created_at=created_at,
        )
        if status == "approved":
            try:
                self._execute_approved_action(approval=approval, created_at=created_at + 1)
            except Exception as exc:
                self.append_event(
                    kind="approved_action_failed",
                    summary=f"Approved action {approval.action} failed: {type(exc).__name__}: {exc}",
                    created_at=created_at + 1,
                    thread_id=approval.thread_id,
                    task_id=approval.task_id,
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
        source: str | None = None,
        confidence: float | None = None,
        tags: tuple[str, ...] = (),
        last_used_at: int | None = None,
    ) -> MemoryFactRecord:
        fact = self.memory_store.add_fact(
            fact_id=fact_id,
            scope=scope,
            kind=kind,
            content=content,
            created_at=created_at,
            source=source,
            confidence=confidence,
            tags=tags,
            last_used_at=last_used_at,
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
        session = self.current_session()
        self.conversation_store.ensure_session(
            self.main_session_id,
            created_at=0,
            title=session.title if session is not None else None,
        )
        self.conversation_store.ensure_thread(
            session_id=self.main_session_id,
            thread_id=self.main_thread_id,
            kind="main",
            created_at=0,
            assigned_agent_id=None,
        )

    # ── Context compaction ──

    def context_window_size(self) -> int:
        """Return the context window size for the orchestrator's provider."""
        provider_name = self.assigned_provider_name("orchestrator")
        if provider_name is None:
            return _DEFAULT_CONTEXT_LENGTH
        provider = self.provider_details(provider_name)
        if provider is None:
            return _DEFAULT_CONTEXT_LENGTH
        ctx = provider.get("context_length")
        if isinstance(ctx, int) and ctx > 0:
            return ctx
        return _DEFAULT_CONTEXT_LENGTH

    def accumulated_tokens(self) -> int:
        return self._accumulated_tokens

    def track_token_usage(self, response) -> None:
        """Accumulate token usage from an AgentResponse."""
        usage = getattr(response, "usage_details", None)
        if usage is None:
            return
        total = usage.get("total_token_count")
        if isinstance(total, int):
            object.__setattr__(self, "_accumulated_tokens", self._accumulated_tokens + total)
            return
        input_t = usage.get("input_token_count") or 0
        output_t = usage.get("output_token_count") or 0
        if input_t or output_t:
            object.__setattr__(self, "_accumulated_tokens", self._accumulated_tokens + input_t + output_t)

    def needs_compaction(self) -> bool:
        """Check if accumulated tokens exceed the compaction threshold."""
        if self._compaction_failure_count >= _COMPACTION_MAX_FAILURES:
            return False
        ctx = self.context_window_size()
        return self._accumulated_tokens >= int(ctx * _COMPACTION_THRESHOLD)

    async def auto_compact(self, *, focus: str | None = None, created_at: int | None = None) -> str | None:
        """Run automatic context compaction. Returns the summary or None on failure.

        Compaction summarizes older messages and DELETES them from the
        conversation store, keeping only the last few exchanges plus the
        summary. This actually reduces context window usage.
        """
        if created_at is None:
            created_at = int(time.time())
        self.append_event(
            kind="compaction_started",
            summary=f"Auto-compaction triggered at {self._accumulated_tokens} tokens",
            created_at=created_at,
            thread_id=self.main_thread_id,
        )

        messages = self.list_main_messages()
        if len(messages) < 6:
            return None

        # Keep the last 4 messages (recent context) — compact everything before
        keep_count = 4
        messages_to_compact = messages[:-keep_count]
        if not messages_to_compact:
            return None

        # Build the conversation text from messages being compacted only
        conversation_parts: list[str] = []
        for msg in messages_to_compact:
            body = self.conversation_store.read_message_body(msg).rstrip("\n")
            conversation_parts.append(f"[{msg.sender}] {body}")
        full_text = "\n\n".join(conversation_parts)

        # Truncate if the conversation text itself is too large for the
        # compaction request (use ~40% of context window as limit)
        ctx = self.context_window_size()
        max_chars = int(ctx * 0.4 * 4)  # ~40% of context, ~4 chars per token
        if len(full_text) > max_chars:
            full_text = full_text[-max_chars:]

        focus_hint = f"\nFocus especially on: {focus}" if focus else ""
        prompt = f"{_COMPACTION_PROMPT}{focus_hint}\n\nConversation:\n{full_text}"

        try:
            summary = await self.generate_agent_text_without_tools(
                agent_id="orchestrator",
                body=prompt,
                created_at=created_at + 1,
            )
        except Exception as exc:
            object.__setattr__(self, "_compaction_failure_count", self._compaction_failure_count + 1)
            self.append_event(
                kind="compaction_failed",
                summary=f"Compaction failed ({self._compaction_failure_count}/{_COMPACTION_MAX_FAILURES}): {exc}",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
            return None

        if not summary or not summary.strip():
            # Empty output is not a hard failure — don't increment circuit breaker
            self.append_event(
                kind="compaction_empty",
                summary="Compaction returned empty output",
                created_at=created_at + 1,
                thread_id=self.main_thread_id,
            )
            return None

        # Delete the old messages that were compacted
        self.conversation_store.delete_messages(
            self.main_thread_id, messages_to_compact,
        )

        # Inject summary as the new earliest message
        self.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="system",
            kind="compaction_summary",
            body=f"[Context compacted — {len(messages_to_compact)} messages summarized]\n\n{summary.strip()}",
            created_at=created_at + 2,
        )

        object.__setattr__(self, "_accumulated_tokens", 0)
        object.__setattr__(self, "_compaction_failure_count", 0)

        self.append_event(
            kind="compaction_completed",
            summary=f"Compacted {len(messages_to_compact)} messages, kept last {keep_count}",
            created_at=created_at + 2,
            thread_id=self.main_thread_id,
        )
        return summary.strip()

    def _approval_mode_for_action(self, action: str) -> str:
        approvals = self.registry.config.get("approvals", {})
        if not isinstance(approvals, dict):
            return "auto"
        value = approvals.get(action, approvals.get("default", "auto"))
        if value not in {"auto", "notify", "ask", "block"}:
            return "auto"
        return str(value)

    def _execute_approved_action(self, *, approval: ApprovalRecord, created_at: int) -> None:
        payload = self.approval_store.read_payload(approval)
        if payload is None:
            return
        thread_id = payload.get("thread_id") if isinstance(payload.get("thread_id"), str) else None
        task_id = payload.get("task_id") if isinstance(payload.get("task_id"), str) else None
        agent_id = payload.get("agent_id") if isinstance(payload.get("agent_id"), str) else None

        if approval.action == "run_command":
            command = payload.get("command")
            if not isinstance(command, str) or not command:
                return
            timeout = payload.get("timeout", 60)
            if type(timeout) is not int:
                timeout = 60
            self.run_workspace_command(
                command,
                timeout=timeout,
                created_at=created_at,
                thread_id=thread_id,
                task_id=task_id,
                agent_id=agent_id,
                require_approval=False,
            )
            return

        if approval.action == "write_file":
            path = payload.get("path")
            content = payload.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                return
            self.write_workspace_file(
                path,
                content,
                created_at=created_at,
                thread_id=thread_id,
                task_id=task_id,
                agent_id=agent_id,
                require_approval=False,
            )
            return

        if approval.action == "patch_file":
            path = payload.get("path")
            old_text = payload.get("old_text")
            new_text = payload.get("new_text")
            if not isinstance(path, str) or not isinstance(old_text, str) or not isinstance(new_text, str):
                return
            self.patch_workspace_file(
                path,
                old_text,
                new_text,
                created_at=created_at,
                thread_id=thread_id,
                task_id=task_id,
                agent_id=agent_id,
                require_approval=False,
            )

    def _resolve_workspace_path(self, path: str) -> Path:
        workspace_root = self.paths.project_root.resolve()
        candidate = (workspace_root / path).resolve()
        if workspace_root not in (candidate, *candidate.parents):
            raise ValueError("path is outside the project workspace")
        return candidate


def load_runtime(
    project_root: Path,
    home_dir: Path,
    *,
    session_id: str | None = None,
    create_session: bool = False,
    session_title: str | None = None,
) -> RuntimeContext:
    paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
    registry = load_registry(paths)
    session_store = SessionStore(paths)
    agent_session_store = AgentSessionStore(paths)
    conversation_store = ConversationStore(paths)
    task_store = TaskStore(paths)
    workflow_store = WorkflowStore(paths)
    event_store = EventStore(paths)
    approval_store = ApprovalStore(paths)
    memory_store = MemoryStore(paths)
    whiteboard_store = WhiteboardStore(paths)
    artifact_store = ArtifactStore(paths)
    command_store = CommandStore(paths)
    tool_call_store = ToolCallStore(paths)
    retrieval_index = RetrievalIndex(paths)
    resolved_session = _resolve_runtime_session(
        session_store=session_store,
        session_id=session_id,
        create_session=create_session,
        session_title=session_title,
    )
    runtime = RuntimeContext(
        paths=paths,
        registry=registry,
        tool_registry={},
        session_store=session_store,
        agent_session_store=agent_session_store,
        conversation_store=conversation_store,
        task_store=task_store,
        workflow_store=workflow_store,
        event_store=event_store,
        approval_store=approval_store,
        memory_store=memory_store,
        whiteboard_store=whiteboard_store,
        artifact_store=artifact_store,
        command_store=command_store,
        tool_call_store=tool_call_store,
        retrieval_index=retrieval_index,
        live_state=LiveRuntimeState(),
        main_session_id=resolved_session.id,
        main_thread_id=_main_thread_id_for_session(resolved_session.id),
        _accumulated_tokens=0,
        _compaction_failure_count=0,
    )
    object.__setattr__(
        runtime,
        "tool_registry",
        build_workspace_tool_registry(
            paths.project_root,
            run_command_handler=runtime.run_workspace_command,
            write_file_handler=runtime.write_workspace_file,
            patch_file_handler=runtime.patch_workspace_file,
            list_agents_handler=runtime.list_agent_summaries,
            describe_agent_handler=runtime.describe_agent_definition,
            list_workflows_handler=runtime.list_workflow_summaries,
            describe_workflow_handler=runtime.describe_workflow_definition,
            delegate_to_agent_handler=lambda agent_id, request, title=None: runtime.delegate_to_agent(
                agent_id=agent_id,
                request=request,
                title=title,
            ),
            run_workflow_handler=lambda workflow_id, goal: runtime.run_workflow(
                workflow_id=workflow_id,
                goal=goal,
            ),
        ),
    )
    runtime.ensure_main_conversation()
    return runtime


def _resolve_runtime_session(
    *,
    session_store: SessionStore,
    session_id: str | None,
    create_session: bool,
    session_title: str | None,
) -> SessionRecord:
    now = int(time.time())
    if session_id is not None:
        existing = session_store.get_session(session_id)
        if existing is not None:
            return existing
        if not create_session:
            raise ValueError(f"unknown session: {session_id}")
        return session_store.create_session(
            session_id=session_id,
            title=session_title,
            created_at=now,
        )
    if create_session:
        return session_store.create_session(
            title=session_title,
            created_at=now,
        )
    latest = session_store.latest_session()
    if latest is not None:
        return latest
    return session_store.create_session(
        title=session_title,
        created_at=now,
    )


def _main_thread_id_for_session(session_id: str) -> str:
    return f"{MAIN_THREAD_PREFIX}-{session_id}"


def _session_title_from_message(body: str, *, limit: int = 60) -> str:
    normalized = " ".join(body.split()).strip()
    if not normalized:
        return "Untitled Session"
    if len(normalized) <= limit:
        return normalized
    trimmed = normalized[: limit - 1].rstrip()
    return f"{trimmed}…"


def _truncate_preview(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _orchestrator_turn_planner_instructions(runtime: RuntimeContext) -> str:
    workflow_lines = []
    for summary in runtime.list_workflow_summaries():
        hints = ", ".join(summary["selection_hints"]) or "none"
        workflow_lines.append(
            f"- {summary['id']}: orchestration={summary['orchestration']} "
            f"delivery_candidate={summary['delivery_candidate']} "
            f"acceptance={summary['acceptance_mode']} "
            f"selection_hints={hints} purpose={summary['purpose']}"
        )
    agent_lines = []
    for summary in runtime.list_agent_summaries():
        if summary["id"] == "orchestrator":
            continue
        agent_lines.append(f"- {summary['id']}: role={summary['role']}")
    return "\n".join(
        [
            "You are the internal planning layer for the orchestrator.",
            "Choose the single best next action for the current user turn.",
            "Output JSON only. No markdown. No explanation outside the JSON object.",
            "",
            "Allowed modes:",
            '- "act": let the orchestrator handle the turn directly with its normal tool-enabled agent run. Use this for discussion, clarification, or direct execution by the orchestrator.',
            '- "delegate": hand the work to one specialist thread.',
            '- "workflow": run a named workflow end to end.',
            "",
            "Rules:",
            "- Prefer workflow for non-trivial implementation, debugging, verification, or delivery work.",
            "- If the user wants runnable code or an implemented project outcome, preserve the full delivery goal instead of narrowing it to a design-only or review-only subtask.",
            "- For tiny, bounded delivery, prefer workflows tagged `tiny_delivery`.",
            "- For staged implementation with explicit build/test/review structure, prefer workflows tagged `staged_delivery`.",
            "- For broad, evolving, or uncertainty-heavy delivery, prefer workflows tagged `adaptive_delivery`.",
            "- For exploratory specialist-to-specialist discussion, prefer workflows tagged `exploratory`.",
            "- When the user expects delivery, strongly prefer workflows marked `delivery_candidate=true`.",
            "- Avoid workflows tagged `design_only`, `research_only`, or other non-delivery-only patterns when the user explicitly expects implementation in the same turn.",
            "- Set `deliverable_expected` to true whenever the user expects repo changes, runnable code, tests, bug fixes, or completed implementation work.",
            "- If the recent conversation already settled the approach and the latest user turn is approval to proceed, choose delivery work instead of another reply.",
            "- Prefer delegate only for narrow specialist work.",
            "- Use act when discussion is the real goal, when the orchestrator should answer directly, or when the orchestrator should work the turn directly.",
            "- Do not rely on keyword matching. Infer intent from the actual request and context.",
            "- If the user asked to build or change the repo and enough information exists, do not choose a non-delivery workflow.",
            "- For workflow mode, make `goal` a self-contained summary of the work using the recent conversation, not just the latest short acknowledgment.",
            "- For delegate mode, make `request` a self-contained specialist brief using the recent conversation.",
            "",
            "Available workflows:",
            *workflow_lines,
            "",
            "Available specialists:",
            *agent_lines,
            "",
            "Required JSON shape:",
            '{"mode":"workflow|delegate|act","workflow_id":null,"agent_id":null,"title":"","request":"","goal":"","deliverable_expected":false}',
        ]
    )


def _orchestrator_turn_planner_prompt(runtime: RuntimeContext, body: str) -> str:
    recent_messages = runtime.list_main_messages()[-8:]
    transcript_lines = []
    for message in recent_messages:
        text = runtime.conversation_store.read_message_body(message).rstrip("\n")
        if not text:
            continue
        transcript_lines.append(f"{message.sender}: {text}")
    return "\n".join(
        [
            "Main thread transcript:",
            *transcript_lines,
            "",
            "Latest user request:",
            body,
        ]
    )


def _delivery_audit_instructions() -> str:
    return "\n".join(
        [
            "You are a narrow internal delivery audit for the orchestrator.",
            "Decide whether the current planner decision is aligned with what the user expects right now.",
            "Output JSON only.",
            "",
            "Required JSON shape:",
            '{"deliverable_expected": false, "reconsider": false, "reason": ""}',
            "",
            "Set `deliverable_expected` to true when the user expects implementation, bug fixing, verification, repo changes, or completed execution now.",
            "Set `deliverable_expected` to true when the recent conversation already settled the direction and the latest user turn is approval to proceed.",
            "Set `reconsider` to true when the planner decision should be escalated for concrete delivery work instead of being accepted as-is.",
            "Set `reconsider` to false when the planner decision is acceptable for this turn.",
            "Use the whole recent conversation, not just the latest short acknowledgment.",
        ]
    )


def _delivery_reconsideration_instructions(runtime: RuntimeContext, reason: str) -> str:
    workflow_lines = []
    for summary in runtime.list_workflow_summaries():
        if not summary["delivery_candidate"]:
            continue
        workflow_lines.append(
            f"- {summary['id']}: orchestration={summary['orchestration']} "
            f"acceptance={summary['acceptance_mode']} purpose={summary['purpose']}"
        )
    agent_lines = []
    for summary in runtime.list_agent_summaries():
        if summary["id"] == "orchestrator":
            continue
        agent_lines.append(f"- {summary['id']}: role={summary['role']}")
    return "\n".join(
        [
            "You are the internal delivery replanner for the orchestrator.",
            f"The previous decision was rejected for this reason: {reason}",
            "The user still expects concrete delivery work now.",
            "Choose the best next move for delivery.",
            "Output JSON only.",
            "",
            "Allowed modes:",
            '- "act": let the orchestrator handle the turn directly with its normal tool-enabled run.',
            '- "delegate": hand the work to one specialist thread.',
            '- "workflow": run a named delivery workflow end to end.',
            "",
            "Rules:",
            "- Do not choose reply.",
            "- Do not choose a non-delivery workflow.",
            "- Preserve the full delivery goal from the recent conversation.",
            "- Prefer workflow when implementation, verification, or review loops are likely needed.",
            "- Prefer delegate only for narrow specialist work.",
            "",
            "Available workflows:",
            *workflow_lines,
            "",
            "Available specialists:",
            *agent_lines,
            "",
            "Required JSON shape:",
            '{"mode":"workflow|delegate|act","workflow_id":null,"agent_id":null,"title":"","request":"","goal":""}',
        ]
    )


def _delivery_audit_prompt(
    runtime: RuntimeContext,
    body: str,
    decision: OrchestratorTurnDecision,
) -> str:
    recent_messages = runtime.list_main_messages()[-8:]
    transcript_lines = []
    for message in recent_messages:
        text = runtime.conversation_store.read_message_body(message).rstrip("\n")
        if not text:
            continue
        transcript_lines.append(f"{message.sender}: {text}")
    workflow_note = "(none)"
    if decision.mode == "workflow" and decision.workflow_id is not None:
        workflow_note = decision.workflow_id
    return "\n".join(
        [
            "Main thread transcript:",
            *transcript_lines,
            "",
            "Latest user request:",
            body,
            "",
            "Planner decision:",
            f"- mode: {decision.mode}",
            f"- workflow_id: {workflow_note}",
            f"- agent_id: {decision.agent_id or '(none)'}",
            f"- deliverable_expected: {decision.deliverable_expected}",
        ]
    )


def _delegation_review_instructions() -> str:
    return "\n".join(
        [
            "You are a narrow internal acceptance reviewer for a delegated specialist result.",
            "Decide whether the delegated result appears complete enough to accept without another worker turn.",
            "Prefer clarification or rejection when the result is vague, missing evidence, or clearly does not satisfy the request.",
            "Do not delegate, do not plan, and do not call tools.",
            "Output JSON only with this shape:",
            '{"accepted": true, "summary": "one concise sentence", "findings": ["specific issue"]}',
        ]
    )


def _render_delegation_review_prompt(
    *,
    agent_id: str,
    request: str,
    result: str,
    evidence_lines: list[str],
) -> str:
    lines = [
        f"Delegated agent: {agent_id}",
        "",
        "Original request:",
        request.strip() or "(no request provided)",
        "",
        "Delegated result:",
        result.strip() or "(no result provided)",
    ]
    if evidence_lines:
        lines.extend(["", *evidence_lines])
    lines.extend(
        [
            "",
            "Decide whether this delegated result should be accepted as complete.",
            "If it is not complete enough, reject it and explain the most important missing point.",
        ]
    )
    return "\n".join(lines).strip()


def _delegation_review_verdict_from_payload(parsed: dict[str, object]) -> DelegationReviewVerdict:
    summary = _optional_text(parsed.get("summary")) or "No delegation review summary provided."
    findings_raw = parsed.get("findings", [])
    findings: list[str] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                findings.append(stripped)
    return DelegationReviewVerdict(
        accepted=_as_bool(parsed.get("accepted")),
        summary=summary,
        findings=tuple(findings),
    )


def _format_delegation_review_summary(verdict: DelegationReviewVerdict) -> str:
    prefix = "ACCEPTED" if verdict.accepted else "REJECTED"
    lines = [f"{prefix}: {verdict.summary}"]
    if verdict.findings:
        lines.append("")
        lines.append("Findings:")
        lines.extend(f"- {finding}" for finding in verdict.findings)
    return "\n".join(lines)


def _parse_turn_decision_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("no JSON object found")
        text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("planner output must decode to an object")
    return parsed


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return False


def _validate_runtime_step_groups(
    *,
    workflow_id: str,
    step_groups: tuple[tuple[str, ...], ...],
    known_agents: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if not step_groups:
        return ()
    allowed = set(known_agents)
    validated: list[tuple[str, ...]] = []
    for group in step_groups:
        if not group:
            raise ValueError(f"workflow '{workflow_id}' step groups must be non-empty")
        cleaned: list[str] = []
        for agent_id in group:
            if agent_id not in allowed:
                raise ValueError(f"workflow '{workflow_id}' references unknown agent '{agent_id}'")
            cleaned.append(agent_id)
        validated.append(tuple(cleaned))
    return tuple(validated)


def _workflow_supports_adaptive_staffing(
    *,
    adaptive_staffing_enabled: bool,
    orchestration: str,
    default_step_groups: tuple[tuple[str, ...], ...],
) -> bool:
    if not default_step_groups:
        return False
    if not adaptive_staffing_enabled:
        return False
    if orchestration not in {"sequential", "magentic", "handoff"}:
        return False
    participants = {
        agent_id
        for group in default_step_groups
        for agent_id in group
    }
    if len(participants) <= 1:
        return False
    return True


def _workflow_staffing_selector_instructions(
    *,
    workflow_id: str,
    orchestration: str,
    available_agents: tuple[str, ...],
) -> str:
    if orchestration in {"magentic", "handoff"}:
        staffing_rule = (
            "Choose the smallest useful participant set for the shared specialist room. "
            "List agents in the order they are most likely to matter first."
        )
    else:
        staffing_rule = (
            "Choose the smallest team that is likely to finish the work cleanly. "
            "Do not add extra steps unless they materially improve delivery quality. "
            "Use at most one agent per step for staged workflows unless parallel specialists are clearly justified."
        )
    return "\n".join(
        [
            "You are choosing the initial specialist sequence for a workflow run.",
            f"Workflow id: {workflow_id}.",
            f"Orchestration style: {orchestration}.",
            f"Available specialists: {', '.join(available_agents) or '(none)'}.",
            staffing_rule,
            "Return JSON only with this shape:",
            '{"step_groups":[["coder"]]}',
        ]
    )


def _workflow_staffing_selector_prompt(
    *,
    workflow_id: str,
    orchestration: str,
    goal: str,
    default_step_groups: tuple[tuple[str, ...], ...],
) -> str:
    return "\n".join(
        [
            f"Workflow: {workflow_id}",
            f"Orchestration: {orchestration}",
            "",
            "Goal:",
            goal,
            "",
            "Default staffing:",
            _render_workflow_plan_lines(default_step_groups),
            "",
            "Choose the initial staffing plan.",
        ]
    ).strip()


def _parse_selected_workflow_step_groups(
    *,
    workflow_id: str,
    raw: str,
    known_agents: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    payload = _parse_turn_decision_json(raw)
    return _selected_workflow_step_groups_from_payload(
        workflow_id=workflow_id,
        payload=payload,
        known_agents=known_agents,
    )


def _selected_workflow_step_groups_from_payload(
    *,
    workflow_id: str,
    payload: dict[str, object],
    known_agents: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    raw_step_groups = payload.get("step_groups")
    if not isinstance(raw_step_groups, list):
        raise ValueError("step_groups is required")
    parsed: list[tuple[str, ...]] = []
    allowed = set(known_agents)
    for raw_group in raw_step_groups:
        group = validate_workflow_group(workflow_id, raw_group)
        cleaned: list[str] = []
        for agent_id in group:
            if agent_id not in allowed or agent_id == "orchestrator":
                raise ValueError(f"unknown staffing agent: {agent_id}")
            cleaned.append(agent_id)
        parsed.append(tuple(cleaned))
    return tuple(parsed)


def _workflow_group_label(group: tuple[str, ...]) -> str:
    if len(set(group)) == 1 and len(group) > 1:
        return f"{group[0]} x{len(group)}"
    return ", ".join(group)


def _render_workflow_plan_lines(step_groups: tuple[tuple[str, ...], ...]) -> str:
    if not step_groups:
        return "(no steps)"
    return "\n".join(
        f"{index}. {' + '.join(group)}"
        for index, group in enumerate(step_groups, start=1)
    )


def _render_runtime_workflow_graph(
    *,
    workflow_id: str,
    orchestration: str,
    step_groups: tuple[tuple[str, ...], ...],
) -> str:
    if orchestration in {"group_chat", "magentic", "handoff"}:
        label = " + ".join(agent_id for group in step_groups for agent_id in group) or "workroom"
        return "\n".join(
            [
                "flowchart TD",
                '    start["Goal"] --> step1["%s: %s"]' % (workflow_id, label),
                '    step1 --> review["orchestrator review"]',
            ]
        )
    lines = ["flowchart TD", '    start["Goal"]']
    previous = "start"
    for index, group in enumerate(step_groups, start=1):
        node = f"step{index}"
        label = _workflow_group_label(group)
        lines.append(f'    {previous} --> {node}["{label}"]')
        previous = node
    lines.append(f'    {previous} --> review["orchestrator review"]')
    return "\n".join(lines)


def _workflow_artifact_priority(kind: str) -> int:
    if kind == "workflow-report":
        return 0
    if kind == "workflow-graph":
        return 1
    return 2


def _render_command_output(
    *,
    command: str,
    cwd: str,
    exit_code: int,
    status: str,
    timeout: int,
    stdout: str,
    stderr: str,
    thread_id: str | None,
    task_id: str | None,
    agent_id: str | None,
) -> str:
    lines = [
        "# Command Run",
        "",
        f"- Command: `{command}`",
        f"- Cwd: `{cwd}`",
        f"- Exit Code: {exit_code}",
        f"- Status: {status}",
        f"- Timeout: {timeout}",
    ]
    if agent_id is not None:
        lines.append(f"- Agent: {agent_id}")
    if thread_id is not None:
        lines.append(f"- Thread: {thread_id}")
    if task_id is not None:
        lines.append(f"- Task: {task_id}")
    lines.extend(
        [
            "",
            "## Stdout",
            "```text",
            stdout.rstrip("\n"),
            "```",
            "",
            "## Stderr",
            "```text",
            stderr.rstrip("\n"),
            "```",
        ]
    )
    return "\n".join(lines)
