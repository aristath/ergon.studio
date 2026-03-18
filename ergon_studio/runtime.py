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
from ergon_studio.storage.models import ApprovalRecord, ArtifactRecord, EventRecord, MemoryFactRecord, MessageRecord, TaskRecord, ThreadRecord, WorkflowRunRecord
from ergon_studio.task_store import TaskStore
from ergon_studio.tool_registry import build_workspace_tool_registry
from ergon_studio.workflow_store import WorkflowStore


MAIN_SESSION_ID = "session-main"
MAIN_THREAD_ID = "thread-main"


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
class RuntimeContext:
    paths: StudioPaths
    registry: RuntimeRegistry
    tool_registry: dict[str, object]
    agent_session_store: AgentSessionStore
    conversation_store: ConversationStore
    task_store: TaskStore
    workflow_store: WorkflowStore
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

    def read_workflow_definition_text(self, workflow_id: str) -> str:
        definition = self.registry.workflow_definitions[workflow_id]
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

    def save_workflow_definition_text(
        self,
        *,
        workflow_id: str,
        text: str,
        created_at: int | None = None,
    ) -> DefinitionDocument:
        definition = self.registry.workflow_definitions[workflow_id]
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
        thread_ids: set[str] = set()
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)
            for thread in step.threads:
                thread_ids.add(thread.id)

        artifacts = [
            artifact
            for artifact in self.list_artifacts()
            if artifact.task_id in task_ids or artifact.thread_id in thread_ids
        ]
        return sorted(artifacts, key=lambda artifact: (artifact.created_at, artifact.id))

    def list_events_for_workflow_run(self, workflow_run_id: str) -> list[EventRecord]:
        run_view = self.describe_workflow_run(workflow_run_id)
        if run_view is None:
            return []

        task_ids: set[str] = set()
        thread_ids: set[str] = set()
        if run_view.root_task is not None:
            task_ids.add(run_view.root_task.id)
        for step in run_view.steps:
            task_ids.add(step.task.id)
            for thread in step.threads:
                thread_ids.add(thread.id)

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
        return await self._run_agent_turn(
            thread_id=self.main_thread_id,
            agent_id="orchestrator",
            prompt_sender="user",
            reply_sender="orchestrator",
            body=body,
            created_at=created_at,
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
    ) -> tuple[MessageRecord, MessageRecord | None]:
        prompt_message = self.append_message_to_thread(
            thread_id=thread_id,
            message_id=f"message-{uuid4().hex}",
            sender=prompt_sender,
            kind="chat",
            body=body,
            created_at=created_at,
        )

        try:
            agent = self.build_agent(agent_id)
        except (KeyError, ValueError) as exc:
            self.append_event(
                kind="agent_unavailable",
                summary=f"{agent_id} unavailable: {exc}",
                created_at=created_at + 1,
                thread_id=thread_id,
            )
            return prompt_message, None

        session = self.agent_session_store.load_or_create_session(
            thread_id=thread_id,
            agent_id=agent_id,
            session_factory=lambda session_id: agent.create_session(session_id=session_id),
        )

        try:
            response = await agent.run(
                [
                    Message(
                        role="user",
                        text=body,
                        author_name=prompt_sender,
                    )
                ],
                session=session,
            )
        except Exception as exc:
            self.append_event(
                kind="agent_failed",
                summary=f"{agent_id} run failed: {type(exc).__name__}: {exc}",
                created_at=created_at + 1,
                thread_id=thread_id,
            )
            self.agent_session_store.save_session(
                thread_id=thread_id,
                agent_id=agent_id,
                session=session,
            )
            return prompt_message, None

        self.agent_session_store.save_session(
            thread_id=thread_id,
            agent_id=agent_id,
            session=session,
        )
        response_text = response.text.strip()
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
        threads: list[ThreadRecord],
        next_index: int,
    ) -> str:
        if next_index == 0:
            goal = self.latest_main_user_message_body()
            if goal:
                return f"Workflow kickoff: {workflow_run.workflow_id}\n\nGoal:\n{goal}"
            return f"Workflow kickoff: {workflow_run.workflow_id}"

        previous_thread = threads[next_index - 1]
        previous_output = self._latest_agent_message_body(previous_thread.id)
        if previous_output:
            return (
                f"Continue workflow: {workflow_run.workflow_id}\n\n"
                f"Previous step output from {previous_thread.assigned_agent_id}:\n"
                f"{previous_output}"
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

    def start_workflow_run(self, *, workflow_id: str, created_at: int) -> tuple[WorkflowRunRecord, list[ThreadRecord]]:
        participants = _workflow_participants(workflow_id)
        root_task = self.create_task(
            task_id=f"task-{uuid4().hex[:8]}",
            title=f"Workflow: {workflow_id}",
            state="in_progress",
            created_at=created_at,
        )
        workflow_run = self.workflow_store.create_workflow_run(
            session_id=self.main_session_id,
            workflow_run_id=f"workflow-run-{uuid4().hex[:8]}",
            workflow_id=workflow_id,
            state="running",
            created_at=created_at + 1,
            root_task_id=root_task.id,
        )
        self.append_event(
            kind="workflow_started",
            summary=f"Started workflow {workflow_id}",
            created_at=created_at + 1,
            task_id=root_task.id,
        )
        threads = self._append_workflow_steps(
            workflow_id=workflow_id,
            root_task_id=root_task.id,
            participants=participants,
            created_at=created_at + 2,
        )
        return workflow_run, threads

    def request_workflow_fix_cycle(
        self,
        *,
        workflow_run_id: str,
        created_at: int,
    ) -> tuple[WorkflowRunRecord, list[ThreadRecord]]:
        workflow_run = self.get_workflow_run(workflow_run_id)
        if workflow_run is None:
            raise ValueError(f"unknown workflow run: {workflow_run_id}")

        existing_threads = self._workflow_threads_for_run(workflow_run)
        if workflow_run.current_step_index < len(existing_threads):
            raise ValueError("workflow run still has pending steps")
        if workflow_run.root_task_id is None:
            raise ValueError("workflow run has no root task")

        threads = self._append_workflow_steps(
            workflow_id=workflow_run.workflow_id,
            root_task_id=workflow_run.root_task_id,
            participants=("fixer", "reviewer"),
            created_at=created_at,
        )
        updated = WorkflowRunRecord(
            id=workflow_run.id,
            session_id=workflow_run.session_id,
            workflow_id=workflow_run.workflow_id,
            state="repairing",
            created_at=workflow_run.created_at,
            updated_at=created_at + 1,
            root_task_id=workflow_run.root_task_id,
            current_step_index=len(existing_threads),
            last_thread_id=workflow_run.last_thread_id,
        )
        self.workflow_store.update_workflow_run(updated)
        self.update_task_state(
            task_id=workflow_run.root_task_id,
            state="in_progress",
            updated_at=created_at + 1,
        )
        self.append_event(
            kind="workflow_fix_cycle_requested",
            summary=f"Requested fix cycle for workflow {workflow_run.workflow_id}",
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

        threads = self._workflow_threads_for_run(workflow_run)
        next_index = workflow_run.current_step_index
        if next_index >= len(threads):
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

        thread = threads[next_index]
        if thread.parent_task_id is not None:
            self.update_task_state(
                task_id=thread.parent_task_id,
                state="in_progress",
                updated_at=created_at,
            )
        prompt = self._workflow_prompt_for_step(
            workflow_run=workflow_run,
            threads=threads,
            next_index=next_index,
        )
        _, reply = await self.send_message_to_agent_thread(
            thread_id=thread.id,
            body=prompt,
            created_at=created_at,
        )
        if reply is None:
            blocked = WorkflowRunRecord(
                id=workflow_run.id,
                session_id=workflow_run.session_id,
                workflow_id=workflow_run.workflow_id,
                state="blocked",
                created_at=workflow_run.created_at,
                updated_at=created_at + 1,
                root_task_id=workflow_run.root_task_id,
                current_step_index=workflow_run.current_step_index,
                last_thread_id=thread.id,
            )
            self.workflow_store.update_workflow_run(blocked)
            if thread.parent_task_id is not None:
                self.update_task_state(
                    task_id=thread.parent_task_id,
                    state="blocked",
                    updated_at=created_at + 1,
                )
            self.append_event(
                kind="workflow_blocked",
                summary=f"Workflow {workflow_run.workflow_id} blocked at {thread.assigned_agent_id}",
                created_at=created_at + 1,
                thread_id=thread.id,
                task_id=workflow_run.root_task_id,
            )
            return blocked, thread, None

        updated = WorkflowRunRecord(
            id=workflow_run.id,
            session_id=workflow_run.session_id,
            workflow_id=workflow_run.workflow_id,
            state="completed" if next_index + 1 >= len(threads) else "running",
            created_at=workflow_run.created_at,
            updated_at=created_at + 1,
            root_task_id=workflow_run.root_task_id,
            current_step_index=next_index + 1,
            last_thread_id=thread.id,
        )
        self.workflow_store.update_workflow_run(updated)
        if thread.parent_task_id is not None:
            self.update_task_state(
                task_id=thread.parent_task_id,
                state="completed",
                updated_at=created_at + 1,
            )
        self.append_event(
            kind="workflow_advanced",
            summary=f"Advanced workflow {workflow_run.workflow_id} to {thread.assigned_agent_id}",
            created_at=created_at + 1,
            thread_id=thread.id,
            task_id=workflow_run.root_task_id,
        )
        if updated.state == "completed":
            if workflow_run.root_task_id is not None:
                self.update_task_state(
                    task_id=workflow_run.root_task_id,
                    state="completed",
                    updated_at=created_at + 2,
                )
            self._create_workflow_completion_artifact(
                workflow_run=updated,
                thread=thread,
                created_at=created_at + 2,
            )
            self._append_workflow_completion_summary(
                workflow_run=updated,
                thread=thread,
                created_at=created_at + 3,
            )
            self.append_event(
                kind="workflow_completed",
                summary=f"Completed workflow {workflow_run.workflow_id}",
                created_at=created_at + 4,
                task_id=workflow_run.root_task_id,
            )
        return updated, thread, reply

    def _append_workflow_steps(
        self,
        *,
        workflow_id: str,
        root_task_id: str,
        participants: tuple[str, ...],
        created_at: int,
    ) -> list[ThreadRecord]:
        threads: list[ThreadRecord] = []
        for offset, agent_id in enumerate(participants):
            child_task = self.create_task(
                task_id=f"task-{uuid4().hex[:8]}",
                title=f"{workflow_id}: {agent_id}",
                state="planned",
                created_at=created_at + offset,
                parent_task_id=root_task_id,
            )
            thread = self.create_agent_thread(
                agent_id=agent_id,
                created_at=created_at + len(participants) + offset,
                parent_task_id=child_task.id,
            )
            threads.append(thread)
        return threads

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
            assigned_agent_id=None,
        )


def load_runtime(project_root: Path, home_dir: Path) -> RuntimeContext:
    paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
    registry = load_registry(paths)
    tool_registry = build_workspace_tool_registry(paths.project_root)
    agent_session_store = AgentSessionStore(paths)
    conversation_store = ConversationStore(paths)
    task_store = TaskStore(paths)
    workflow_store = WorkflowStore(paths)
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
        workflow_store=workflow_store,
        event_store=event_store,
        approval_store=approval_store,
        memory_store=memory_store,
        artifact_store=artifact_store,
    )
    runtime.ensure_main_conversation()
    return runtime


def _workflow_participants(workflow_id: str) -> tuple[str, ...]:
    workflow_map = {
        "direct-response": (),
        "single-agent-execution": ("coder",),
        "architecture-first": ("architect",),
        "research-then-decide": ("researcher",),
        "standard-build": ("architect", "coder", "reviewer"),
        "best-of-n": ("coder", "reviewer"),
        "debate": ("architect", "brainstormer", "reviewer"),
        "review-driven-repair": ("reviewer", "fixer"),
        "test-driven-repair": ("tester", "fixer", "reviewer"),
        "approval-gated": (),
        "replanning": ("architect",),
    }
    return workflow_map.get(workflow_id, ())
