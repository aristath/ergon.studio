from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
import json
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_framework import Agent, Executor, Message, WorkflowBuilder, WorkflowContext, handler
from agent_framework.orchestrations import GroupChatBuilder, clean_conversation_for_handoff
from agent_framework_orchestrations._base_group_chat_orchestrator import GroupChatParticipantMessage, GroupChatRequestMessage, GroupChatResponseMessage

from ergon_studio.agent_factory import compose_instructions
from ergon_studio.context_providers import WORKSPACE_STATE_KEY
from ergon_studio.tool_context import ToolExecutionContext, use_tool_execution_context

if TYPE_CHECKING:
    from ergon_studio.runtime import RuntimeContext, WorkflowRunView
    from ergon_studio.storage.models import ThreadRecord, WorkflowRunRecord


@dataclass(frozen=True)
class WorkflowExecutionSummary:
    workflow_run_id: str
    workflow_id: str
    state: str
    current_step_index: int
    last_thread_id: str | None
    review_thread_id: str
    review_summary: str | None
    review_accepted: bool | None
    artifact_id: str | None


@dataclass
class _ExecutionTracker:
    completed_step_indices: set[int] = field(default_factory=set)
    thread_outputs: dict[str, str] = field(default_factory=dict)
    blocked_step_index: int | None = None
    blocked_thread_id: str | None = None
    failed: bool = False
    last_thread_id: str | None = None
    review_summary: str | None = None
    review_accepted: bool | None = None
    artifact_id: str | None = None
    repair_cycles: int = 0
    replan_cycles: int = 0
    review_findings: tuple[str, ...] = ()
    review_requires_replan: bool = False
    review_replan_summary: str | None = None


@dataclass(frozen=True)
class WorkflowReviewVerdict:
    accepted: bool
    summary: str
    findings: tuple[str, ...] = ()
    requires_replan: bool = False
    replan_summary: str | None = None


class _TimestampCursor:
    def __init__(self, start_at: int) -> None:
        self._counter = count(start_at)

    def next(self) -> int:
        return next(self._counter)


class _KickoffExecutor(Executor):
    def __init__(self) -> None:
        super().__init__("workflow-kickoff")

    @handler
    async def start(self, goal: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(goal)


class _ThreadExecutor(Executor):
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        thread: ThreadRecord,
        workflow_id: str,
        step_index: int,
        total_steps: int,
        goal: str,
        cursor: _TimestampCursor,
        tracker: _ExecutionTracker,
    ) -> None:
        super().__init__(thread.id)
        self.runtime = runtime
        self.thread = thread
        self.workflow_id = workflow_id
        self.step_index = step_index
        self.total_steps = total_steps
        self.goal = goal
        self.cursor = cursor
        self.tracker = tracker

    @handler
    async def run_text(self, payload: str, ctx: WorkflowContext[str]) -> None:
        await self._run(payload, ctx)

    @handler
    async def run_list(self, payload: list[str], ctx: WorkflowContext[str]) -> None:
        await self._run(payload, ctx)

    async def _run(self, payload: str | list[str], ctx: WorkflowContext[str]) -> None:
        task_id = self.thread.parent_task_id
        if task_id is not None:
            self.runtime.update_task_state(
                task_id=task_id,
                state="in_progress",
                updated_at=self.cursor.next(),
            )

        prompt = _render_step_prompt(
            workflow_id=self.workflow_id,
            goal=self.goal,
            step_index=self.step_index,
            total_steps=self.total_steps,
            agent_id=self.thread.assigned_agent_id or "unknown",
            payload=payload,
        )
        before_tool_ids = {
            tool_call.id
            for tool_call in self.runtime.list_tool_calls()
            if tool_call.thread_id == self.thread.id
        }
        _, reply = await self.runtime.send_message_to_agent_thread(
            thread_id=self.thread.id,
            body=prompt,
            created_at=self.cursor.next(),
        )
        if reply is None:
            self.tracker.blocked_step_index = self.step_index
            self.tracker.blocked_thread_id = self.thread.id
            self.tracker.last_thread_id = self.thread.id
            if task_id is not None:
                self.runtime.update_task_state(
                    task_id=task_id,
                    state="blocked",
                    updated_at=self.cursor.next(),
                )
            raise RuntimeError(f"workflow step blocked: {self.thread.assigned_agent_id or self.thread.id}")

        reply_body = self.runtime.conversation_store.read_message_body(reply).rstrip("\n")
        reply, reply_body = await self._ensure_required_tool_use(
            reply=reply,
            reply_body=reply_body,
            before_tool_ids=before_tool_ids,
        )
        self.tracker.thread_outputs[self.thread.id] = reply_body
        self.tracker.last_thread_id = self.thread.id
        await ctx.send_message(reply_body)

    async def _ensure_required_tool_use(
        self,
        *,
        reply,
        reply_body: str,
        before_tool_ids: set[str],
    ):
        required_tools = _required_tool_names(self.thread.assigned_agent_id or "")
        if not required_tools:
            return reply, reply_body
        if _has_required_tool_calls(
            tool_calls=self.runtime.list_tool_calls(),
            thread_id=self.thread.id,
            new_tool_ids=before_tool_ids,
            required_tools=required_tools,
        ):
            return reply, reply_body

        self.runtime.append_event(
            kind="workflow_step_retry",
            summary=f"Retrying {self.thread.assigned_agent_id or self.thread.id} because required tool evidence is missing",
            created_at=self.cursor.next(),
            thread_id=self.thread.id,
            task_id=self.thread.parent_task_id,
        )
        retry_prompt = _render_tool_requirement_retry(self.thread.assigned_agent_id or "")
        before_retry_ids = {
            tool_call.id
            for tool_call in self.runtime.list_tool_calls()
            if tool_call.thread_id == self.thread.id
        }
        _, retry_reply = await self.runtime.send_message_to_agent_thread(
            thread_id=self.thread.id,
            body=retry_prompt,
            created_at=self.cursor.next(),
        )
        if retry_reply is None:
            self.tracker.blocked_step_index = self.step_index
            self.tracker.blocked_thread_id = self.thread.id
            self.tracker.last_thread_id = self.thread.id
            raise RuntimeError(f"workflow step blocked after retry: {self.thread.assigned_agent_id or self.thread.id}")

        retry_body = self.runtime.conversation_store.read_message_body(retry_reply).rstrip("\n")
        if _has_required_tool_calls(
            tool_calls=self.runtime.list_tool_calls(),
            thread_id=self.thread.id,
            new_tool_ids=before_retry_ids,
            required_tools=required_tools,
        ):
            return retry_reply, retry_body

        self.tracker.blocked_step_index = self.step_index
        self.tracker.blocked_thread_id = self.thread.id
        self.tracker.last_thread_id = self.thread.id
        if self.thread.parent_task_id is not None:
            self.runtime.update_task_state(
                task_id=self.thread.parent_task_id,
                state="blocked",
                updated_at=self.cursor.next(),
            )
        raise RuntimeError(f"workflow step missing required tool use: {self.thread.assigned_agent_id or self.thread.id}")


class _GroupChatParticipantExecutor(Executor):
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        thread: ThreadRecord,
        agent_id: str,
        workflow_id: str,
        goal: str,
        cursor: _TimestampCursor,
        tracker: _ExecutionTracker,
    ) -> None:
        super().__init__(agent_id)
        self.runtime = runtime
        self.thread = thread
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.goal = goal
        self.cursor = cursor
        self.tracker = tracker
        self._cache: list[Message] = []

    @handler
    async def sync_messages(
        self,
        payload: GroupChatParticipantMessage,
        ctx: WorkflowContext,
    ) -> None:
        del ctx
        self._cache.extend(clean_conversation_for_handoff(payload.messages))

    @handler
    async def handle_request(
        self,
        payload: GroupChatRequestMessage,
        ctx: WorkflowContext[GroupChatResponseMessage],
    ) -> None:
        messages = list(self._cache)
        instruction = payload.additional_instruction.strip() if payload.additional_instruction else ""
        if not instruction:
            instruction = (
                f"Continue the shared discussion as {self.agent_id}. "
                "Add one substantive turn that moves the team toward a decision-ready recommendation."
            )
        messages.append(
            Message(
                role="user",
                text=instruction,
                author_name="orchestrator",
            )
        )
        if not messages:
            messages = [Message(role="user", text=self.goal, author_name="workflow")]

        response = await self._run_group_chat_turn(messages)
        response_text = response.text.strip()
        if not response_text:
            response_text = f"{self.agent_id} had no response."

        self.runtime.append_message_to_thread(
            thread_id=self.thread.id,
            message_id=f"message-{uuid4().hex}",
            sender=self.agent_id,
            kind="chat",
            body=response_text,
            created_at=self.cursor.next(),
        )
        _append_thread_output(
            tracker=self.tracker,
            thread_id=self.thread.id,
            speaker=self.agent_id,
            body=response_text,
        )
        self.tracker.last_thread_id = self.thread.id
        self._cache.clear()
        await ctx.send_message(
            GroupChatResponseMessage(
                message=Message(role="assistant", text=response_text, author_name=self.agent_id)
            )
        )

    async def _run_group_chat_turn(self, messages: list[Message]):
        try:
            base_agent = self.runtime.build_agent(self.agent_id)
        except (KeyError, ValueError) as exc:
            self.runtime.append_event(
                kind="agent_unavailable",
                summary=f"{self.agent_id} unavailable: {exc}",
                created_at=self.cursor.next(),
                thread_id=self.thread.id,
                task_id=self.thread.parent_task_id,
            )
            raise

        session = self.runtime.agent_session_store.load_or_create_session(
            thread_id=self.thread.id,
            agent_id=self.agent_id,
            session_factory=lambda session_id: base_agent.create_session(session_id=session_id),
        )
        session.state[WORKSPACE_STATE_KEY] = {
            "session_id": self.runtime.main_session_id,
            "thread_id": self.thread.id,
            "task_id": self.thread.parent_task_id,
            "agent_id": self.agent_id,
            "created_at": self.cursor.next(),
        }
        tool_context = ToolExecutionContext(
            session_id=self.runtime.main_session_id,
            thread_id=self.thread.id,
            task_id=self.thread.parent_task_id,
            agent_id=self.agent_id,
        )
        agent = _group_chat_agent(base_agent, self.runtime, self.workflow_id, self.agent_id)
        try:
            with use_tool_execution_context(tool_context):
                response = await agent.run(messages=messages, session=session)
        except Exception as exc:
            self.runtime.append_event(
                kind="group_chat_participant_failed",
                summary=f"{self.agent_id} failed in group chat: {type(exc).__name__}: {exc}",
                created_at=self.cursor.next(),
                thread_id=self.thread.id,
                task_id=self.thread.parent_task_id,
            )
            self.runtime.agent_session_store.save_session(
                thread_id=self.thread.id,
                agent_id=self.agent_id,
                session=session,
            )
            raise
        self.runtime.agent_session_store.save_session(
            thread_id=self.thread.id,
            agent_id=self.agent_id,
            session=session,
        )
        return response


class _OrchestratorReviewExecutor(Executor):
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        workflow_run: WorkflowRunRecord,
        review_thread: ThreadRecord,
        goal: str,
        cursor: _TimestampCursor,
        tracker: _ExecutionTracker,
    ) -> None:
        super().__init__(review_thread.id)
        self.runtime = runtime
        self.workflow_run = workflow_run
        self.review_thread = review_thread
        self.goal = goal
        self.cursor = cursor
        self.tracker = tracker

    @handler
    async def review_text(self, payload: str, ctx: WorkflowContext[None, str]) -> None:
        await self._review(payload, ctx)

    @handler
    async def review_list(self, payload: list[str], ctx: WorkflowContext[None, str]) -> None:
        await self._review(payload, ctx)

    async def _review(self, payload: str | list[str], ctx: WorkflowContext[None, str]) -> None:
        review_summary = await _perform_orchestrator_review(
            runtime=self.runtime,
            workflow_run=self.workflow_run,
            review_thread=self.review_thread,
            goal=self.goal,
            payload=payload,
            cursor=self.cursor,
            tracker=self.tracker,
        )
        await ctx.yield_output(review_summary)


async def _perform_orchestrator_review(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    review_thread: ThreadRecord,
    goal: str,
    payload: str | list[str],
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
) -> str:
    prompt = _render_review_prompt(
        runtime=runtime,
        workflow_id=workflow_run.workflow_id,
        goal=goal,
        payload=payload,
        workspace_files=runtime._workspace_file_list(limit=16),
        prior_findings=tracker.review_findings,
    )
    runtime.append_message_to_thread(
        thread_id=review_thread.id,
        message_id=f"message-{uuid4().hex}",
        sender="workflow",
        kind="chat",
        body=prompt,
        created_at=cursor.next(),
    )
    reply = await _run_structured_review(
        runtime=runtime,
        review_thread=review_thread,
        workflow_run=workflow_run,
        prompt=prompt,
        created_at=cursor.next(),
    )
    if reply is None:
        verdict = WorkflowReviewVerdict(
            accepted=False,
            summary="I could not complete a reliable acceptance review from the current workflow evidence.",
        )
    else:
        try:
            verdict = _parse_review_verdict(reply)
        except ValueError:
            verdict = WorkflowReviewVerdict(
                accepted=False,
                summary="I could not produce a structured acceptance review from the workflow output.",
            )
    review_summary = _format_review_summary(verdict)
    runtime.append_message_to_thread(
        thread_id=review_thread.id,
        message_id=f"message-{uuid4().hex}",
        sender="orchestrator",
        kind="review",
        body=review_summary,
        created_at=cursor.next(),
    )
    tracker.review_summary = review_summary
    tracker.review_accepted = verdict.accepted
    if verdict.findings or not verdict.accepted:
        tracker.review_findings = verdict.findings
    tracker.review_requires_replan = verdict.requires_replan
    tracker.review_replan_summary = verdict.replan_summary
    tracker.last_thread_id = review_thread.id
    return review_summary


async def _run_structured_review(
    *,
    runtime: RuntimeContext,
    review_thread: ThreadRecord,
    workflow_run: WorkflowRunRecord,
    prompt: str,
    created_at: int,
) -> str | None:
    try:
        orchestrator = runtime.build_agent("orchestrator")
    except (KeyError, ValueError):
        return None
    client = getattr(orchestrator, "client", None)
    if client is None:
        return await runtime.generate_agent_text_without_tools(
            agent_id="orchestrator",
            body=prompt,
            created_at=created_at,
            thread_id=review_thread.id,
            extra_instructions=(
                "This is a review-only turn. Do not delegate, do not run workflows, and do not call tools. "
                "Return JSON only with `accepted`, `summary`, `findings`, `requires_replan`, and `replan_summary`."
            ),
        )
    review_agent = Agent(
        client=client,
        id="workflow-orchestrator-review",
        name="Workflow Orchestrator Review",
        description="Structured acceptance review",
        instructions=_orchestrator_review_instructions(),
    )
    try:
        response = await review_agent.run(
            [
                Message(
                    role="user",
                    text=prompt,
                    author_name="workflow",
                )
            ],
            session=review_agent.create_session(session_id=f"{review_thread.id}:review"),
        )
    except Exception as exc:
        runtime.append_event(
            kind="orchestrator_review_failed",
            summary=f"Review agent failed: {type(exc).__name__}: {exc}",
            created_at=created_at,
            thread_id=review_thread.id,
            task_id=workflow_run.root_task_id,
        )
        return None
    return response.text.strip() or None


async def execute_defined_workflow(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    run_view: WorkflowRunView,
    goal: str,
    review_thread: ThreadRecord,
    created_at: int,
) -> WorkflowExecutionSummary:
    cursor = _TimestampCursor(created_at)
    tracker = _ExecutionTracker()
    if _workflow_orchestration(runtime, workflow_run.workflow_id) == "group_chat":
        return await _execute_group_chat_workflow(
            runtime=runtime,
            workflow_run=workflow_run,
            run_view=run_view,
            goal=goal,
            review_thread=review_thread,
            cursor=cursor,
            tracker=tracker,
        )
    active_workflow_run = workflow_run
    active_run_view = run_view
    start_step_index = 0
    initial_payload: str | list[str] = goal
    result = None

    while True:
        result = await _execute_workflow_pass(
            runtime=runtime,
            workflow_run=active_workflow_run,
            run_view=active_run_view,
            goal=goal,
            review_thread=review_thread,
            cursor=cursor,
            tracker=tracker,
            start_step_index=start_step_index,
            initial_payload=initial_payload,
        )
        if tracker.failed or tracker.blocked_step_index is not None:
            break
        if tracker.review_accepted is not False:
            break
        followup = _next_followup_cycle(runtime=runtime, workflow_id=active_workflow_run.workflow_id, tracker=tracker)
        if followup is None:
            break
        cycle_kind, step_groups, event_kind, event_summary = followup
        followup_payload = _render_followup_payload(
            runtime=runtime,
            workflow_id=active_workflow_run.workflow_id,
            goal=goal,
            tracker=tracker,
            run_view=active_run_view,
            cycle_kind=cycle_kind,
        )
        runtime.append_event(
            kind=event_kind,
            summary=event_summary,
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=active_workflow_run.root_task_id,
        )
        next_state = "repairing" if cycle_kind == "repair" else "replanning"
        active_workflow_run = type(active_workflow_run)(
            id=active_workflow_run.id,
            session_id=active_workflow_run.session_id,
            workflow_id=active_workflow_run.workflow_id,
            state=next_state,
            created_at=active_workflow_run.created_at,
            updated_at=cursor.next(),
            root_task_id=active_workflow_run.root_task_id,
            current_step_index=len(active_run_view.steps),
            last_thread_id=tracker.last_thread_id,
        )
        runtime.workflow_store.update_workflow_run(active_workflow_run)
        active_workflow_run, _ = runtime.request_workflow_followup_cycle(
            workflow_run_id=active_workflow_run.id,
            created_at=cursor.next(),
            step_groups=step_groups,
            state=next_state,
            event_kind=f"workflow_{cycle_kind}_cycle_requested",
            event_summary=event_summary,
        )
        refreshed_view = runtime.describe_workflow_run(active_workflow_run.id)
        if refreshed_view is None:
            tracker.failed = True
            break
        start_step_index = len(active_run_view.steps)
        active_run_view = refreshed_view
        initial_payload = followup_payload

    return _finalize_workflow_run(
        runtime=runtime,
        workflow_run=active_workflow_run,
        run_view=active_run_view,
        goal=goal,
        review_thread=review_thread,
        tracker=tracker,
        cursor=cursor,
        result=result,
    )


async def _execute_group_chat_workflow(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    run_view: WorkflowRunView,
    goal: str,
    review_thread: ThreadRecord,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
) -> WorkflowExecutionSummary:
    if not run_view.steps or not run_view.steps[0].threads:
        tracker.failed = True
        return _finalize_workflow_run(
            runtime=runtime,
            workflow_run=workflow_run,
            run_view=run_view,
            goal=goal,
            review_thread=review_thread,
            tracker=tracker,
            cursor=cursor,
            result=None,
        )

    workroom_thread = run_view.steps[0].threads[0]
    participant_ids = _group_chat_participants(runtime, workflow_run.workflow_id)
    runtime.update_task_state(
        task_id=run_view.steps[0].task.id,
        state="in_progress",
        updated_at=cursor.next(),
    )
    runtime.append_message_to_thread(
        thread_id=workroom_thread.id,
        message_id=f"message-{uuid4().hex}",
        sender="workflow",
        kind="assignment",
        body=_render_group_chat_assignment(
            workflow_id=workflow_run.workflow_id,
            goal=goal,
            participants=participant_ids,
        ),
        created_at=cursor.next(),
    )
    participant_executors = [
        _GroupChatParticipantExecutor(
            runtime=runtime,
            thread=workroom_thread,
            agent_id=agent_id,
            workflow_id=workflow_run.workflow_id,
            goal=goal,
            cursor=cursor,
            tracker=tracker,
        )
        for agent_id in participant_ids
    ]
    manager_agent = _build_group_chat_manager(runtime, workflow_run.workflow_id, participant_ids)
    if manager_agent is not None:
        group_chat = GroupChatBuilder(
            participants=participant_executors,
            orchestrator_agent=manager_agent,
            max_rounds=_workflow_max_rounds(runtime, workflow_run.workflow_id),
        ).build()
    else:
        selection_func = _build_group_chat_selection_func(runtime, workflow_run.workflow_id)
        group_chat = GroupChatBuilder(
            participants=participant_executors,
            selection_func=selection_func,
            max_rounds=_workflow_max_rounds(runtime, workflow_run.workflow_id),
        ).build()

    try:
        result = await group_chat.run(goal, include_status_events=True)
    except Exception as exc:
        tracker.failed = True
        tracker.last_thread_id = workroom_thread.id
        runtime.append_event(
            kind="workflow_failed",
            summary=f"Group chat workflow failed: {type(exc).__name__}: {exc}",
            created_at=cursor.next(),
            thread_id=workroom_thread.id,
            task_id=workflow_run.root_task_id,
        )
        return _finalize_workflow_run(
            runtime=runtime,
            workflow_run=workflow_run,
            run_view=run_view,
            goal=goal,
            review_thread=review_thread,
            tracker=tracker,
            cursor=cursor,
            result=None,
        )

    transcript = _read_thread_transcript(runtime, workroom_thread.id)
    tracker.thread_outputs[workroom_thread.id] = transcript
    tracker.last_thread_id = workroom_thread.id
    runtime.update_task_state(
        task_id=run_view.steps[0].task.id,
        state="completed",
        updated_at=cursor.next(),
    )
    await _perform_orchestrator_review(
        runtime=runtime,
        workflow_run=workflow_run,
        review_thread=review_thread,
        goal=goal,
        payload=transcript,
        cursor=cursor,
        tracker=tracker,
    )
    return _finalize_workflow_run(
        runtime=runtime,
        workflow_run=workflow_run,
        run_view=run_view,
        goal=goal,
        review_thread=review_thread,
        tracker=tracker,
        cursor=cursor,
        result=result,
    )


async def _execute_workflow_pass(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    run_view: WorkflowRunView,
    goal: str,
    review_thread: ThreadRecord,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
    start_step_index: int,
    initial_payload: str | list[str],
):
    steps = list(run_view.steps[start_step_index:])
    review_executor = _OrchestratorReviewExecutor(
        runtime=runtime,
        workflow_run=workflow_run,
        review_thread=review_thread,
        goal=goal,
        cursor=cursor,
        tracker=tracker,
    )

    if not steps:
        kickoff = _KickoffExecutor()
        workflow = WorkflowBuilder(
            name=f"workflow-{workflow_run.id}-review",
            description=workflow_run.workflow_id,
            start_executor=kickoff,
            output_executors=[review_executor],
        ).add_edge(kickoff, review_executor).build()
        return await workflow.run(initial_payload, include_status_events=True)

    executors_by_step: list[tuple[_ThreadExecutor, ...]] = []
    for local_index, step in enumerate(steps):
        absolute_index = start_step_index + local_index
        executors_by_step.append(
            tuple(
                _ThreadExecutor(
                    runtime=runtime,
                    thread=thread,
                    workflow_id=workflow_run.workflow_id,
                    step_index=absolute_index,
                    total_steps=len(run_view.steps),
                    goal=goal,
                    cursor=cursor,
                    tracker=tracker,
                )
                for thread in step.threads
            )
        )

    kickoff = _KickoffExecutor()
    builder = WorkflowBuilder(
        name=f"workflow-{workflow_run.id}-{start_step_index}",
        description=workflow_run.workflow_id,
        start_executor=kickoff if len(executors_by_step[0]) > 1 else executors_by_step[0][0],
        output_executors=[review_executor],
    )

    if len(executors_by_step[0]) > 1:
        builder.add_fan_out_edges(kickoff, list(executors_by_step[0]))

    previous_group = executors_by_step[0]
    for current_group in executors_by_step[1:]:
        _connect_executor_group(builder, previous_group, current_group)
        previous_group = current_group
    _connect_executor_group(builder, previous_group, (review_executor,))

    try:
        return await builder.build().run(initial_payload, include_status_events=True)
    except Exception:
        if tracker.blocked_step_index is None:
            tracker.failed = True
        return None


def _connect_executor_group(
    builder: WorkflowBuilder,
    previous_group: tuple[Executor, ...],
    current_group: tuple[Executor, ...],
) -> None:
    if len(previous_group) == 1 and len(current_group) == 1:
        builder.add_edge(previous_group[0], current_group[0])
        return
    if len(previous_group) == 1:
        builder.add_fan_out_edges(previous_group[0], list(current_group))
        return
    if len(current_group) == 1:
        builder.add_fan_in_edges(list(previous_group), current_group[0])
        return
    raise ValueError("workflow step graph requires a fan-in/fan-out stage between grouped steps")


def _workflow_orchestration(runtime: RuntimeContext, workflow_id: str) -> str:
    definition = runtime.registry.workflow_definitions[workflow_id]
    return str(definition.metadata.get("orchestration", "sequential"))


def _workflow_int_metadata(runtime: RuntimeContext, workflow_id: str, key: str, default: int) -> int:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get(key, default)
    if type(value) is not int:
        return default
    return value


def _workflow_step_group_metadata(
    runtime: RuntimeContext,
    workflow_id: str,
    key: str,
) -> tuple[tuple[str, ...], ...]:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"workflow '{workflow_id}' metadata '{key}' must be a list")
    groups: list[tuple[str, ...]] = []
    for group in value:
        if isinstance(group, str):
            if not group:
                raise ValueError(f"workflow '{workflow_id}' metadata '{key}' contains an empty step")
            groups.append((group,))
            continue
        if not isinstance(group, list) or not group:
            raise ValueError(f"workflow '{workflow_id}' metadata '{key}' must contain non-empty lists")
        validated: list[str] = []
        for item in group:
            if not isinstance(item, str) or not item:
                raise ValueError(f"workflow '{workflow_id}' metadata '{key}' contains an invalid step")
            validated.append(item)
        groups.append(tuple(validated))
    return tuple(groups)


def _workflow_max_rounds(runtime: RuntimeContext, workflow_id: str) -> int | None:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get("max_rounds")
    if value is None:
        return 6
    if type(value) is not int or value < 1:
        return 6
    return value


def _workflow_group_chat_manager_mode(runtime: RuntimeContext, workflow_id: str) -> str:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get("group_chat_manager")
    if not isinstance(value, str):
        return "round_robin"
    normalized = value.strip().lower()
    if normalized in {"orchestrator", "round_robin"}:
        return normalized
    return "round_robin"


def _workflow_group_chat_sequence(runtime: RuntimeContext, workflow_id: str) -> tuple[str, ...]:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get("selection_sequence")
    if not isinstance(value, list):
        return ()
    sequence: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            sequence.append(normalized)
    return tuple(sequence)


def _workflow_acceptance_rule(runtime: RuntimeContext, workflow_id: str) -> str:
    acceptance_mode = str(runtime.registry.workflow_definitions[workflow_id].metadata.get("acceptance_mode", "delivery"))
    if acceptance_mode == "decision_ready":
        return "Decide whether the work produced a concrete decision-ready recommendation that addresses the goal."
    if acceptance_mode == "research_brief":
        return "Decide whether the work produced a concrete research brief with enough evidence for the orchestrator to choose the next step."
    if acceptance_mode == "design_brief":
        return "Decide whether the work produced a concrete design brief that is implementation-ready and aligned with the goal."
    if acceptance_mode == "revised_plan":
        return "Decide whether the work produced an explicit revised plan that realigns the project and is actionable."
    return "Decide whether the work satisfies the goal and represents a minimal working delivery."


def _build_group_chat_manager(
    runtime: RuntimeContext,
    workflow_id: str,
    participants: Sequence[str],
) -> Agent | None:
    if _workflow_group_chat_manager_mode(runtime, workflow_id) != "orchestrator":
        return None
    try:
        orchestrator = runtime.build_agent("orchestrator")
    except (KeyError, ValueError):
        return None
    client = getattr(orchestrator, "client", None)
    if client is None:
        return None
    participant_list = ", ".join(participants)
    return Agent(
        client=client,
        id=f"{workflow_id}-group-chat-manager",
        name="Workroom Manager",
        description="Routes the next speaker in a shared agent discussion",
        instructions="\n".join(
            [
                "You manage a multi-agent workroom.",
                f"The active participants are: {participant_list}.",
                "Your job is to pick who should speak next and when the discussion should end.",
                "Do not solve the task yourself. Route the discussion.",
                "Terminate only when the conversation has produced a clear decision-ready recommendation for the orchestrator.",
                "Always return valid JSON matching the schema requested by the workflow runtime.",
            ]
        ),
    )


def _group_chat_agent(base_agent, runtime: RuntimeContext, workflow_id: str, agent_id: str) -> Agent:
    client = getattr(base_agent, "client", None)
    if client is None:
        return base_agent
    definition = runtime.registry.agent_definitions[agent_id]
    instructions = compose_instructions(definition)
    extra = _group_chat_agent_instructions(runtime, workflow_id, agent_id)
    if extra:
        instructions = f"{instructions}\n\n{extra}".strip()
    return Agent(
        client=client,
        id=f"{agent_id}-group-chat",
        name=base_agent.name,
        description=base_agent.description,
        instructions=instructions,
        context_providers=getattr(base_agent, "context_providers", None),
        default_options=getattr(base_agent, "default_options", None),
    )


def _group_chat_agent_instructions(runtime: RuntimeContext, workflow_id: str, agent_id: str) -> str:
    acceptance_mode = str(runtime.registry.workflow_definitions[workflow_id].metadata.get("acceptance_mode", "delivery"))
    if acceptance_mode == "decision_ready":
        lines = [
            "You are in a shared strategy discussion, not a code review or implementation task.",
            "Work under reasonable explicit assumptions when the repo does not provide enough context.",
            "Do not stop at asking for more context if the goal itself can be answered with a concrete recommendation.",
            "Every turn should move the discussion toward a clear recommendation.",
        ]
        if agent_id == "reviewer":
            lines.extend(
                [
                    "Review the strength of the arguments, call out the most important assumptions, and then state the strongest recommendation plainly.",
                    "Do not require file diffs, tests, or repository validation unless the goal explicitly asks for them.",
                ]
            )
        return "\n".join(lines)
    if acceptance_mode in {"research_brief", "design_brief", "revised_plan"}:
        return (
            "You are in a collaborative planning discussion. Focus on producing the requested brief or plan, "
            "not on requesting implementation evidence."
        )
    return ""


def _round_robin_group_chat_selector(state) -> str | None:
    spoken = [
        message.author_name
        for message in state.conversation
        if message.role == "assistant" and isinstance(message.author_name, str)
    ]
    for participant in state.participants:
        if participant not in spoken:
            return participant
    return None


def _build_group_chat_selection_func(runtime: RuntimeContext, workflow_id: str):
    sequence = _workflow_group_chat_sequence(runtime, workflow_id)
    if not sequence:
        return _round_robin_group_chat_selector

    def _sequence_selector(state) -> str | None:
        index = state.current_round
        if index >= len(sequence):
            return None
        target = sequence[index]
        if target not in state.participants:
            raise RuntimeError(f"Selection sequence returned unknown participant '{target}'.")
        return target

    return _sequence_selector


def _group_chat_participants(runtime: RuntimeContext, workflow_id: str) -> tuple[str, ...]:
    groups = runtime.workflow_step_groups(workflow_id)
    participant_ids: list[str] = []
    for group in groups:
        for agent_id in group:
            if agent_id not in participant_ids:
                participant_ids.append(agent_id)
    return tuple(participant_ids)


def _append_thread_output(
    *,
    tracker: _ExecutionTracker,
    thread_id: str,
    speaker: str,
    body: str,
) -> None:
    entry = f"{speaker}:\n{body.strip()}"
    existing = tracker.thread_outputs.get(thread_id, "").strip()
    tracker.thread_outputs[thread_id] = entry if not existing else f"{existing}\n\n{entry}"


def _read_thread_transcript(runtime: RuntimeContext, thread_id: str) -> str:
    messages = runtime.list_thread_messages(thread_id)
    lines: list[str] = []
    for message in messages:
        body = runtime.conversation_store.read_message_body(message).rstrip("\n")
        if not body:
            continue
        lines.append(f"{message.sender}:\n{body}")
    return "\n\n".join(lines).strip()


def _render_group_chat_assignment(
    *,
    workflow_id: str,
    goal: str,
    participants: Sequence[str],
) -> str:
    lines = [
        f"Workflow: {workflow_id}",
        "",
        "Participants:",
        *[f"- {participant}" for participant in participants],
        "",
        "Goal:",
        goal,
        "",
        "Discuss the tradeoffs directly and leave the thread with a decision-ready recommendation.",
    ]
    return "\n".join(lines).strip()


def _max_repair_cycles(runtime: RuntimeContext, workflow_id: str) -> int:
    return max(0, _workflow_int_metadata(runtime, workflow_id, "max_repair_cycles", 0))


def _max_replan_cycles(runtime: RuntimeContext, workflow_id: str) -> int:
    return max(0, _workflow_int_metadata(runtime, workflow_id, "max_replan_cycles", 0))


def _repair_step_groups(runtime: RuntimeContext, workflow_id: str) -> tuple[tuple[str, ...], ...]:
    groups = _workflow_step_group_metadata(runtime, workflow_id, "repair_step_groups")
    if groups:
        return groups
    return (("fixer",), ("reviewer",))


def _replan_step_groups(runtime: RuntimeContext, workflow_id: str) -> tuple[tuple[str, ...], ...]:
    return _workflow_step_group_metadata(runtime, workflow_id, "replan_step_groups")


def _next_followup_cycle(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    tracker: _ExecutionTracker,
) -> tuple[str, tuple[tuple[str, ...], ...], str, str] | None:
    if tracker.review_requires_replan:
        replan_groups = _replan_step_groups(runtime, workflow_id)
        if replan_groups and tracker.replan_cycles < _max_replan_cycles(runtime, workflow_id):
            tracker.replan_cycles += 1
            summary = tracker.review_replan_summary or f"Starting replanning cycle {tracker.replan_cycles} for {workflow_id}"
            return ("replan", replan_groups, "workflow_auto_replan_started", summary)

    repair_groups = _repair_step_groups(runtime, workflow_id)
    if repair_groups and tracker.repair_cycles < _max_repair_cycles(runtime, workflow_id):
        tracker.repair_cycles += 1
        return (
            "repair",
            repair_groups,
            "workflow_auto_repair_started",
            f"Starting automatic fix cycle {tracker.repair_cycles} for {workflow_id}",
        )

    replan_groups = _replan_step_groups(runtime, workflow_id)
    if replan_groups and tracker.replan_cycles < _max_replan_cycles(runtime, workflow_id):
        tracker.replan_cycles += 1
        return (
            "replan",
            replan_groups,
            "workflow_auto_replan_started",
            f"Escalating {workflow_id} to replanning after the current approach was rejected",
        )

    return None


def _render_followup_payload(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    goal: str,
    tracker: _ExecutionTracker,
    run_view: WorkflowRunView,
    cycle_kind: str,
) -> str:
    changed_files = runtime._workflow_changed_files(run_view.workflow_run.id)
    lines = [
        f"Workflow: {workflow_id}",
        f"Follow-up: {cycle_kind}",
        "",
        "Goal:",
        goal,
    ]
    if tracker.review_summary:
        lines.extend(["", "Latest orchestrator review:", tracker.review_summary])
    if tracker.review_findings:
        lines.extend(["", "Specific findings to address:"])
        lines.extend(f"- {finding}" for finding in tracker.review_findings)
    if changed_files:
        lines.extend(["", "Files touched so far:"])
        lines.extend(f"- {path}" for path in changed_files)
    step_outputs = _recent_step_outputs(run_view=run_view, tracker=tracker, limit=3)
    if step_outputs:
        lines.extend(["", "Recent workflow evidence:"])
        lines.extend(step_outputs)
    if cycle_kind == "replan":
        lines.extend(
            [
                "",
                "Revise the plan explicitly before continuing. Update the approach instead of making a narrow patch.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Fix the identified issues in the actual workspace, then re-verify and hand the work back for review.",
            ]
        )
    return "\n".join(lines).strip()


def _recent_step_outputs(
    *,
    run_view: WorkflowRunView,
    tracker: _ExecutionTracker,
    limit: int,
) -> list[str]:
    evidence: list[str] = []
    for step in reversed(run_view.steps):
        for thread in reversed(step.threads):
            body = tracker.thread_outputs.get(thread.id)
            if not body:
                continue
            label = thread.summary or thread.assigned_agent_id or thread.id
            evidence.append(f"- {label}: {_truncate_text(body, 600)}")
            if len(evidence) >= limit:
                return list(reversed(evidence))
    return list(reversed(evidence))


def _finalize_workflow_run(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    run_view: WorkflowRunView,
    goal: str,
    review_thread: ThreadRecord,
    tracker: _ExecutionTracker,
    cursor: _TimestampCursor,
    result,
) -> WorkflowExecutionSummary:
    steps = list(run_view.steps)
    if tracker.failed:
        completed_count = 0
        for step in steps:
            if all(thread.id in tracker.thread_outputs for thread in step.threads):
                completed_count += 1
                continue
            break
        state = "failed"
    elif tracker.blocked_step_index is None:
        completed_count = len(steps)
        state = "completed" if tracker.review_accepted is not False else "blocked"
    else:
        completed_count = tracker.blocked_step_index
        state = "blocked"

    for step_index, step in enumerate(steps):
        if step_index < completed_count:
            runtime.update_task_state(
                task_id=step.task.id,
                state="completed",
                updated_at=cursor.next(),
            )
            tracker.completed_step_indices.add(step_index)
        elif tracker.blocked_step_index == step_index:
            runtime.update_task_state(
                task_id=step.task.id,
                state="blocked",
                updated_at=cursor.next(),
            )

    if run_view.root_task is not None:
        runtime.update_task_state(
            task_id=run_view.root_task.id,
            state=state,
            updated_at=cursor.next(),
        )

    updated = type(workflow_run)(
        id=workflow_run.id,
        session_id=workflow_run.session_id,
        workflow_id=workflow_run.workflow_id,
        state=state,
        created_at=workflow_run.created_at,
        updated_at=cursor.next(),
        root_task_id=workflow_run.root_task_id,
        current_step_index=completed_count,
        last_thread_id=tracker.last_thread_id,
    )
    runtime.workflow_store.update_workflow_run(updated)

    if state in {"completed", "blocked", "failed"}:
        tracker.artifact_id = runtime.create_artifact(
            artifact_id=f"artifact-{uuid4().hex[:8]}",
            kind="workflow-report",
            title=f"Workflow Report: {workflow_run.workflow_id}",
            content=_render_workflow_report(
                workflow_run=updated,
                run_view=run_view,
                goal=goal,
                tracker=tracker,
                result=result,
            ),
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=workflow_run.root_task_id,
        ).id
    if state == "completed":
        runtime.append_event(
            kind="workflow_completed",
            summary=f"Completed workflow {workflow_run.workflow_id}",
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=workflow_run.root_task_id,
        )
    elif state == "blocked" and tracker.review_accepted is False:
        runtime.append_event(
            kind="workflow_rejected",
            summary=f"Workflow {workflow_run.workflow_id} rejected by orchestrator review",
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=workflow_run.root_task_id,
        )
    elif state == "blocked":
        runtime.append_event(
            kind="workflow_blocked",
            summary=f"Workflow {workflow_run.workflow_id} blocked",
            created_at=cursor.next(),
            thread_id=tracker.blocked_thread_id,
            task_id=workflow_run.root_task_id,
        )
    else:
        runtime.append_event(
            kind="workflow_failed",
            summary=f"Workflow {workflow_run.workflow_id} failed during execution",
            created_at=cursor.next(),
            thread_id=tracker.last_thread_id,
            task_id=workflow_run.root_task_id,
        )

    return WorkflowExecutionSummary(
        workflow_run_id=workflow_run.id,
        workflow_id=workflow_run.workflow_id,
        state=state,
        current_step_index=completed_count,
        last_thread_id=tracker.last_thread_id,
        review_thread_id=review_thread.id,
        review_summary=tracker.review_summary,
        review_accepted=tracker.review_accepted,
        artifact_id=tracker.artifact_id,
    )


def _render_step_prompt(
    *,
    workflow_id: str,
    goal: str,
    step_index: int,
    total_steps: int,
    agent_id: str,
    payload: str | list[str],
) -> str:
    role_rules = _step_role_rules(agent_id)
    lines = [
        f"Workflow: {workflow_id}",
        f"Step: {step_index + 1}/{total_steps}",
        f"Assigned agent: {agent_id}",
        "",
        "Goal:",
        goal,
        "",
        "Role requirements:",
        role_rules,
        "",
    ]
    if isinstance(payload, list):
        lines.append("Upstream outputs:")
        for index, item in enumerate(payload, start=1):
            lines.extend([f"{index}.", item, ""])
    else:
        lines.extend(["Current input:", payload, ""])
    lines.append("Return the best next result for your role.")
    return "\n".join(lines).strip()


def _render_review_prompt(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    goal: str,
    payload: str | list[str],
    workspace_files: list[str],
    prior_findings: tuple[str, ...] = (),
) -> str:
    lines = [
        f"Review workflow {workflow_id} as the orchestrator.",
        "",
        "Goal:",
        _truncate_text(goal, 400),
        "",
        "Workspace files:",
    ]
    if workspace_files:
        lines.extend(f"- {path}" for path in workspace_files)
    else:
        lines.append("(no files)")
    lines.append("")
    if isinstance(payload, list):
        lines.append("Final upstream outputs:")
        for index, item in enumerate(payload, start=1):
            lines.extend([f"{index}.", _truncate_text(item, 800), ""])
    else:
        lines.extend(["Final workflow output:", _truncate_text(payload, 1200), ""])
    if prior_findings:
        lines.append("Previous findings that should now be addressed:")
        lines.extend(f"- {finding}" for finding in prior_findings)
        lines.append("")
    lines.extend(
        [
            _workflow_acceptance_rule(runtime, workflow_id),
            "If the work is not acceptable, include specific findings.",
            "Set `requires_replan` to true only when the current approach is wrong enough that replanning is better than another narrow fix cycle.",
            "Do not create new requirements beyond the stated goal.",
            "Return JSON only with this shape:",
            '{"accepted": true, "summary": "one concise review summary", "findings": ["specific issue"], "requires_replan": false, "replan_summary": ""}',
        ]
    )
    return "\n".join(lines).strip()


def _step_role_rules(agent_id: str) -> str:
    if agent_id == "architect":
        return (
            "Produce a concrete handoff for implementation: proposed files, responsibilities, CLI behavior, "
            "and any key edge cases. Do not stop at saying you will design it."
        )
    if agent_id == "coder":
        return (
            "Produce actual repo changes in this step. Use file tools to create or modify the implementation. "
            "A prose-only answer is not sufficient. Keep the solution minimal, avoid extra files unless required, "
            "and run at most two focused verification commands."
        )
    if agent_id == "tester":
        return (
            "Verify the actual implementation with focused commands. Report the concrete commands and outcomes. "
            "If something is missing or broken, say so plainly. Run at most two focused verification commands."
        )
    if agent_id == "reviewer":
        return (
            "Review the actual workspace state and verification evidence. Approve only if the requested deliverable exists "
            "and the essential behavior is covered. Run at most two focused checks."
        )
    return "Move the assigned work forward concretely."


def _render_workflow_report(
    *,
    workflow_run,
    run_view,
    goal: str,
    tracker: _ExecutionTracker,
    result,
) -> str:
    lines = [
        f"# Workflow Report: {workflow_run.workflow_id}",
        "",
        f"- Run ID: {workflow_run.id}",
        f"- Status: {workflow_run.state}",
        f"- Goal: {goal}",
        f"- Final Thread: {tracker.last_thread_id or 'n/a'}",
    ]
    if result is not None:
        try:
            final_state = result.get_final_state()
        except Exception:
            final_state = None
        if final_state is not None:
            lines.append(f"- Agent Framework State: {final_state}")
    lines.extend(["", "## Step Outputs"])
    for step_index, step in enumerate(run_view.steps, start=1):
        agents = ", ".join(thread.assigned_agent_id or thread.summary or thread.id for thread in step.threads)
        lines.extend([f"### Step {step_index}: {agents}"])
        for thread in step.threads:
            body = tracker.thread_outputs.get(thread.id, "(no output)")
            lines.extend([f"- {thread.assigned_agent_id or thread.summary or thread.id}", body])
        lines.append("")
    lines.extend(["## Orchestrator Review", tracker.review_summary or "(no review summary)"])
    if tracker.review_findings:
        lines.extend(["", "## Findings", *[f"- {finding}" for finding in tracker.review_findings]])
    if tracker.repair_cycles:
        lines.extend(["", f"- Automatic repair cycles: {tracker.repair_cycles}"])
    if tracker.replan_cycles:
        lines.extend(["", f"- Automatic replanning cycles: {tracker.replan_cycles}"])
    return "\n".join(lines).strip()


def _supports_auto_repair(workflow_id: str) -> bool:
    return workflow_id in {
        "standard-build",
        "single-agent-execution",
        "best-of-n",
        "review-repair-loop",
        "review-driven-repair",
        "test-driven-repair",
    }


def _required_tool_names(agent_id: str) -> tuple[str, ...]:
    requirements = {
        "coder": ("write_file", "patch_file"),
        "fixer": ("write_file", "patch_file", "run_command"),
        "tester": ("run_command",),
        "reviewer": ("list_files", "read_file", "search_files", "run_command"),
    }
    return requirements.get(agent_id, ())


def _has_required_tool_calls(
    *,
    tool_calls,
    thread_id: str,
    new_tool_ids: set[str],
    required_tools: tuple[str, ...],
) -> bool:
    required = set(required_tools)
    for tool_call in tool_calls:
        if tool_call.thread_id != thread_id:
            continue
        if tool_call.id in new_tool_ids:
            continue
        if tool_call.status != "completed":
            continue
        if tool_call.tool_name in required:
            return True
    return False


def _render_tool_requirement_retry(agent_id: str) -> str:
    if agent_id == "coder":
        return (
            "Your last turn did not record any real file-edit tool call. "
            "Use `write_file` or `patch_file` now and then summarize the actual implementation."
        )
    if agent_id == "fixer":
        return (
            "Your last turn did not record any real repair action. "
            "Use `write_file`, `patch_file`, or `run_command` now to apply and verify the fix."
        )
    if agent_id == "tester":
        return (
            "Your last turn did not record any real verification command. "
            "Use `run_command` now against the actual implementation and report the concrete result."
        )
    if agent_id == "reviewer":
        return (
            "Your last turn did not record any real inspection or verification tool use. "
            "Use `list_files`, `read_file`, `search_files`, or `run_command` now, then return the review."
        )
    return "Use the appropriate workspace tools now and return the concrete result."


def _orchestrator_review_instructions() -> str:
    return "\n".join(
        [
            "You are a dedicated workflow acceptance reviewer.",
            "Judge whether the work satisfies the stated goal and whether it is a minimal working delivery.",
            "Do not suggest optional polish.",
            "Do not delegate, do not plan, and do not call tools.",
            "Output JSON only with this shape:",
            '{"accepted": true, "summary": "one concise review summary", "findings": ["specific issue"], "requires_replan": false, "replan_summary": ""}',
        ]
    )


def _truncate_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ..."


def _parse_review_verdict(raw: str) -> WorkflowReviewVerdict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("no JSON object found in review verdict")
        text = match.group(0)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("review verdict must be an object")
    accepted = bool(payload.get("accepted"))
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        summary = "No review summary provided."
    findings_raw = payload.get("findings", [])
    findings: list[str] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                findings.append(stripped)
    requires_replan = bool(payload.get("requires_replan"))
    replan_summary = str(payload.get("replan_summary", "")).strip() or None
    return WorkflowReviewVerdict(
        accepted=accepted,
        summary=summary,
        findings=tuple(findings),
        requires_replan=requires_replan,
        replan_summary=replan_summary,
    )


def _format_review_summary(verdict: WorkflowReviewVerdict) -> str:
    prefix = "ACCEPTED" if verdict.accepted else "REJECTED"
    return f"{prefix}: {verdict.summary}"
