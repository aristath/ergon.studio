from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
import json
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_framework import Agent, AgentResponse, AgentResponseUpdate, Executor, Message, WorkflowBuilder, WorkflowContext, handler
from agent_framework.orchestrations import GroupChatBuilder, HandoffBuilder, MagenticBuilder, clean_conversation_for_handoff
from agent_framework_orchestrations._base_group_chat_orchestrator import GroupChatParticipantMessage, GroupChatRequestMessage, GroupChatResponseMessage

from ergon_studio.agent_factory import compose_instructions
from ergon_studio.context_providers import WORKSPACE_STATE_KEY
from ergon_studio.tool_context import ToolExecutionContext, use_tool_execution_context
from ergon_studio.workflow_policy import acceptance_mode_for_metadata, acceptance_rule_for_mode, is_decision_ready_acceptance_mode, is_planning_acceptance_mode, step_groups_for_metadata

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
    blocked_summary: str | None = None


@dataclass
class _ExecutionTracker:
    completed_step_indices: set[int] = field(default_factory=set)
    thread_outputs: dict[str, str] = field(default_factory=dict)
    thread_tool_modes: dict[str, str] = field(default_factory=dict)
    evidence_gaps: dict[str, str] = field(default_factory=dict)
    blocked_step_index: int | None = None
    blocked_thread_id: str | None = None
    blocked_summary: str | None = None
    blocked_reason: str | None = None
    failed: bool = False
    last_thread_id: str | None = None
    review_summary: str | None = None
    review_accepted: bool | None = None
    artifact_id: str | None = None
    repair_cycles: int = 0
    replan_cycles: int = 0
    clarification_cycles: int = 0
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


@dataclass(frozen=True)
class WorkflowFollowupPlan:
    cycle_kind: str
    step_groups: tuple[tuple[str, ...], ...]
    event_kind: str
    event_summary: str
    payload: str
    tool_mode: str = "default"


@dataclass(frozen=True)
class WorkflowFollowupDecision:
    action: str
    summary: str
    agent_id: str | None = None
    request: str | None = None
    step_groups: tuple[tuple[str, ...], ...] = ()
    tool_mode: str = "default"


@dataclass
class _ProjectedThreadDraft:
    draft_id: str
    sender: str
    body: str = ""


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
            self.tracker.blocked_summary = (
                f"{self.thread.assigned_agent_id or self.thread.id} did not produce a reply."
            )
            self.tracker.blocked_reason = "no_reply"
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
        tool_mode = self.tracker.thread_tool_modes.get(self.thread.id, "default")
        if tool_mode == "none":
            return reply, reply_body
        required_tools = _required_tool_names_for_workflow(
            runtime=self.runtime,
            workflow_id=self.workflow_id,
            agent_id=self.thread.assigned_agent_id or "",
        )
        if not required_tools:
            return reply, reply_body
        if _has_required_tool_calls(
            tool_calls=self.runtime.list_tool_calls(),
            thread_id=self.thread.id,
            new_tool_ids=before_tool_ids,
            required_tools=required_tools,
        ):
            self.tracker.evidence_gaps.pop(self.thread.id, None)
            return reply, reply_body

        self.runtime.append_event(
            kind="workflow_step_retry",
            summary=f"Retrying {self.thread.assigned_agent_id or self.thread.id} because required tool evidence is missing",
            created_at=self.cursor.next(),
            thread_id=self.thread.id,
            task_id=self.thread.parent_task_id,
        )
        retry_prompt = _render_tool_requirement_retry(
            agent_id=self.thread.assigned_agent_id or "",
            required_tools=required_tools,
        )
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
            self.tracker.blocked_summary = (
                f"{self.thread.assigned_agent_id or self.thread.id} did not answer the retry request for tool-backed evidence."
            )
            self.tracker.blocked_reason = "retry_no_reply"
            self.tracker.last_thread_id = self.thread.id
            raise RuntimeError(f"workflow step blocked after retry: {self.thread.assigned_agent_id or self.thread.id}")

        retry_body = self.runtime.conversation_store.read_message_body(retry_reply).rstrip("\n")
        if _has_required_tool_calls(
            tool_calls=self.runtime.list_tool_calls(),
            thread_id=self.thread.id,
            new_tool_ids=before_retry_ids,
            required_tools=required_tools,
        ):
            self.tracker.evidence_gaps.pop(self.thread.id, None)
            return retry_reply, retry_body

        evidence_gap = (
            f"{self.thread.assigned_agent_id or self.thread.id} replied without the required tool evidence "
            f"({', '.join(required_tools)})."
        )
        self.tracker.evidence_gaps[self.thread.id] = evidence_gap
        self.tracker.last_thread_id = self.thread.id
        self.runtime.append_event(
            kind="workflow_step_missing_evidence",
            summary=evidence_gap,
            created_at=self.cursor.next(),
            thread_id=self.thread.id,
            task_id=self.thread.parent_task_id,
        )
        return retry_reply, retry_body


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

        response_text = await self._run_group_chat_turn(messages)
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

    async def _run_group_chat_turn(self, messages: list[Message]) -> str | None:
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
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            agent_id=self.agent_id,
            session_factory=lambda session_id: base_agent.create_session(session_id=session_id),
        )
        session.state[WORKSPACE_STATE_KEY] = {
            "session_id": self.thread.session_id,
            "thread_id": self.thread.id,
            "task_id": self.thread.parent_task_id,
            "agent_id": self.agent_id,
            "created_at": self.cursor.next(),
        }
        tool_context = ToolExecutionContext(
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            task_id=self.thread.parent_task_id,
            agent_id=self.agent_id,
        )
        agent = _group_chat_agent(base_agent, self.runtime, self.workflow_id, self.agent_id)
        try:
            with use_tool_execution_context(tool_context):
                response_stream = self.runtime._stream_visible_agent_reply(
                    thread_id=self.thread.id,
                    reply_sender=self.agent_id,
                    kind="chat",
                    created_at=self.cursor.next(),
                    run_callable=lambda: agent.run(messages=messages, session=session, stream=True),
                    persist_reply=lambda final_text, reply_created_at: self.runtime.append_message_to_thread(
                        thread_id=self.thread.id,
                        message_id=f"message-{uuid4().hex}",
                        sender=self.agent_id,
                        kind="chat",
                        body=final_text,
                        created_at=self.cursor.next(),
                    ),
                    on_exception=lambda exc, event_created_at: self.runtime.append_event(
                        kind="group_chat_participant_failed",
                        summary=f"{self.agent_id} failed in group chat: {type(exc).__name__}: {exc}",
                        created_at=event_created_at,
                        thread_id=self.thread.id,
                        task_id=self.thread.parent_task_id,
                    ),
                    on_empty_response=lambda event_created_at: self.runtime.append_event(
                        kind="group_chat_participant_failed",
                        summary=f"{self.agent_id} returned an empty response in group chat",
                        created_at=event_created_at,
                        thread_id=self.thread.id,
                        task_id=self.thread.parent_task_id,
                    ),
                )
                async for _ in response_stream:
                    pass
                response_text, _reply_message = await response_stream.get_final_response()
        except Exception as exc:
            self.runtime.append_event(
                kind="group_chat_participant_failed",
                summary=f"{self.agent_id} failed in group chat: {type(exc).__name__}: {exc}",
                created_at=self.cursor.next(),
                thread_id=self.thread.id,
                task_id=self.thread.parent_task_id,
            )
            self.runtime.agent_session_store.save_session(
                session_id=self.thread.session_id,
                thread_id=self.thread.id,
                agent_id=self.agent_id,
                session=session,
            )
            raise
        self.runtime.agent_session_store.save_session(
            session_id=self.thread.session_id,
            thread_id=self.thread.id,
            agent_id=self.agent_id,
            session=session,
        )
        return response_text


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
        workflow_run=workflow_run,
        workflow_id=workflow_run.workflow_id,
        goal=goal,
        payload=payload,
        workspace_files=runtime._workspace_file_list(limit=16),
        tracker=tracker,
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
    runtime.track_token_usage(response)
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
    if _workflow_orchestration(runtime, workflow_run.workflow_id) == "magentic":
        return await _execute_magentic_workflow(
            runtime=runtime,
            workflow_run=workflow_run,
            run_view=run_view,
            goal=goal,
            review_thread=review_thread,
            cursor=cursor,
            tracker=tracker,
        )
    if _workflow_orchestration(runtime, workflow_run.workflow_id) == "handoff":
        return await _execute_handoff_workflow(
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
        if tracker.failed:
            break
        followup = await _decide_followup_action(
            runtime=runtime,
            workflow_run=active_workflow_run,
            run_view=active_run_view,
            goal=goal,
            review_thread=review_thread,
            cursor=cursor,
            tracker=tracker,
        )
        if tracker.blocked_step_index is None and tracker.review_accepted is not False:
            break
        if followup is None:
            break
        runtime.append_event(
            kind=followup.event_kind,
            summary=followup.event_summary,
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=active_workflow_run.root_task_id,
        )
        next_state = {
            "repair": "repairing",
            "replan": "replanning",
            "clarify": "clarifying",
        }.get(followup.cycle_kind, "in_progress")
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
        active_workflow_run, new_threads = runtime.request_workflow_followup_cycle(
            workflow_run_id=active_workflow_run.id,
            created_at=cursor.next(),
            step_groups=followup.step_groups,
            state=next_state,
            event_kind=(
                "workflow_clarification_cycle_requested"
                if followup.cycle_kind == "clarify"
                else f"workflow_{followup.cycle_kind}_cycle_requested"
            ),
            event_summary=followup.event_summary,
        )
        if followup.tool_mode != "default":
            for thread in new_threads:
                tracker.thread_tool_modes[thread.id] = followup.tool_mode
        tracker.blocked_step_index = None
        tracker.blocked_thread_id = None
        tracker.blocked_summary = None
        tracker.blocked_reason = None
        refreshed_view = runtime.describe_workflow_run(active_workflow_run.id)
        if refreshed_view is None:
            tracker.failed = True
            break
        start_step_index = len(active_run_view.steps)
        active_run_view = refreshed_view
        initial_payload = followup.payload

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


async def _execute_magentic_workflow(
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
    participant_ids = _workflow_participants(runtime, workflow_run.workflow_id)
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
    manager_agent = _build_magentic_manager_agent(runtime, workflow_run.workflow_id, participant_ids)
    workflow = MagenticBuilder(
        participants=participant_executors,
        manager_agent=manager_agent,
        max_round_count=_workflow_max_rounds(runtime, workflow_run.workflow_id),
        enable_plan_review=False,
    ).build()

    try:
        result = await _run_streamed_magentic_workflow(
            runtime=runtime,
            workflow=workflow,
            thread=workroom_thread,
            task_id=workflow_run.root_task_id,
            cursor=cursor,
            tracker=tracker,
            goal=goal,
        )
    except Exception as exc:
        tracker.failed = True
        tracker.last_thread_id = workroom_thread.id
        runtime.append_event(
            kind="workflow_failed",
            summary=f"Magentic workflow failed: {type(exc).__name__}: {exc}",
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

    _append_output_messages_to_thread(
        runtime=runtime,
        thread=workroom_thread,
        outputs=result.get_outputs(),
        cursor=cursor,
        tracker=tracker,
        assistant_only=False,
    )
    if result.get_request_info_events():
        request_lines = _request_info_lines(result.get_request_info_events())
        blocked_summary = request_lines[2] if len(request_lines) >= 3 else request_lines[0]
        tracker.blocked_step_index = 0
        tracker.blocked_thread_id = workroom_thread.id
        tracker.blocked_summary = blocked_summary
        tracker.blocked_reason = "request_info"
        tracker.last_thread_id = workroom_thread.id
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


async def _execute_handoff_workflow(
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
    participant_ids = _workflow_participants(runtime, workflow_run.workflow_id)
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
    agents = _build_handoff_agents(runtime, workflow_run.workflow_id, participant_ids)
    builder = HandoffBuilder(
        name=f"workflow-{workflow_run.id}-handoff",
        participants=list(agents.values()),
        termination_condition=_build_handoff_termination_condition(runtime, workflow_run.workflow_id),
    ).with_start_agent(agents[_handoff_start_agent(runtime, workflow_run.workflow_id, participant_ids)])
    builder = builder.with_autonomous_mode(
        agents=list(_handoff_autonomous_agents(runtime, workflow_run.workflow_id, participant_ids)),
        turn_limits={
            agent_id: _handoff_autonomous_turn_limit(runtime, workflow_run.workflow_id)
            for agent_id in participant_ids
        },
    )
    _apply_handoff_topology(builder, runtime, workflow_run.workflow_id, agents, participant_ids)
    workflow = builder.build()
    executor_senders = _handoff_executor_sender_map(agents)

    try:
        result = await _run_streamed_handoff_workflow(
            runtime=runtime,
            workflow=workflow,
            thread=workroom_thread,
            executor_senders=executor_senders,
            task_id=workflow_run.root_task_id,
            cursor=cursor,
            tracker=tracker,
            goal=goal,
        )
    except Exception as exc:
        tracker.failed = True
        tracker.last_thread_id = workroom_thread.id
        runtime.append_event(
            kind="workflow_failed",
            summary=f"Handoff workflow failed: {type(exc).__name__}: {exc}",
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

    request_info_events = result.get_request_info_events()
    if request_info_events:
        request_lines = _request_info_lines(request_info_events)
        blocked_summary = request_lines[2] if len(request_lines) >= 3 else request_lines[0]
        tracker.blocked_step_index = 0
        tracker.blocked_thread_id = workroom_thread.id
        tracker.blocked_summary = blocked_summary
        tracker.blocked_reason = "request_info"
        tracker.last_thread_id = workroom_thread.id
        runtime.append_event(
            kind="workflow_info_requested",
            summary=blocked_summary,
            created_at=cursor.next(),
            thread_id=workroom_thread.id,
            task_id=workflow_run.root_task_id,
        )
        runtime.append_message_to_thread(
            thread_id=workroom_thread.id,
            message_id=f"message-{uuid4().hex}",
            sender="workflow",
            kind="question",
            body="\n".join(request_lines),
            created_at=cursor.next(),
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
    return step_groups_for_metadata(
        workflow_id=workflow_id,
        metadata=runtime.registry.workflow_definitions[workflow_id].metadata,
        metadata_key=key,
    )


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


def _workflow_participants(runtime: RuntimeContext, workflow_id: str) -> tuple[str, ...]:
    groups = runtime.workflow_step_groups(workflow_id)
    participant_ids: list[str] = []
    for group in groups:
        for agent_id in group:
            if agent_id not in participant_ids:
                participant_ids.append(agent_id)
    return tuple(participant_ids)


def _workflow_acceptance_rule(runtime: RuntimeContext, workflow_id: str) -> str:
    acceptance_mode = acceptance_mode_for_metadata(
        runtime.registry.workflow_definitions[workflow_id].metadata
    )
    return acceptance_rule_for_mode(acceptance_mode)


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
    acceptance_mode = acceptance_mode_for_metadata(
        runtime.registry.workflow_definitions[workflow_id].metadata
    )
    if is_decision_ready_acceptance_mode(acceptance_mode):
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
    if is_planning_acceptance_mode(acceptance_mode):
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
    return _workflow_participants(runtime, workflow_id)


def _build_magentic_manager_agent(
    runtime: RuntimeContext,
    workflow_id: str,
    participants: Sequence[str],
) -> Agent:
    orchestrator = runtime.build_agent("orchestrator")
    client = getattr(orchestrator, "client", None)
    if client is None:
        raise ValueError("orchestrator client is unavailable for magentic workflow management")
    participant_list = ", ".join(participants)
    return Agent(
        client=client,
        id=f"{workflow_id}-magentic-manager",
        name="Adaptive Workflow Manager",
        description="Coordinates dynamic specialist delegation",
        instructions="\n".join(
            [
                "You are the adaptive manager for a local AI software team.",
                f"The available participants are: {participant_list}.",
                "Make a concrete plan, pick the next best specialist, track progress, and replan when necessary.",
                "Do not implement the task yourself unless producing the final manager answer is required.",
                "Use the specialists to move the task forward and keep the final answer concrete.",
            ]
        ),
        context_providers=getattr(orchestrator, "context_providers", None),
        default_options=getattr(orchestrator, "default_options", None),
    )


def _build_handoff_agents(
    runtime: RuntimeContext,
    workflow_id: str,
    participants: Sequence[str],
) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for agent_id in participants:
        base_agent = runtime.build_agent(agent_id)
        client = getattr(base_agent, "client", None)
        if client is None:
            raise ValueError(f"{agent_id} is unavailable for handoff workflow execution")
        definition = runtime.registry.agent_definitions[agent_id]
        instructions = compose_instructions(definition)
        extra = _handoff_agent_instructions(runtime, workflow_id, agent_id)
        if extra:
            instructions = f"{instructions}\n\n{extra}".strip()
        agents[agent_id] = Agent(
            client=client,
            id=base_agent.id,
            name=base_agent.name,
            description=base_agent.description,
            instructions=instructions,
            context_providers=getattr(base_agent, "context_providers", None),
            default_options=getattr(base_agent, "default_options", None),
        )
    return agents


def _handoff_executor_sender_map(agents: dict[str, Agent]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sender_id, agent in agents.items():
        aliases = {sender_id}
        agent_id = getattr(agent, "id", None)
        if isinstance(agent_id, str) and agent_id.strip():
            aliases.add(agent_id.strip())
        agent_name = getattr(agent, "name", None)
        if isinstance(agent_name, str) and agent_name.strip():
            aliases.add(agent_name.strip())
        for alias in aliases:
            mapping[alias] = sender_id
    return mapping


def _handoff_agent_instructions(runtime: RuntimeContext, workflow_id: str, agent_id: str) -> str:
    acceptance_mode = acceptance_mode_for_metadata(
        runtime.registry.workflow_definitions[workflow_id].metadata
    )
    lines = [
        "You are participating in a decentralized specialist handoff workflow.",
        "If another specialist is better placed to continue, use a handoff tool instead of trying to do everything yourself.",
        "If you can finish your part cleanly, respond directly and stop.",
    ]
    if is_decision_ready_acceptance_mode(acceptance_mode):
        lines.extend(
            [
                "This is a discussion workflow, not an implementation workflow.",
                "Aim for a concrete decision-ready recommendation, not code changes.",
            ]
        )
    if agent_id == "reviewer":
        lines.append("When the discussion is ready to conclude, return the final recommendation plainly without handing off.")
    return "\n".join(lines)


def _handoff_start_agent(
    runtime: RuntimeContext,
    workflow_id: str,
    participants: Sequence[str],
) -> str:
    configured = runtime.registry.workflow_definitions[workflow_id].metadata.get("start_agent")
    if isinstance(configured, str) and configured in participants:
        return configured
    if not participants:
        raise ValueError(f"handoff workflow '{workflow_id}' has no participants")
    return participants[0]


def _handoff_finalizers(runtime: RuntimeContext, workflow_id: str) -> tuple[str, ...]:
    configured = runtime.registry.workflow_definitions[workflow_id].metadata.get("finalizers")
    if not isinstance(configured, list):
        return ("reviewer",)
    finalizers: list[str] = []
    for item in configured:
        if isinstance(item, str) and item.strip():
            finalizers.append(item.strip())
    return tuple(finalizers) or ("reviewer",)


def _handoff_autonomous_agents(
    runtime: RuntimeContext,
    workflow_id: str,
    participants: Sequence[str],
) -> tuple[str, ...]:
    configured = runtime.registry.workflow_definitions[workflow_id].metadata.get("autonomous_agents")
    if not isinstance(configured, list):
        return tuple(participants)
    enabled: list[str] = []
    for item in configured:
        if isinstance(item, str) and item in participants:
            enabled.append(item)
    return tuple(enabled) or tuple(participants)


def _handoff_autonomous_turn_limit(runtime: RuntimeContext, workflow_id: str) -> int:
    value = runtime.registry.workflow_definitions[workflow_id].metadata.get("autonomous_turn_limit")
    if type(value) is int and value > 0:
        return value
    return 2


def _handoff_topology(runtime: RuntimeContext, workflow_id: str) -> dict[str, tuple[str, ...]]:
    configured = runtime.registry.workflow_definitions[workflow_id].metadata.get("handoffs")
    if not isinstance(configured, dict):
        return {}
    topology: dict[str, tuple[str, ...]] = {}
    for source, raw_targets in configured.items():
        if not isinstance(source, str):
            continue
        if isinstance(raw_targets, str):
            topology[source] = (raw_targets,)
            continue
        if not isinstance(raw_targets, list):
            continue
        targets = tuple(item for item in raw_targets if isinstance(item, str) and item)
        if targets:
            topology[source] = targets
    return topology


def _apply_handoff_topology(
    builder: HandoffBuilder,
    runtime: RuntimeContext,
    workflow_id: str,
    agents: dict[str, Agent],
    participants: Sequence[str],
) -> None:
    topology = _handoff_topology(runtime, workflow_id)
    if not topology:
        return
    for source_id in participants:
        targets = topology.get(source_id)
        if not targets:
            continue
        builder.add_handoff(agents[source_id], [agents[target_id] for target_id in targets if target_id in agents])


def _build_handoff_termination_condition(runtime: RuntimeContext, workflow_id: str):
    finalizers = set(_handoff_finalizers(runtime, workflow_id))
    max_rounds = _workflow_max_rounds(runtime, workflow_id)

    def _termination(messages: list[Message]) -> bool:
        assistant_messages = [message for message in messages if message.role == "assistant"]
        if not assistant_messages:
            return False
        last = assistant_messages[-1]
        author_name = last.author_name if isinstance(last.author_name, str) else ""
        if author_name in finalizers and len(assistant_messages) >= 2:
            return True
        return max_rounds is not None and len(assistant_messages) >= max_rounds

    return _termination


async def _run_streamed_magentic_workflow(
    *,
    runtime: RuntimeContext,
    workflow,
    thread: ThreadRecord,
    task_id: str | None,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
    goal: str,
):
    stream = workflow.run(goal, stream=True)
    async for event in stream:
        if event.type != "magentic_orchestrator":
            continue
        _record_magentic_event(
            runtime=runtime,
            event=event,
            thread=thread,
            task_id=task_id,
            cursor=cursor,
            tracker=tracker,
        )
    return await stream.get_final_response()


async def _run_streamed_handoff_workflow(
    *,
    runtime: RuntimeContext,
    workflow,
    thread: ThreadRecord,
    executor_senders: dict[str, str],
    task_id: str | None,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
    goal: str,
):
    projector = _WorkflowThreadProjector(
        runtime=runtime,
        thread=thread,
        cursor=cursor,
        tracker=tracker,
    )
    participant_set = set(executor_senders)
    stream = workflow.run(
        Message(role="user", text=goal, author_name="workflow"),
        stream=True,
    )
    async for event in stream:
        if event.type == "handoff_sent":
            data = getattr(event, "data", None)
            source = getattr(data, "source", None)
            target = getattr(data, "target", None)
            if isinstance(source, str) and isinstance(target, str):
                runtime.append_event(
                    kind="workflow_handoff",
                    summary=f"{source} handed off to {target}",
                    created_at=cursor.next(),
                    thread_id=thread.id,
                    task_id=task_id,
                )
            continue
        if event.type == "output" and event.executor_id in participant_set:
            delta_text = _workflow_event_text_delta(getattr(event, "data", None))
            if delta_text:
                projector.append_delta(sender=executor_senders[event.executor_id], delta=delta_text)
            continue
        if event.type == "executor_completed" and event.executor_id in participant_set:
            projector.complete(executor_senders[event.executor_id])
            continue
        if event.type == "executor_failed" and event.executor_id in participant_set:
            details = getattr(event, "details", None)
            message = getattr(details, "message", None)
            projector.fail(
                executor_senders[event.executor_id],
                error=message if isinstance(message, str) else "workflow executor failed",
            )
            continue
    result = await stream.get_final_response()
    projector.flush()
    if not tracker.thread_outputs.get(thread.id):
        _append_output_messages_to_thread(
            runtime=runtime,
            thread=thread,
            outputs=result.get_outputs(),
            cursor=cursor,
            tracker=tracker,
            assistant_only=True,
        )
    return result


class _WorkflowThreadProjector:
    def __init__(
        self,
        *,
        runtime: RuntimeContext,
        thread: ThreadRecord,
        cursor: _TimestampCursor,
        tracker: _ExecutionTracker,
    ) -> None:
        self.runtime = runtime
        self.thread = thread
        self.cursor = cursor
        self.tracker = tracker
        self._drafts: dict[str, _ProjectedThreadDraft] = {}

    def append_delta(self, *, sender: str, delta: str) -> None:
        draft = self._drafts.get(sender)
        if draft is None:
            draft = _ProjectedThreadDraft(
                draft_id=f"draft-{uuid4().hex}",
                sender=sender,
            )
            self._drafts[sender] = draft
            self.runtime.live_state.start_draft(
                draft_id=draft.draft_id,
                thread_id=self.thread.id,
                sender=sender,
                kind="chat",
                created_at=self.cursor.next(),
            )
        draft.body += delta
        self.runtime.live_state.append_delta(
            draft_id=draft.draft_id,
            delta=delta,
            created_at=self.cursor.next(),
        )

    def complete(self, sender: str) -> None:
        draft = self._drafts.pop(sender, None)
        if draft is None:
            return
        final_text = draft.body.strip()
        if not final_text:
            self.runtime.live_state.fail_draft(
                draft_id=draft.draft_id,
                error="empty response",
                created_at=self.cursor.next(),
            )
            return
        message = self.runtime.append_message_to_thread(
            thread_id=self.thread.id,
            message_id=f"message-{uuid4().hex}",
            sender=sender,
            kind="chat",
            body=final_text,
            created_at=self.cursor.next(),
        )
        self.runtime.live_state.complete_draft(
            draft_id=draft.draft_id,
            message_id=message.id,
            created_at=self.cursor.next(),
        )
        _append_thread_output(
            tracker=self.tracker,
            thread_id=self.thread.id,
            speaker=sender,
            body=final_text,
        )
        self.tracker.last_thread_id = self.thread.id

    def fail(self, sender: str, *, error: str) -> None:
        draft = self._drafts.pop(sender, None)
        if draft is None:
            return
        self.runtime.live_state.fail_draft(
            draft_id=draft.draft_id,
            error=error,
            created_at=self.cursor.next(),
        )
        self.tracker.last_thread_id = self.thread.id

    def flush(self) -> None:
        for sender in tuple(self._drafts):
            self.complete(sender)


def _workflow_event_text_delta(data: object) -> str:
    if isinstance(data, Message):
        return _message_text(data)
    text = getattr(data, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    messages = getattr(data, "messages", None)
    if not isinstance(messages, list) or not messages:
        return ""
    last_message = messages[-1]
    if isinstance(last_message, Message):
        return _message_text(last_message)
    return ""


def _record_magentic_event(
    *,
    runtime: RuntimeContext,
    event,
    thread: ThreadRecord,
    task_id: str | None,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker | None,
) -> None:
    if event.type != "magentic_orchestrator":
        return
    data = getattr(event, "data", None)
    event_type = str(getattr(data, "event_type", "")).split(".")[-1].lower()
    content = getattr(data, "content", None)
    if not isinstance(content, Message):
        return
    sender = content.author_name if isinstance(content.author_name, str) and content.author_name else "magentic_manager"
    body = _message_text(content)
    if not body:
        return
    runtime.append_message_to_thread(
        thread_id=thread.id,
        message_id=f"message-{uuid4().hex}",
        sender=sender,
        kind="status_update",
        body=body,
        created_at=cursor.next(),
    )
    runtime.append_event(
        kind=f"magentic_{event_type or 'update'}",
        summary=f"Magentic manager updated {thread.summary or thread.id}",
        created_at=cursor.next(),
        thread_id=thread.id,
        task_id=task_id,
    )
    if tracker is not None:
        _append_thread_output(
            tracker=tracker,
            thread_id=thread.id,
            speaker=sender,
            body=body,
        )
        tracker.last_thread_id = thread.id


def _append_output_messages_to_thread(
    *,
    runtime: RuntimeContext,
    thread: ThreadRecord,
    outputs: list[object],
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
    assistant_only: bool,
) -> None:
    messages = _collect_output_messages(outputs)
    for message in messages:
        if assistant_only and message.role != "assistant":
            continue
        body = _message_text(message)
        if not body:
            continue
        sender = message.author_name if isinstance(message.author_name, str) and message.author_name else (
            thread.assigned_agent_id or message.role or "workflow"
        )
        runtime.append_message_to_thread(
            thread_id=thread.id,
            message_id=f"message-{uuid4().hex}",
            sender=sender,
            kind="chat",
            body=body,
            created_at=cursor.next(),
        )
        _append_thread_output(
            tracker=tracker,
            thread_id=thread.id,
            speaker=sender,
            body=body,
        )
        tracker.last_thread_id = thread.id


def _collect_output_messages(outputs: list[object]) -> list[Message]:
    messages: list[Message] = []
    for output in outputs:
        if isinstance(output, AgentResponse):
            messages.extend(message for message in output.messages if isinstance(message, Message))
            continue
        if isinstance(output, AgentResponseUpdate):
            messages.append(_message_from_response_update(output))
            continue
        if isinstance(output, Message):
            messages.append(output)
            continue
        if not isinstance(output, list):
            continue
        for item in output:
            if isinstance(item, Message):
                messages.append(item)
                continue
            if isinstance(item, AgentResponseUpdate):
                messages.append(_message_from_response_update(item))
    return messages


def _message_from_response_update(update: AgentResponseUpdate) -> Message:
    text_fragments: list[str] = []
    normalized_contents: list[object] = []
    for content in getattr(update, "contents", []) or []:
        if isinstance(content, str):
            text_fragments.append(content)
            continue
        content_text = getattr(content, "text", None)
        if isinstance(content_text, str) and content_text:
            text_fragments.append(content_text)
        normalized_contents.append(content)
    return Message(
        role=update.role or "assistant",
        text="".join(text_fragments).strip(),
        author_name=update.author_name or update.agent_id,
        contents=normalized_contents,
    )


def _request_info_lines(events: Sequence[object]) -> list[str]:
    details: list[str] = []
    for index, event in enumerate(events, start=1):
        question = getattr(event, "question", None)
        request_id = getattr(event, "request_id", None)
        if isinstance(question, str) and question.strip():
            label = f"{index}. {question.strip()}"
            if isinstance(request_id, str) and request_id.strip():
                label = f"{label} ({request_id.strip()})"
            details.append(label)
            continue
        if isinstance(request_id, str) and request_id.strip():
            details.append(f"{index}. Additional information requested ({request_id.strip()})")
        else:
            details.append(f"{index}. Additional information requested")
    if not details:
        return ["The workroom requested more information before it could continue."]
    return [
        "The workroom requested more information before it could continue.",
        "",
        *details,
    ]


def _message_text(message: Message) -> str:
    text = message.text.strip() if isinstance(message.text, str) else ""
    if text:
        return text
    contents = getattr(message, "contents", None)
    if not isinstance(contents, list):
        return ""
    lines: list[str] = []
    for content in contents:
        content_text = getattr(content, "text", None)
        if isinstance(content_text, str) and content_text.strip():
            lines.append(content_text.strip())
    return "\n\n".join(lines).strip()



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


def _max_clarification_cycles(runtime: RuntimeContext, workflow_id: str) -> int:
    return max(0, _workflow_int_metadata(runtime, workflow_id, "max_clarification_cycles", 2))


def _repair_step_groups(runtime: RuntimeContext, workflow_id: str) -> tuple[tuple[str, ...], ...]:
    groups = _workflow_step_group_metadata(runtime, workflow_id, "repair_step_groups")
    if groups:
        return groups
    return (("fixer",), ("reviewer",))


def _replan_step_groups(runtime: RuntimeContext, workflow_id: str) -> tuple[tuple[str, ...], ...]:
    return _workflow_step_group_metadata(runtime, workflow_id, "replan_step_groups")


def _is_valid_followup_step_groups(
    runtime: RuntimeContext,
    step_groups: tuple[tuple[str, ...], ...],
) -> bool:
    if not step_groups:
        return False
    known_agents = runtime.registry.agent_definitions
    for group in step_groups:
        if not group:
            return False
        for agent_id in group:
            if agent_id not in known_agents:
                return False
    return True


async def _decide_followup_action(
    *,
    runtime: RuntimeContext,
    workflow_run: WorkflowRunRecord,
    run_view: WorkflowRunView,
    goal: str,
    review_thread: ThreadRecord,
    cursor: _TimestampCursor,
    tracker: _ExecutionTracker,
) -> WorkflowFollowupPlan | None:
    response = await runtime.generate_agent_text_without_tools(
        agent_id="orchestrator",
        body=_workflow_followup_decision_prompt(
            runtime=runtime,
            workflow_id=workflow_run.workflow_id,
            goal=goal,
            tracker=tracker,
            run_view=run_view,
        ),
        created_at=cursor.next(),
        thread_id=review_thread.id,
        extra_instructions=_workflow_followup_decision_instructions(),
    )
    if not response:
        return None
    try:
        decision = _parse_followup_decision(response)
    except ValueError:
        return None

    if decision.action == "stop":
        return None

    if decision.action == "clarify":
        if tracker.clarification_cycles >= _max_clarification_cycles(runtime, workflow_run.workflow_id):
            return None
        if decision.request is None:
            return None
        step_groups = decision.step_groups
        if step_groups:
            if not _is_valid_followup_step_groups(runtime, step_groups):
                return None
        else:
            if decision.agent_id is None or decision.agent_id not in runtime.registry.agent_definitions:
                return None
            step_groups = ((decision.agent_id,),)
        tracker.clarification_cycles += 1
        primary_agent = step_groups[0][0]
        summary = decision.summary or f"Requesting clarification from {primary_agent}"
        return WorkflowFollowupPlan(
            cycle_kind="clarify",
            step_groups=step_groups,
            event_kind="workflow_clarification_requested",
            event_summary=summary,
            payload=_render_followup_payload(
                runtime=runtime,
                workflow_id=workflow_run.workflow_id,
                goal=goal,
                tracker=tracker,
                run_view=run_view,
                cycle_kind="clarify",
                custom_request=decision.request,
            ),
            tool_mode=decision.tool_mode,
        )

    if decision.action == "repair":
        repair_groups = decision.step_groups or _repair_step_groups(runtime, workflow_run.workflow_id)
        if decision.step_groups and not _is_valid_followup_step_groups(runtime, decision.step_groups):
            return None
        if not repair_groups or tracker.repair_cycles >= _max_repair_cycles(runtime, workflow_run.workflow_id):
            return None
        tracker.repair_cycles += 1
        return WorkflowFollowupPlan(
            cycle_kind="repair",
            step_groups=repair_groups,
            event_kind="workflow_auto_repair_started",
            event_summary=decision.summary or f"Starting automatic fix cycle {tracker.repair_cycles} for {workflow_run.workflow_id}",
            payload=_render_followup_payload(
                runtime=runtime,
                workflow_id=workflow_run.workflow_id,
                goal=goal,
                tracker=tracker,
                run_view=run_view,
                cycle_kind="repair",
                custom_request=decision.request,
            ),
            tool_mode=decision.tool_mode,
        )

    if decision.action == "replan":
        replan_groups = decision.step_groups or _replan_step_groups(runtime, workflow_run.workflow_id)
        if decision.step_groups and not _is_valid_followup_step_groups(runtime, decision.step_groups):
            return None
        if not replan_groups or tracker.replan_cycles >= _max_replan_cycles(runtime, workflow_run.workflow_id):
            return None
        tracker.replan_cycles += 1
        return WorkflowFollowupPlan(
            cycle_kind="replan",
            step_groups=replan_groups,
            event_kind="workflow_auto_replan_started",
            event_summary=decision.summary or tracker.review_replan_summary or f"Escalating {workflow_run.workflow_id} to replanning",
            payload=_render_followup_payload(
                runtime=runtime,
                workflow_id=workflow_run.workflow_id,
                goal=goal,
                tracker=tracker,
                run_view=run_view,
                cycle_kind="replan",
                custom_request=decision.request,
            ),
            tool_mode=decision.tool_mode,
        )

    return None


def _render_followup_payload(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    goal: str,
    tracker: _ExecutionTracker,
    run_view: WorkflowRunView | None,
    cycle_kind: str,
    custom_request: str | None = None,
) -> str:
    changed_files = runtime._workflow_changed_files(run_view.workflow_run.id) if run_view is not None else []
    evidence_gap_lines = _workflow_evidence_gap_lines(run_view=run_view, tracker=tracker)
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
    if evidence_gap_lines:
        lines.extend(["", "Recorded evidence gaps:"])
        lines.extend(evidence_gap_lines)
    step_outputs = _recent_step_outputs(run_view=run_view, tracker=tracker, limit=3) if run_view is not None else []
    if step_outputs:
        lines.extend(["", "Recent workflow evidence:"])
        lines.extend(step_outputs)
    if cycle_kind == "clarify":
        if custom_request:
            lines.extend(["", "Clarification needed:", custom_request])
        lines.extend(
            [
                "",
                "Answer the clarification directly from the current state. Provide concrete evidence if you have it. "
                "Do not redo unrelated work.",
            ]
        )
    elif cycle_kind == "replan":
        if custom_request:
            lines.extend(["", "Specific follow-up request:", custom_request])
        lines.extend(
            [
                "",
                "Revise the plan explicitly before continuing. Update the approach instead of making a narrow patch.",
            ]
        )
    else:
        if custom_request:
            lines.extend(["", "Specific follow-up request:", custom_request])
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
        blocked_summary=tracker.blocked_summary,
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
        "Step contract:",
        "Complete only the work for your assigned role in this step.",
        "Do not claim later workflow steps are already done unless you actually performed them with tools available to you.",
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
    workflow_run: WorkflowRunRecord,
    workflow_id: str,
    goal: str,
    payload: str | list[str],
    workspace_files: list[str],
    tracker: _ExecutionTracker,
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
    evidence_lines = _workflow_review_evidence_lines(runtime, workflow_run.id)
    if evidence_lines:
        lines.extend(
            [
                "Recorded execution evidence (authoritative system facts):",
                *evidence_lines,
                "",
                "When agent narrative conflicts with recorded execution evidence, trust the recorded evidence.",
                "",
            ]
        )
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
    evidence_gap_lines = _workflow_evidence_gap_lines(
        run_view=runtime.describe_workflow_run(workflow_run.id),
        tracker=tracker,
    )
    if evidence_gap_lines:
        lines.extend(["Recorded evidence gaps from workflow execution:"])
        lines.extend(evidence_gap_lines)
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


def _workflow_review_evidence_lines(runtime: RuntimeContext, workflow_run_id: str) -> list[str]:
    lines: list[str] = []

    file_changes: list[str] = []
    for tool_call in runtime.list_tool_calls_for_workflow_run(workflow_run_id):
        if tool_call.status != "completed" or tool_call.tool_name not in {"write_file", "patch_file"}:
            continue
        try:
            request = json.loads(runtime.read_tool_call_request(tool_call.id))
        except json.JSONDecodeError:
            request = {}
        path = request.get("path")
        if isinstance(path, str) and path:
            file_changes.append(f"- {tool_call.tool_name} by {tool_call.agent_id or 'unknown'}: {path}")

    if file_changes:
        lines.append("Recorded file changes:")
        lines.extend(file_changes[:8])
        if len(file_changes) > 8:
            lines.append(f"- ... and {len(file_changes) - 8} more")
        lines.append("")

    command_evidence: list[str] = []
    for command_run in runtime.list_command_runs_for_workflow_run(workflow_run_id):
        if command_run.status != "completed":
            continue
        output = runtime.command_store.read_command_output(command_run).strip()
        output_excerpt = _truncate_text(output, 200) if output else "(no output)"
        command_evidence.append(
            f"- {command_run.command} -> exit {command_run.exit_code}; output: {output_excerpt}"
        )

    if command_evidence:
        lines.append("Recorded command runs:")
        lines.extend(command_evidence[:8])
        if len(command_evidence) > 8:
            lines.append(f"- ... and {len(command_evidence) - 8} more")
        lines.append("")

    return lines


def _workflow_evidence_gap_lines(
    *,
    run_view: WorkflowRunView | None,
    tracker: _ExecutionTracker,
) -> list[str]:
    if not tracker.evidence_gaps:
        return []
    labels: dict[str, str] = {}
    if run_view is not None:
        for step in run_view.steps:
            for thread in step.threads:
                labels[thread.id] = thread.assigned_agent_id or thread.summary or thread.id
    lines: list[str] = []
    for thread_id, summary in tracker.evidence_gaps.items():
        label = labels.get(thread_id, thread_id)
        lines.append(f"- {label}: {summary}")
    return lines


def _step_role_rules(agent_id: str) -> str:
    if agent_id == "architect":
        return (
            "Produce a concrete handoff for implementation: proposed files, responsibilities, CLI behavior, "
            "and any key edge cases. For runnable deliverables, name one obvious entrypoint and one obvious way to run it. "
            "Do not stop at saying you will design it, and do not claim the implementation already exists."
        )
    if agent_id == "coder":
        return (
            "Produce actual repo changes in this step. Use file tools to create or modify the implementation. "
            "A prose-only answer is not sufficient. Keep the solution minimal, avoid extra files unless required, "
            "and run at most two focused verification commands. For runnable deliverables, prefer one obvious entrypoint "
            "and include one exact working invocation in your handoff."
        )
    if agent_id == "tester":
        return (
            "Verify the actual implementation with focused commands. Report the concrete commands and outcomes. "
            "If something is missing or broken, say so plainly. Run at most two focused verification commands. "
            "For runnable deliverables, confirm that at least one direct non-interactive invocation succeeds."
        )
    if agent_id == "reviewer":
        return (
            "Review the actual workspace state and verification evidence. Approve only if the requested deliverable exists "
            "and the essential behavior is covered. For runnable deliverables, require concrete black-box command evidence. "
            "Run at most two focused checks."
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
    evidence_gap_lines = _workflow_evidence_gap_lines(run_view=run_view, tracker=tracker)
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
    if evidence_gap_lines:
        lines.extend(["", "## Evidence Gaps Observed", *evidence_gap_lines])
    if tracker.clarification_cycles:
        lines.extend(["", f"- Clarification cycles: {tracker.clarification_cycles}"])
    if tracker.repair_cycles:
        lines.extend(["", f"- Automatic repair cycles: {tracker.repair_cycles}"])
    if tracker.replan_cycles:
        lines.extend(["", f"- Automatic replanning cycles: {tracker.replan_cycles}"])
    return "\n".join(lines).strip()
def _required_tool_names(agent_id: str) -> tuple[str, ...]:
    requirements = {
        "coder": ("write_file", "patch_file"),
        "fixer": ("write_file", "patch_file", "run_command"),
        "tester": ("run_command",),
        "reviewer": ("list_files", "read_file", "search_files", "run_command"),
    }
    return requirements.get(agent_id, ())


def _required_tool_names_for_workflow(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    agent_id: str,
) -> tuple[str, ...]:
    metadata = runtime.registry.workflow_definitions[workflow_id].metadata
    configured = metadata.get("tool_evidence")
    if isinstance(configured, dict):
        agent_tools = configured.get(agent_id)
        if isinstance(agent_tools, list):
            validated = tuple(
                item for item in agent_tools if isinstance(item, str) and item.strip()
            )
            if validated:
                return validated
            return ()
    return _required_tool_names(agent_id)


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


def _render_tool_requirement_retry(*, agent_id: str, required_tools: tuple[str, ...]) -> str:
    if agent_id == "coder" and required_tools == ("write_file", "patch_file"):
        return (
            "Your last turn did not record any real file-edit tool call. "
            "Use `write_file` or `patch_file` now and then summarize the actual implementation."
        )
    if agent_id == "fixer" and required_tools == ("write_file", "patch_file", "run_command"):
        return (
            "Your last turn did not record any real repair action. "
            "Use `write_file`, `patch_file`, or `run_command` now to apply and verify the fix."
        )
    if agent_id == "tester" and required_tools == ("run_command",):
        return (
            "Your last turn did not record any real verification command. "
            "Use `run_command` now against the actual implementation and report the concrete result."
        )
    if agent_id == "reviewer" and required_tools == ("list_files", "read_file", "search_files", "run_command"):
        return (
            "Your last turn did not record any real inspection or verification tool use. "
            "Use `list_files`, `read_file`, `search_files`, or `run_command` now, then return the review."
        )
    tools = ", ".join(f"`{tool}`" for tool in required_tools)
    return (
        "Your last turn did not record the required workspace-tool evidence. "
        f"Use one of {tools} now and then return the concrete result."
    )


def _orchestrator_review_instructions() -> str:
    return "\n".join(
        [
            "You are a dedicated workflow acceptance reviewer.",
            "Judge whether the work satisfies the stated goal and whether it is a minimal working delivery.",
            "For runnable deliverables, reject the work if the evidence does not show at least one concrete successful command run.",
            "Do not suggest optional polish.",
            "Do not delegate, do not plan, and do not call tools.",
            "Output JSON only with this shape:",
            '{"accepted": true, "summary": "one concise review summary", "findings": ["specific issue"], "requires_replan": false, "replan_summary": ""}',
        ]
    )


def _workflow_followup_decision_instructions() -> str:
    return "\n".join(
        [
            "You are deciding the next workflow move after either a rejected orchestrator review or a blocked worker step.",
            "Prefer the smallest action that can resolve the problem.",
            "If a worker step is blocked, prefer a clarification or narrow repair before replanning.",
            "Use `clarify` when the work may already be acceptable but evidence or explanation is missing.",
            "Use `repair` when the underlying work needs a focused fix.",
            "Use `replan` when the current approach is wrong enough that another narrow fix is wasteful.",
            "Use `stop` when no safe automatic follow-up is justified.",
            "For `clarify`, provide a specific `request` and either `agent_id` or `step_groups`.",
            "You may provide `step_groups` as a list of agent-id lists when the next follow-up should use custom staffing instead of the workflow defaults.",
            "If you omit `step_groups`, the runtime will use the workflow defaults for `repair` and `replan`.",
            "Set `tool_mode` to `none` only when the clarification should be explanation-only and should not require fresh tool evidence.",
            "Return JSON only with this shape:",
            '{"action":"clarify","summary":"one concise sentence","agent_id":"tester","request":"Run one concrete command and report the result.","step_groups":[["tester"]],"tool_mode":"default"}',
        ]
    )


def _workflow_followup_decision_prompt(
    *,
    runtime: RuntimeContext,
    workflow_id: str,
    goal: str,
    tracker: _ExecutionTracker,
    run_view: WorkflowRunView | None,
) -> str:
    changed_files = runtime._workflow_changed_files(run_view.workflow_run.id) if run_view is not None else []
    evidence_gap_lines = _workflow_evidence_gap_lines(run_view=run_view, tracker=tracker)
    trigger = "blocked worker step" if tracker.blocked_step_index is not None else "rejected orchestrator review"
    lines = [
        f"Workflow: {workflow_id}",
        f"Current trigger: {trigger}",
        "",
        "Goal:",
        _truncate_text(goal, 800),
    ]
    if tracker.blocked_step_index is not None:
        lines.extend(
            [
                "",
                "Blocked step:",
                _blocked_agent_label(run_view=run_view, tracker=tracker),
                f"Blocked reason: {tracker.blocked_reason or '(unknown)'}",
                "",
                "Blocked summary:",
                tracker.blocked_summary or "(no blocked summary)",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Latest orchestrator review:",
                tracker.review_summary or "(no review summary)",
            ]
        )
    if tracker.review_findings:
        lines.extend(["", "Current findings:"])
        lines.extend(f"- {finding}" for finding in tracker.review_findings)
    if evidence_gap_lines:
        lines.extend(["", "Recorded evidence gaps:"])
        lines.extend(evidence_gap_lines)
    if changed_files:
        lines.extend(["", "Files touched so far:"])
        lines.extend(f"- {path}" for path in changed_files)
    step_outputs = _recent_step_outputs(run_view=run_view, tracker=tracker, limit=4) if run_view is not None else []
    if step_outputs:
        lines.extend(["", "Recent workflow evidence:"])
        lines.extend(step_outputs)
    lines.extend(
        [
            "",
            "Cycles used:",
            f"- clarification: {tracker.clarification_cycles}",
            f"- repair: {tracker.repair_cycles}",
            f"- replan: {tracker.replan_cycles}",
            "",
            "Decide the next move.",
        ]
    )
    return "\n".join(lines).strip()


def _truncate_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ..."


def _blocked_agent_label(
    *,
    run_view: WorkflowRunView | None,
    tracker: _ExecutionTracker,
) -> str:
    blocked_agent = _blocked_agent_id(run_view=run_view, tracker=tracker)
    if blocked_agent is not None:
        return blocked_agent
    return tracker.blocked_thread_id or "(unknown)"


def _blocked_agent_id(
    *,
    run_view: WorkflowRunView | None,
    tracker: _ExecutionTracker,
) -> str | None:
    if run_view is not None and tracker.blocked_thread_id is not None:
        for step in run_view.steps:
            for thread in step.threads:
                if thread.id == tracker.blocked_thread_id:
                    return thread.assigned_agent_id
    return None


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


def _parse_followup_decision(raw: str) -> WorkflowFollowupDecision:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise ValueError("no JSON object found in followup decision")
        text = match.group(0)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("followup decision must be an object")
    action = str(payload.get("action", "")).strip().lower()
    if action not in {"clarify", "repair", "replan", "stop"}:
        raise ValueError("invalid followup action")
    summary = str(payload.get("summary", "")).strip()
    agent_id_raw = payload.get("agent_id")
    request_raw = payload.get("request")
    step_groups_raw = payload.get("step_groups", [])
    tool_mode = str(payload.get("tool_mode", "default")).strip().lower() or "default"
    if tool_mode not in {"default", "none"}:
        tool_mode = "default"
    step_groups: list[tuple[str, ...]] = []
    if isinstance(step_groups_raw, list):
        for raw_group in step_groups_raw:
            if not isinstance(raw_group, list):
                raise ValueError("followup step_groups must be a list of lists")
            cleaned_group: list[str] = []
            for item in raw_group:
                if not isinstance(item, str):
                    raise ValueError("followup step_groups entries must be strings")
                agent_id = item.strip()
                if not agent_id:
                    raise ValueError("followup step_groups entries must be non-empty")
                cleaned_group.append(agent_id)
            if not cleaned_group:
                raise ValueError("followup step_groups may not contain empty groups")
            step_groups.append(tuple(cleaned_group))
    return WorkflowFollowupDecision(
        action=action,
        summary=summary,
        agent_id=str(agent_id_raw).strip() if isinstance(agent_id_raw, str) and str(agent_id_raw).strip() else None,
        request=str(request_raw).strip() if isinstance(request_raw, str) and str(request_raw).strip() else None,
        step_groups=tuple(step_groups),
        tool_mode=tool_mode,
    )


def _format_review_summary(verdict: WorkflowReviewVerdict) -> str:
    prefix = "ACCEPTED" if verdict.accepted else "REJECTED"
    return f"{prefix}: {verdict.summary}"
