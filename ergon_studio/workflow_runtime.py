from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

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
    artifact_id: str | None = None


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
        self.tracker.thread_outputs[self.thread.id] = reply_body
        self.tracker.last_thread_id = self.thread.id
        await ctx.send_message(reply_body)


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
        prompt = _render_review_prompt(
            workflow_id=self.workflow_run.workflow_id,
            goal=self.goal,
            payload=payload,
        )
        _, reply = await self.runtime.send_message_to_agent_thread(
            thread_id=self.review_thread.id,
            body=prompt,
            created_at=self.cursor.next(),
        )
        if reply is None:
            self.tracker.blocked_thread_id = self.review_thread.id
            self.tracker.last_thread_id = self.review_thread.id
            raise RuntimeError("orchestrator review did not return a result")

        review_summary = self.runtime.conversation_store.read_message_body(reply).rstrip("\n")
        self.tracker.review_summary = review_summary
        self.tracker.last_thread_id = self.review_thread.id
        await ctx.yield_output(review_summary)


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
    steps = list(run_view.steps)

    if not steps:
        kickoff = _KickoffExecutor()
        review_executor = _OrchestratorReviewExecutor(
            runtime=runtime,
            workflow_run=workflow_run,
            review_thread=review_thread,
            goal=goal,
            cursor=cursor,
            tracker=tracker,
        )
        workflow = WorkflowBuilder(
            name=f"workflow-{workflow_run.id}",
            description=workflow_run.workflow_id,
            start_executor=kickoff,
            output_executors=[review_executor],
        ).add_edge(kickoff, review_executor).build()
        result = await workflow.run(goal, include_status_events=True)
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

    executors_by_step: list[tuple[_ThreadExecutor, ...]] = []
    for step_index, step in enumerate(steps):
        executors_by_step.append(
            tuple(
                _ThreadExecutor(
                    runtime=runtime,
                    thread=thread,
                    workflow_id=workflow_run.workflow_id,
                    step_index=step_index,
                    total_steps=len(steps),
                    goal=goal,
                    cursor=cursor,
                    tracker=tracker,
                )
                for thread in step.threads
            )
        )

    review_executor = _OrchestratorReviewExecutor(
        runtime=runtime,
        workflow_run=workflow_run,
        review_thread=review_thread,
        goal=goal,
        cursor=cursor,
        tracker=tracker,
    )

    kickoff = _KickoffExecutor()
    builder = WorkflowBuilder(
        name=f"workflow-{workflow_run.id}",
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
        result = await builder.build().run(goal, include_status_events=True)
    except Exception:
        if tracker.blocked_step_index is None:
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
        state = "completed"
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

    if state == "completed":
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
        runtime.append_event(
            kind="workflow_completed",
            summary=f"Completed workflow {workflow_run.workflow_id}",
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
    lines = [
        f"Workflow: {workflow_id}",
        f"Step: {step_index + 1}/{total_steps}",
        f"Assigned agent: {agent_id}",
        "",
        "Goal:",
        goal,
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


def _render_review_prompt(*, workflow_id: str, goal: str, payload: str | list[str]) -> str:
    lines = [
        f"Review workflow {workflow_id} as the orchestrator.",
        "",
        "Goal:",
        goal,
        "",
    ]
    if isinstance(payload, list):
        lines.append("Final upstream outputs:")
        for index, item in enumerate(payload, start=1):
            lines.extend([f"{index}.", item, ""])
    else:
        lines.extend(["Final workflow output:", payload, ""])
    lines.extend(
        [
            "Decide whether the work satisfies the goal, the project direction, and the expected quality bar.",
            "Respond with a concise acceptance review suitable for the main user-facing chat.",
        ]
    )
    return "\n".join(lines).strip()


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
        agents = ", ".join(thread.assigned_agent_id or thread.id for thread in step.threads)
        lines.extend([f"### Step {step_index}: {agents}"])
        for thread in step.threads:
            body = tracker.thread_outputs.get(thread.id, "(no output)")
            lines.extend([f"- {thread.assigned_agent_id or thread.id}", body])
        lines.append("")
    lines.extend(["## Orchestrator Review", tracker.review_summary or "(no review summary)"])
    return "\n".join(lines).strip()
