from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import json
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from agent_framework import Agent, Executor, Message, WorkflowBuilder, WorkflowContext, handler

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


@dataclass(frozen=True)
class WorkflowReviewVerdict:
    accepted: bool
    summary: str


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
            workspace_files=self.runtime._workspace_file_list(limit=16),
        )
        self.runtime.append_message_to_thread(
            thread_id=self.review_thread.id,
            message_id=f"message-{uuid4().hex}",
            sender="workflow",
            kind="chat",
            body=prompt,
            created_at=self.cursor.next(),
        )
        reply = await self._run_structured_review(prompt)
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
        self.runtime.append_message_to_thread(
            thread_id=self.review_thread.id,
            message_id=f"message-{uuid4().hex}",
            sender="orchestrator",
            kind="review",
            body=review_summary,
            created_at=self.cursor.next(),
        )
        self.tracker.review_summary = review_summary
        self.tracker.review_accepted = verdict.accepted
        self.tracker.last_thread_id = self.review_thread.id
        await ctx.yield_output(review_summary)

    async def _run_structured_review(self, prompt: str) -> str | None:
        try:
            orchestrator = self.runtime.build_agent("orchestrator")
        except (KeyError, ValueError):
            return None
        client = getattr(orchestrator, "client", None)
        if client is None:
            return await self.runtime.generate_agent_text_without_tools(
                agent_id="orchestrator",
                body=prompt,
                created_at=self.cursor.next(),
                thread_id=self.review_thread.id,
                extra_instructions=(
                    "This is a review-only turn. Do not delegate, do not run workflows, and do not call tools. "
                    "Return JSON only with `accepted` and `summary`."
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
                session=review_agent.create_session(session_id=f"{self.review_thread.id}:review"),
            )
        except Exception as exc:
            self.runtime.append_event(
                kind="orchestrator_review_failed",
                summary=f"Review agent failed: {type(exc).__name__}: {exc}",
                created_at=self.cursor.next(),
                thread_id=self.review_thread.id,
                task_id=self.workflow_run.root_task_id,
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
        if tracker.repair_cycles >= 1 or not _supports_auto_repair(active_workflow_run.workflow_id):
            break

        tracker.repair_cycles += 1
        runtime.append_event(
            kind="workflow_auto_repair_started",
            summary=f"Starting automatic fix cycle {tracker.repair_cycles} for {active_workflow_run.workflow_id}",
            created_at=cursor.next(),
            thread_id=review_thread.id,
            task_id=active_workflow_run.root_task_id,
        )
        active_workflow_run = type(active_workflow_run)(
            id=active_workflow_run.id,
            session_id=active_workflow_run.session_id,
            workflow_id=active_workflow_run.workflow_id,
            state="repairing",
            created_at=active_workflow_run.created_at,
            updated_at=cursor.next(),
            root_task_id=active_workflow_run.root_task_id,
            current_step_index=len(active_run_view.steps),
            last_thread_id=tracker.last_thread_id,
        )
        runtime.workflow_store.update_workflow_run(active_workflow_run)
        active_workflow_run, _ = runtime.request_workflow_fix_cycle(
            workflow_run_id=active_workflow_run.id,
            created_at=cursor.next(),
        )
        refreshed_view = runtime.describe_workflow_run(active_workflow_run.id)
        if refreshed_view is None:
            tracker.failed = True
            break
        start_step_index = len(active_run_view.steps)
        active_run_view = refreshed_view
        initial_payload = tracker.review_summary or goal

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
    workflow_id: str,
    goal: str,
    payload: str | list[str],
    workspace_files: list[str],
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
    lines.extend(
        [
            "Decide whether the work satisfies the goal and represents a minimal working delivery.",
            "Do not create new requirements beyond the stated goal.",
            "Return JSON only with this shape:",
            '{"accepted": true, "summary": "one concise review summary"}',
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
        agents = ", ".join(thread.assigned_agent_id or thread.id for thread in step.threads)
        lines.extend([f"### Step {step_index}: {agents}"])
        for thread in step.threads:
            body = tracker.thread_outputs.get(thread.id, "(no output)")
            lines.extend([f"- {thread.assigned_agent_id or thread.id}", body])
        lines.append("")
    lines.extend(["## Orchestrator Review", tracker.review_summary or "(no review summary)"])
    if tracker.repair_cycles:
        lines.extend(["", f"- Automatic repair cycles: {tracker.repair_cycles}"])
    return "\n".join(lines).strip()


def _supports_auto_repair(workflow_id: str) -> bool:
    return workflow_id in {"standard-build", "single-agent-execution", "best-of-n"}


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
            '{"accepted": true, "summary": "one concise review summary"}',
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
    return WorkflowReviewVerdict(accepted=accepted, summary=summary)


def _format_review_summary(verdict: WorkflowReviewVerdict) -> str:
    prefix = "ACCEPTED" if verdict.accepted else "REJECTED"
    return f"{prefix}: {verdict.summary}"
