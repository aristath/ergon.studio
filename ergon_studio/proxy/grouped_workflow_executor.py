from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import summarize_conversation
from ergon_studio.proxy.prompts import workflow_step_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.selection_outcome import ProxySelectionOutcome
from ergon_studio.proxy.tool_passthrough import extract_tool_calls
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


@dataclass(frozen=True)
class _AgentAttemptResult:
    agent_id: str
    agent_label: str
    text: str
    response: Any


class ProxyGroupedWorkflowExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workflow_summary: Callable[..., AsyncIterator[ProxyEvent]],
        select_comparison_outcome: Callable[..., Any],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workflow_summary = emit_workflow_summary
        self._select_comparison_outcome = select_comparison_outcome

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        staffed_specialists = (
            continuation.workflow_specialists
            if continuation is not None
            else specialists
        )
        staffed_specialist_counts = (
            continuation.workflow_specialist_counts
            if continuation is not None
            else specialist_counts
        )
        step_groups = _filtered_step_groups(
            definition,
            staffed_specialists,
            staffed_specialist_counts,
        )
        start_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        start_agent_index = (
            continuation.agent_index
            if continuation and continuation.agent_index is not None
            else 0
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        last_stage_outputs = (
            list(continuation.last_stage_outputs)
            if continuation is not None
            else []
        )
        last_stage_parallel_attempts = (
            continuation.last_stage_parallel_attempts
            if continuation is not None
            else False
        )
        selection_outcome = (
            continuation.selection_outcome
            if continuation is not None
            else (
                loop_state.latest_selection_outcome
                if loop_state is not None
                else None
            )
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        for step_index in range(start_index, len(step_groups)):
            group = step_groups[step_index]
            group_start_index = start_agent_index if step_index == start_index else 0
            stage_entry_brief = current_brief
            stage_outputs: list[str] = []
            comparison_candidates = (
                tuple(last_stage_outputs) if last_stage_parallel_attempts else ()
            )
            if self._should_try_parallel_group(
                group=group,
                pending=pending,
                group_start_index=group_start_index,
            ):
                parallel_results = await self._run_parallel_group(
                    request=request,
                    definition=definition,
                    group=group,
                    goal=goal,
                    current_brief=stage_entry_brief,
                    loop_state=loop_state,
                    selection_outcome=selection_outcome,
                )
                if any(
                    extract_tool_calls(result.response)
                    for result in parallel_results
                    if result.response is not None
                ):
                    fallback_notice = (
                        "Orchestrator: parallel stage requested tool use; "
                        "rerunning this staffed group sequentially for safe "
                        "continuation.\n"
                    )
                    state.append_reasoning(fallback_notice)
                    yield ProxyReasoningDeltaEvent(fallback_notice)
                else:
                    for result in parallel_results:
                        reasoning_delta = f"{result.agent_label}: {result.text}"
                        state.append_reasoning(reasoning_delta)
                        yield ProxyReasoningDeltaEvent(reasoning_delta)
                        stage_outputs.append(reasoning_delta)
                    workflow_outputs.extend(stage_outputs)
                    current_brief = _stage_brief(
                        stage_outputs=stage_outputs,
                        fallback=stage_entry_brief,
                    )
                    last_stage_outputs = list(stage_outputs)
                    last_stage_parallel_attempts = True
                    if result_sink is not None:
                        result_sink(
                            ProxyMoveResult(
                                worklog_lines=tuple(stage_outputs),
                                current_brief=current_brief,
                                workflow_progress=self._next_workflow_progress(
                                    definition=definition,
                                    step_groups=step_groups,
                                    staffed_specialists=staffed_specialists,
                                    staffed_specialist_counts=(
                                        staffed_specialist_counts
                                    ),
                                    step_index=step_index,
                                    goal=goal,
                                    current_brief=current_brief,
                                    loop_state=loop_state,
                                    workflow_outputs=workflow_outputs,
                                    last_stage_outputs=last_stage_outputs,
                                    last_stage_parallel_attempts=(
                                        last_stage_parallel_attempts
                                    ),
                                    selection_outcome=None,
                                ),
                                selection_outcome=None,
                                selection_outcome_changed=True,
                            )
                        )
                        return
                    continue
            for agent_index in range(group_start_index, len(group)):
                agent_id = group[agent_index]
                agent_label = _agent_instance_label(group, agent_index)
                prompt = workflow_step_prompt(
                    workflow_id=definition.id,
                    agent_id=agent_id,
                    role_instance_label=(
                        agent_label if agent_label != agent_id else None
                    ),
                    role_instance_context=_agent_instance_context(group, agent_index),
                    goal=goal,
                    current_brief=(
                        stage_entry_brief
                        if _is_parallel_attempt_group(group)
                        else current_brief
                    ),
                    transcript_summary=summarize_conversation(request.messages),
                    prior_outputs=tuple(workflow_outputs),
                    comparison_candidates=comparison_candidates,
                    selection_outcome=selection_outcome,
                    comparison_mode=(
                        loop_state.current_comparison_mode
                        if loop_state is not None
                        else None
                    ),
                    comparison_criteria=(
                        loop_state.current_comparison_criteria
                        if loop_state is not None
                        else None
                    ),
                    move_rationale=(
                        loop_state.current_move_rationale
                        if loop_state is not None
                        else None
                    ),
                    success_criteria=(
                        loop_state.current_move_success_criteria
                        if loop_state is not None
                        else None
                    ),
                )
                agent_text = ""
                first = True
                response_holder: dict[str, Any] = {}
                async for delta in self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=prompt,
                    session_id=f"proxy-workflow-{definition.id}-{agent_id}-{uuid4().hex}",
                    model_id_override=request.model,
                    host_tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    pending_continuation=pending
                    if step_index == start_index and agent_index == group_start_index
                    else None,
                    final_response_sink=response_holder_sink(response_holder),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_label}: {delta}" if first else delta
                    first = False
                    state.append_reasoning(reasoning_delta)
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                response = response_holder.get("response")
                if response is not None:
                    emitted = self._emit_tool_calls(
                        response=response,
                        request=request,
                        continuation=ContinuationState(
                            mode="workflow",
                            workflow_id=definition.id,
                            workflow_specialists=staffed_specialists,
                            workflow_specialist_counts=staffed_specialist_counts,
                            last_stage_outputs=tuple(last_stage_outputs),
                            last_stage_parallel_attempts=(
                                last_stage_parallel_attempts
                            ),
                            selection_outcome=selection_outcome,
                            step_index=step_index,
                            agent_index=agent_index,
                            agent_id=agent_id,
                            goal=goal,
                            current_brief=agent_text.strip() or stage_entry_brief,
                            decision_history=(
                                loop_state.worklog if loop_state is not None else ()
                            ),
                            workflow_outputs=tuple(workflow_outputs),
                        ),
                        state=state,
                    )
                    if emitted:
                        for event in emitted:
                            yield event
                        return
                stage_outputs.append(f"{agent_label}: {agent_text.strip()}")
                workflow_outputs.append(stage_outputs[-1])
                if not _is_parallel_attempt_group(group):
                    current_brief = agent_text.strip() or current_brief
            if _is_parallel_attempt_group(group):
                next_selection_outcome = None
            elif (
                comparison_candidates
                and loop_state is not None
                and loop_state.current_comparison_mode is not None
            ):
                next_selection_outcome = await self._select_comparison_outcome(
                    workflow_id=definition.id,
                    goal=goal,
                    comparison_mode=loop_state.current_comparison_mode,
                    comparison_candidates=comparison_candidates,
                    stage_outputs=tuple(stage_outputs),
                    comparison_criteria=loop_state.current_comparison_criteria,
                    move_rationale=loop_state.current_move_rationale,
                    success_criteria=loop_state.current_move_success_criteria,
                    model_id_override=request.model,
                )
            else:
                next_selection_outcome = selection_outcome
            if _is_parallel_attempt_group(group):
                current_brief = _stage_brief(
                    stage_outputs=stage_outputs,
                    fallback=stage_entry_brief,
                )
            last_stage_outputs = list(stage_outputs)
            last_stage_parallel_attempts = _is_parallel_attempt_group(group)
            selection_outcome = next_selection_outcome
            if result_sink is not None:
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=tuple(stage_outputs),
                        current_brief=current_brief,
                        selection_outcome=selection_outcome,
                        selection_outcome_changed=True,
                        workflow_progress=self._next_workflow_progress(
                            definition=definition,
                            step_groups=step_groups,
                            staffed_specialists=staffed_specialists,
                            staffed_specialist_counts=staffed_specialist_counts,
                            step_index=step_index,
                            goal=goal,
                            current_brief=current_brief,
                            loop_state=loop_state,
                            workflow_outputs=workflow_outputs,
                            last_stage_outputs=last_stage_outputs,
                            last_stage_parallel_attempts=(
                                last_stage_parallel_attempts
                            ),
                            selection_outcome=selection_outcome,
                        ),
                    )
                )
                return
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield summary_event

    def _should_try_parallel_group(
        self,
        *,
        group: tuple[str, ...],
        pending: PendingContinuation | None,
        group_start_index: int,
    ) -> bool:
        return (
            _is_parallel_attempt_group(group)
            and pending is None
            and group_start_index == 0
        )

    async def _run_parallel_group(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        group: tuple[str, ...],
        goal: str,
        current_brief: str,
        loop_state: ProxyDecisionLoopState | None,
        selection_outcome: ProxySelectionOutcome | None,
    ) -> list[_AgentAttemptResult]:
        tasks = [
            asyncio.create_task(
                self._run_group_agent(
                    request=request,
                    definition=definition,
                    group=group,
                    agent_index=agent_index,
                    goal=goal,
                    current_brief=current_brief,
                    loop_state=loop_state,
                    selection_outcome=selection_outcome,
                )
            )
            for agent_index in range(len(group))
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_group_agent(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        group: tuple[str, ...],
        agent_index: int,
        goal: str,
        current_brief: str,
        loop_state: ProxyDecisionLoopState | None,
        selection_outcome: ProxySelectionOutcome | None,
    ) -> _AgentAttemptResult:
        agent_id = group[agent_index]
        agent_label = _agent_instance_label(group, agent_index)
        prompt = workflow_step_prompt(
            workflow_id=definition.id,
            agent_id=agent_id,
            role_instance_label=(agent_label if agent_label != agent_id else None),
            role_instance_context=_agent_instance_context(group, agent_index),
            goal=goal,
            current_brief=current_brief,
            transcript_summary=summarize_conversation(request.messages),
            prior_outputs=(),
            comparison_candidates=(),
            selection_outcome=selection_outcome,
            comparison_mode=(
                loop_state.current_comparison_mode if loop_state is not None else None
            ),
            comparison_criteria=(
                loop_state.current_comparison_criteria
                if loop_state is not None
                else None
            ),
            move_rationale=(
                loop_state.current_move_rationale if loop_state is not None else None
            ),
            success_criteria=(
                loop_state.current_move_success_criteria
                if loop_state is not None
                else None
            ),
        )
        text = ""
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id=agent_id,
            prompt=prompt,
            session_id=f"proxy-workflow-{definition.id}-{agent_label}-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            final_response_sink=response_holder_sink(response_holder),
        ):
            text += delta
        return _AgentAttemptResult(
            agent_id=agent_id,
            agent_label=agent_label,
            text=text.strip(),
            response=response_holder.get("response"),
        )

    def _next_workflow_progress(
        self,
        *,
        definition: DefinitionDocument,
        step_groups: tuple[tuple[str, ...], ...],
        staffed_specialists: tuple[str, ...],
        staffed_specialist_counts: tuple[tuple[str, int], ...],
        step_index: int,
        goal: str,
        current_brief: str,
        loop_state: ProxyDecisionLoopState | None,
        workflow_outputs: list[str],
        last_stage_outputs: list[str],
        last_stage_parallel_attempts: bool,
        selection_outcome: ProxySelectionOutcome | None,
    ) -> ContinuationState | None:
        next_step_index = step_index + 1
        if next_step_index >= len(step_groups):
            return None
        next_group = step_groups[next_step_index]
        return ContinuationState(
            mode="workflow",
            workflow_id=definition.id,
            workflow_specialists=staffed_specialists,
            workflow_specialist_counts=staffed_specialist_counts,
            last_stage_outputs=tuple(last_stage_outputs),
            last_stage_parallel_attempts=last_stage_parallel_attempts,
            selection_outcome=selection_outcome,
            step_index=next_step_index,
            agent_index=0,
            agent_id=next_group[0],
            goal=goal,
            current_brief=current_brief,
            decision_history=(
                loop_state.worklog if loop_state is not None else ()
            ),
            workflow_outputs=tuple(workflow_outputs),
        )


def _filtered_step_groups(
    definition: DefinitionDocument,
    specialists: tuple[str, ...],
    specialist_counts: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, ...], ...]:
    step_groups = workflow_step_groups_for_definition(definition)
    if not specialists and not specialist_counts:
        return step_groups
    count_map = dict(specialist_counts)
    allowed = set(specialists) if specialists else None
    if allowed is not None:
        allowed.update(count_map)
    filtered: list[tuple[str, ...]] = []
    for group in step_groups:
        filtered_group = _expand_group_staffing(
            group=group,
            allowed=allowed,
            count_map=count_map,
        )
        if filtered_group:
            filtered.append(filtered_group)
    return tuple(filtered)


def _expand_group_staffing(
    *,
    group: tuple[str, ...],
    allowed: set[str] | None,
    count_map: dict[str, int],
) -> tuple[str, ...]:
    expanded: list[str] = []
    seen_roles: set[str] = set()
    for agent_id in group:
        if agent_id in seen_roles:
            continue
        seen_roles.add(agent_id)
        if allowed is not None and agent_id not in allowed:
            continue
        default_count = sum(1 for candidate in group if candidate == agent_id)
        desired_count = count_map.get(agent_id, default_count)
        expanded.extend(agent_id for _ in range(desired_count))
    return tuple(expanded)


def _is_parallel_attempt_group(group: tuple[str, ...]) -> bool:
    return len(group) > 1 and len(set(group)) == 1


def _stage_brief(*, stage_outputs: list[str], fallback: str) -> str:
    if not stage_outputs:
        return fallback
    if len(stage_outputs) == 1:
        return stage_outputs[0].split(": ", 1)[-1]
    return "\n".join(stage_outputs)


def _agent_instance_label(group: tuple[str, ...], agent_index: int) -> str:
    agent_id = group[agent_index]
    total_instances = sum(1 for candidate in group if candidate == agent_id)
    if total_instances <= 1:
        return agent_id
    current_instance = sum(
        1 for candidate in group[: agent_index + 1] if candidate == agent_id
    )
    return f"{agent_id}[{current_instance}]"


def _agent_instance_context(group: tuple[str, ...], agent_index: int) -> str | None:
    agent_id = group[agent_index]
    total_instances = sum(1 for candidate in group if candidate == agent_id)
    if total_instances <= 1:
        return None
    current_instance = sum(
        1 for candidate in group[: agent_index + 1] if candidate == agent_id
    )
    return (
        f"You are instance {current_instance} of {total_instances} for the "
        f"{agent_id} role in this staffed playbook stage. Add an independently "
        "useful attempt instead of repeating the other instances."
    )
