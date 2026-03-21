from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace

from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyWorkflowRequestExecutor:
    def __init__(self, workflow_dispatcher: ProxyWorkflowDispatcher) -> None:
        self._workflow_dispatcher = workflow_dispatcher

    async def execute_active_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan | None = None,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        workflow_progress = (
            loop_state.workflow_progress
            if loop_state is not None
            else None
        )
        if workflow_progress is None:
            state.finish_reason = "error"
            error_text = "No active playbook is available to continue."
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        active_continuation = _override_active_staffing(
            workflow_progress,
            plan=plan,
        )
        async for event in self._workflow_dispatcher.execute_workflow_continuation(
            request=request,
            continuation=active_continuation,
            pending=None,
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def execute_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._workflow_dispatcher.execute_workflow(
            request=request,
            workflow_id=plan.workflow_id,
            specialists=plan.specialists,
            specialist_counts=plan.specialist_counts,
            workflow_request=plan.playbook_request,
            goal=(
                loop_state.goal
                if loop_state is not None
                else request.latest_user_text() or ""
            ),
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def execute_workflow_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        pending: PendingContinuation,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._workflow_dispatcher.execute_workflow_continuation(
            request=request,
            continuation=continuation,
            pending=pending,
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event


def _override_active_staffing(
    continuation: ContinuationState,
    *,
    plan: ProxyTurnPlan | None,
) -> ContinuationState:
    if plan is None:
        return continuation
    workflow_specialists, workflow_specialist_counts = _apply_staffing_action(
        continuation,
        plan=plan,
    )
    workflow_request = (
        plan.playbook_request
        if plan is not None and plan.playbook_request
        else continuation.workflow_request
    )
    if (
        workflow_specialists == continuation.workflow_specialists
        and workflow_specialist_counts == continuation.workflow_specialist_counts
        and workflow_request == continuation.workflow_request
    ):
        return continuation
    return replace(
        continuation,
        workflow_specialists=workflow_specialists,
        workflow_specialist_counts=workflow_specialist_counts,
        workflow_request=workflow_request,
    )


def _apply_staffing_action(
    continuation: ContinuationState,
    *,
    plan: ProxyTurnPlan,
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    current_specialists = list(continuation.workflow_specialists)
    current_count_map = {
        agent_id: count for agent_id, count in continuation.workflow_specialist_counts
    }
    requested_specialists = tuple(plan.specialists)
    requested_count_map = {
        agent_id: count for agent_id, count in plan.specialist_counts
    }
    action = plan.staffing_action
    if action is None:
        if requested_specialists or requested_count_map:
            action = "replace"
        else:
            action = "keep"

    if action == "keep":
        return (
            continuation.workflow_specialists,
            continuation.workflow_specialist_counts,
        )

    if action == "replace":
        if not requested_specialists and not requested_count_map:
            return (
                continuation.workflow_specialists,
                continuation.workflow_specialist_counts,
            )
        replace_specialists = requested_specialists or tuple(requested_count_map)
        return replace_specialists, _normalized_counts(
            replace_specialists,
            requested_count_map,
        )

    if action == "augment":
        specialists = list(current_specialists)
        for agent_id in requested_specialists:
            if agent_id not in specialists:
                specialists.append(agent_id)
        for agent_id in requested_count_map:
            if agent_id not in specialists:
                specialists.append(agent_id)
        count_map = dict(current_count_map)
        for agent_id, count in requested_count_map.items():
            count_map[agent_id] = max(count_map.get(agent_id, 1), count)
        return tuple(specialists), _normalized_counts(tuple(specialists), count_map)

    if action == "trim":
        removed = set(requested_specialists)
        specialists = [
            agent_id for agent_id in current_specialists if agent_id not in removed
        ]
        count_map = {
            agent_id: count
            for agent_id, count in current_count_map.items()
            if agent_id not in removed
        }
        for agent_id, count in requested_count_map.items():
            if agent_id not in specialists:
                continue
            current = count_map.get(agent_id, 1)
            reduced = min(current, count)
            if reduced <= 1:
                count_map.pop(agent_id, None)
            else:
                count_map[agent_id] = reduced
        return tuple(specialists), _normalized_counts(tuple(specialists), count_map)

    return continuation.workflow_specialists, continuation.workflow_specialist_counts


def _normalized_counts(
    specialists: tuple[str, ...],
    count_map: dict[str, int],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (agent_id, count)
        for agent_id in specialists
        if (count := count_map.get(agent_id, 1)) > 1
    )
