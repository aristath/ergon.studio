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
            goal=plan.goal or request.latest_user_text() or "",
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
    workflow_specialists = (
        plan.specialists
        if plan.specialists
        else continuation.workflow_specialists
    )
    workflow_specialist_counts = (
        plan.specialist_counts
        if plan.specialist_counts
        else tuple(
            (agent_id, count)
            for agent_id, count in continuation.workflow_specialist_counts
            if not workflow_specialists or agent_id in workflow_specialists
        )
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
