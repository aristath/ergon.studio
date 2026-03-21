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


class ProxyWorkroomRequestExecutor:
    def __init__(self, workflow_dispatcher: ProxyWorkflowDispatcher) -> None:
        self._workflow_dispatcher = workflow_dispatcher

    async def execute_active_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan | None = None,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        workroom_progress = (
            loop_state.workroom_progress
            if loop_state is not None
            else None
        )
        if workroom_progress is None:
            state.finish_reason = "error"
            error_text = "No active workroom is available to continue."
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        active_continuation = _override_active_staffing(
            workroom_progress,
            plan=plan,
        )
        async for event in self._workflow_dispatcher.execute_workroom_continuation(
            request=request,
            continuation=active_continuation,
            pending=None,
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def execute_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._workflow_dispatcher.execute_workroom(
            request=request,
            workroom_id=plan.workroom_id,
            specialists=plan.specialists,
            specialist_counts=plan.specialist_counts,
            workroom_request=plan.workroom_request,
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

    async def execute_workroom_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        pending: PendingContinuation,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._workflow_dispatcher.execute_workroom_continuation(
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
    if plan.specialists or plan.specialist_counts:
        workroom_specialists = tuple(plan.specialists)
        workroom_specialist_counts = tuple(plan.specialist_counts)
    else:
        workroom_specialists = continuation.workroom_specialists
        workroom_specialist_counts = continuation.workroom_specialist_counts
    workroom_request = (
        plan.workroom_request
        if plan is not None and plan.workroom_request
        else continuation.workroom_request
    )
    if (
        workroom_specialists == continuation.workroom_specialists
        and workroom_specialist_counts == continuation.workroom_specialist_counts
        and workroom_request == continuation.workroom_request
    ):
        return continuation
    return replace(
        continuation,
        workroom_specialists=workroom_specialists,
        workroom_specialist_counts=workroom_specialist_counts,
        workroom_request=workroom_request,
    )
