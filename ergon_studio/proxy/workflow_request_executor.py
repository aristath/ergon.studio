from __future__ import annotations

from collections.abc import AsyncIterator, Callable

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

    async def execute_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        workflow_progress = (
            loop_state.workflow_progress
            if loop_state is not None
            and loop_state.workflow_progress is not None
            and loop_state.workflow_progress.workflow_id == plan.workflow_id
            else None
        )
        if workflow_progress is not None:
            async for event in self._workflow_dispatcher.execute_workflow_continuation(
                request=request,
                continuation=workflow_progress,
                pending=None,
                state=state,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        async for event in self._workflow_dispatcher.execute_workflow(
            request=request,
            workflow_id=plan.workflow_id,
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
