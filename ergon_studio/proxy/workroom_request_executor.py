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
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workroom_dispatcher import ProxyWorkroomDispatcher

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyWorkroomRequestExecutor:
    def __init__(self, workroom_dispatcher: ProxyWorkroomDispatcher) -> None:
        self._workroom_dispatcher = workroom_dispatcher

    async def execute_active_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        message: str | None,
        participants: tuple[str, ...],
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        workroom_progress = (
            loop_state.workroom_progress if loop_state is not None else None
        )
        if workroom_progress is None:
            state.finish_reason = "error"
            error_text = "No active workroom is available to continue."
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        active_continuation = _override_active_staffing(
            workroom_progress,
            message=message,
            participants=participants,
        )
        async for event in self._workroom_dispatcher.execute_workroom_continuation(
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
        workroom_id: str | None,
        participants: tuple[str, ...],
        workroom_request: str | None,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._workroom_dispatcher.execute_workroom(
            request=request,
            workroom_id=workroom_id,
            participants=participants,
            workroom_request=workroom_request,
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
        async for event in self._workroom_dispatcher.execute_workroom_continuation(
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
    message: str | None,
    participants: tuple[str, ...],
) -> ContinuationState:
    workroom_participants = participants or continuation.workroom_participants
    workroom_request = message or continuation.workroom_request
    if (
        workroom_participants == continuation.workroom_participants
        and workroom_request == continuation.workroom_request
    ):
        return continuation
    return replace(
        continuation,
        workroom_participants=workroom_participants,
        workroom_request=workroom_request,
    )
