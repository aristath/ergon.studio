from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from ergon_studio.proxy.continuation import PendingContinuation
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

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)

TurnHandler = Callable[..., AsyncIterator[ProxyEvent]]


class ProxyTurnRouter:
    def __init__(
        self,
        *,
        execute_direct: TurnHandler,
        execute_finish: TurnHandler,
        execute_delegation: TurnHandler,
        execute_workroom: TurnHandler,
        execute_active_workroom: TurnHandler,
        execute_workroom_continuation: TurnHandler,
    ) -> None:
        self._execute_direct = execute_direct
        self._execute_finish = execute_finish
        self._execute_delegation = execute_delegation
        self._execute_workroom = execute_workroom
        self._execute_active_workroom = execute_active_workroom
        self._execute_workroom_continuation = execute_workroom_continuation

    async def execute_plan(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        loop_state: ProxyDecisionLoopState | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        if plan.mode == "finish":
            async for event in self._execute_finish(
                request=request,
                state=state,
                loop_state=loop_state,
            ):
                yield event
            return
        if plan.mode == "delegate" and plan.agent_id is not None:
            async for event in self._execute_delegation(
                request=request,
                plan=plan,
                state=state,
                loop_state=loop_state,
                result_sink=result_sink,
            ):
                yield event
            return
        if plan.mode == "workroom" and plan.workroom_id is not None:
            async for event in self._execute_workroom(
                request=request,
                plan=plan,
                state=state,
                loop_state=loop_state,
                result_sink=result_sink,
            ):
                yield event
            return
        if plan.mode == "continue_workroom":
            async for event in self._execute_active_workroom(
                request=request,
                plan=plan,
                state=state,
                loop_state=loop_state,
                result_sink=result_sink,
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request,
            state=state,
            loop_state=loop_state,
        ):
            yield event

    async def execute_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        pending: PendingContinuation,
        state: ProxyTurnState,
        loop_state: ProxyDecisionLoopState | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        continuation = pending.state
        if continuation.mode == "workroom" and continuation.workroom_id is not None:
            async for event in self._execute_workroom_continuation(
                request=request,
                continuation=continuation,
                pending=pending,
                state=state,
                loop_state=loop_state,
                result_sink=result_sink,
            ):
                yield event
            return
        if continuation.mode == "finish":
            async for event in self._execute_finish(
                request=request,
                state=state,
                loop_state=loop_state,
            ):
                yield event
            return
        if continuation.mode == "delegate":
            plan = ProxyTurnPlan(
                mode="delegate",
                agent_id=continuation.agent_id,
                request=continuation.request_text or request.latest_user_text(),
            )
            async for event in self._execute_delegation(
                request=request,
                plan=plan,
                state=state,
                current_brief=continuation.current_brief,
                pending=pending,
                loop_state=loop_state,
                result_sink=result_sink,
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request,
            state=state,
            pending=pending,
            loop_state=loop_state,
        ):
            yield event
