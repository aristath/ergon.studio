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
from ergon_studio.proxy.turn_state import ProxyTurnState

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
        execute_delegation: TurnHandler,
        execute_workflow: TurnHandler,
        execute_workflow_continuation: TurnHandler,
    ) -> None:
        self._execute_direct = execute_direct
        self._execute_delegation = execute_delegation
        self._execute_workflow = execute_workflow
        self._execute_workflow_continuation = execute_workflow_continuation

    async def execute_plan(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        if plan.mode == "delegate" and plan.agent_id is not None:
            async for event in self._execute_delegation(
                request=request,
                plan=plan,
                state=state,
            ):
                yield event
            return
        if plan.mode == "workflow" and plan.workflow_id is not None:
            async for event in self._execute_workflow(
                request=request,
                plan=plan,
                state=state,
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request,
            state=state,
        ):
            yield event

    async def execute_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        pending: PendingContinuation,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        continuation = pending.state
        if continuation.mode == "workflow" and continuation.workflow_id is not None:
            async for event in self._execute_workflow_continuation(
                request=request,
                continuation=continuation,
                pending=pending,
                state=state,
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
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request,
            state=state,
            pending=pending,
        ):
            yield event
