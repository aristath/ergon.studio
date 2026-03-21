from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_framework import ResponseStream

from ergon_studio.agent_factory import build_agent
from ergon_studio.proxy.agent_runner import ProxyAgentRunner
from ergon_studio.proxy.continuation import (
    PendingContinuation,
    latest_pending_continuation,
)
from ergon_studio.proxy.group_chat_workflow_executor import (
    ProxyGroupChatWorkflowExecutor,
)
from ergon_studio.proxy.grouped_workflow_executor import ProxyGroupedWorkflowExecutor
from ergon_studio.proxy.handoff_workflow_executor import (
    ProxyHandoffWorkflowExecutor,
)
from ergon_studio.proxy.magentic_workflow_executor import (
    ProxyMagenticWorkflowExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.tool_call_emitter import ProxyToolCallEmitter
from ergon_studio.proxy.turn_executor import ProxyTurnExecutor
from ergon_studio.proxy.turn_planner import ProxyTurnPlanner
from ergon_studio.proxy.turn_router import ProxyTurnRouter
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState, ProxyTurnState
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher
from ergon_studio.proxy.workflow_request_executor import (
    ProxyWorkflowRequestExecutor,
)
from ergon_studio.proxy.workflow_support import ProxyWorkflowSupport
from ergon_studio.registry import RuntimeRegistry

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyOrchestrationCore:
    _MAX_INTERNAL_MOVES = 6

    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Callable[..., Any] = build_agent,
    ) -> None:
        self.registry = registry
        agent_runner = ProxyAgentRunner(
            registry,
            agent_builder=agent_builder,
        )
        tool_call_emitter = ProxyToolCallEmitter(agent_runner)
        workflow_support = ProxyWorkflowSupport(
            run_text_agent=agent_runner.run_text_agent,
        )
        turn_executor = ProxyTurnExecutor(
            stream_text_agent=agent_runner.stream_text_agent,
            run_text_agent=agent_runner.run_text_agent,
            emit_tool_calls=tool_call_emitter.emit_tool_calls,
        )
        self._turn_planner = ProxyTurnPlanner(
            registry,
            run_text_agent=agent_runner.run_text_agent,
        )
        grouped_workflow_executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=agent_runner.stream_text_agent,
            emit_tool_calls=tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=workflow_support.emit_summary,
        )
        group_chat_workflow_executor = ProxyGroupChatWorkflowExecutor(
            stream_text_agent=agent_runner.stream_text_agent,
            emit_tool_calls=tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=workflow_support.emit_summary,
        )
        magentic_workflow_executor = ProxyMagenticWorkflowExecutor(
            stream_text_agent=agent_runner.stream_text_agent,
            emit_tool_calls=tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=workflow_support.emit_summary,
            select_manager_agent=workflow_support.select_manager_agent,
        )
        handoff_workflow_executor = ProxyHandoffWorkflowExecutor(
            stream_text_agent=agent_runner.stream_text_agent,
            emit_tool_calls=tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=workflow_support.emit_summary,
            select_handoff_target=workflow_support.select_handoff_target,
        )
        workflow_dispatcher = ProxyWorkflowDispatcher(
            registry,
            execute_grouped_workflow=grouped_workflow_executor.execute,
            execute_group_chat_workflow=group_chat_workflow_executor.execute,
            execute_magentic_workflow=magentic_workflow_executor.execute,
            execute_handoff_workflow=handoff_workflow_executor.execute,
        )
        workflow_request_executor = ProxyWorkflowRequestExecutor(
            workflow_dispatcher,
        )
        self._turn_router = ProxyTurnRouter(
            execute_direct=turn_executor.execute_direct,
            execute_delegation=turn_executor.execute_delegation,
            execute_workflow=workflow_request_executor.execute_workflow,
            execute_workflow_continuation=(
                workflow_request_executor.execute_workflow_continuation
            ),
        )

    def stream_turn(
        self,
        request: ProxyTurnRequest,
        *,
        created_at: int | None = None,
    ) -> ResponseStream[ProxyEvent, ProxyTurnResult]:
        if created_at is None:
            created_at = int(time.time())
        state = ProxyTurnState()

        async def _events() -> AsyncIterator[ProxyEvent]:
            try:
                pending = latest_pending_continuation(request.messages)
                loop_state = self._initial_loop_state(request, pending=pending)
                if pending is not None:
                    state.mode = pending.state.mode
                    result_holder: dict[str, object] = {}
                    async for event in self._turn_router.execute_continuation(
                        request=request,
                        pending=pending,
                        state=state,
                        loop_state=loop_state,
                        result_sink=_result_sink(result_holder),
                    ):
                        yield event
                    if state.finish_reason == "tool_calls":
                        pass
                    elif result_holder:
                        loop_state.absorb_result(
                            worklog_lines=_result_lines(result_holder),
                            current_brief=_result_brief(result_holder, loop_state),
                        )
                        async for event in self._run_decision_loop(
                            request=request,
                            state=state,
                            loop_state=loop_state,
                        ):
                            yield event
                else:
                    async for event in self._run_decision_loop(
                        request=request,
                        state=state,
                        loop_state=loop_state,
                    ):
                        yield event
            except ValueError as exc:
                state.finish_reason = "error"
                state.content = str(exc)
                yield ProxyContentDeltaEvent(state.content)
            except Exception as exc:
                state.finish_reason = "error"
                state.content = f"{type(exc).__name__}: {exc}"
                yield ProxyContentDeltaEvent(state.content)
            yield ProxyFinishEvent(state.finish_reason)

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason=state.finish_reason,
                content=state.content,
                reasoning=state.reasoning,
                mode=state.mode,
                tool_calls=state.tool_calls,
                output_items=state.output_items,
            ),
        )

    async def _run_decision_loop(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        loop_state: ProxyDecisionLoopState,
    ) -> AsyncIterator[ProxyEvent]:
        for _ in range(self._MAX_INTERNAL_MOVES):
            plan = await self._turn_planner.plan_turn(
                request,
                loop_state=loop_state,
            )
            state.mode = plan.mode
            if plan.goal:
                loop_state.goal = plan.goal
            result_holder: dict[str, object] = {}
            async for event in self._turn_router.execute_plan(
                request=request,
                plan=plan,
                state=state,
                loop_state=loop_state,
                result_sink=_result_sink(result_holder),
            ):
                yield event
            if state.finish_reason == "tool_calls":
                return
            if plan.mode == "act":
                return
            if not result_holder:
                return
            loop_state.absorb_result(
                worklog_lines=_result_lines(result_holder),
                current_brief=_result_brief(result_holder, loop_state),
            )

        async for event in self._turn_router.execute_plan(
            request=request,
            plan=ProxyTurnPlan(mode="act"),
            state=state,
            loop_state=loop_state,
        ):
            yield event

    def _initial_loop_state(
        self,
        request: ProxyTurnRequest,
        *,
        pending: PendingContinuation | None,
    ) -> ProxyDecisionLoopState:
        if pending is None:
            goal = request.latest_user_text() or ""
            return ProxyDecisionLoopState(
                goal=goal,
                current_brief=goal,
            )
        continuation = pending.state
        goal = continuation.goal or request.latest_user_text() or ""
        current_brief = continuation.current_brief or goal
        return ProxyDecisionLoopState(
            goal=goal,
            current_brief=current_brief,
            worklog=continuation.decision_history,
        )


def _result_sink(
    holder: dict[str, object],
) -> Callable[[tuple[str, ...], str], None]:
    def _capture(worklog_lines: tuple[str, ...], current_brief: str) -> None:
        holder["worklog_lines"] = worklog_lines
        holder["current_brief"] = current_brief

    return _capture


def _result_lines(holder: dict[str, object]) -> tuple[str, ...]:
    value = holder.get("worklog_lines")
    if isinstance(value, tuple):
        return tuple(line for line in value if isinstance(line, str))
    return ()


def _result_brief(
    holder: dict[str, object],
    loop_state: ProxyDecisionLoopState,
) -> str:
    value = holder.get("current_brief")
    if isinstance(value, str) and value:
        return value
    return loop_state.current_brief
