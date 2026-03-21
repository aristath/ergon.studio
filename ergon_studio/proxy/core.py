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
from ergon_studio.proxy.delivery_requirements import unmet_delivery_requirements
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
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher
from ergon_studio.proxy.workflow_support import ProxyWorkflowSupport
from ergon_studio.proxy.workroom_request_executor import ProxyWorkroomRequestExecutor
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
        workroom_dispatcher = ProxyWorkflowDispatcher(
            registry,
            execute_grouped_workflow=grouped_workflow_executor.execute,
            execute_group_chat_workflow=group_chat_workflow_executor.execute,
            execute_magentic_workflow=magentic_workflow_executor.execute,
            execute_handoff_workflow=handoff_workflow_executor.execute,
        )
        workroom_request_executor = ProxyWorkroomRequestExecutor(
            workroom_dispatcher
        )
        self._turn_router = ProxyTurnRouter(
            execute_direct=turn_executor.execute_direct,
            execute_finish=turn_executor.execute_finish,
            execute_delegation=turn_executor.execute_delegation,
            execute_workroom=workroom_request_executor.execute_workroom,
            execute_active_workroom=(
                workroom_request_executor.execute_active_workroom
            ),
            execute_workroom_continuation=(
                workroom_request_executor.execute_workroom_continuation
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
                            result=_result(result_holder, loop_state)
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
            if plan.delivery_requirements is not None:
                loop_state.delivery_requirements = plan.delivery_requirements
            loop_state.current_move_rationale = plan.rationale
            loop_state.current_workroom_request = plan.workroom_request
            if plan.mode == "finish":
                unmet = unmet_delivery_requirements(
                    loop_state.delivery_requirements,
                    loop_state.delivery_evidence,
                )
                if unmet:
                    note = (
                        "Orchestrator: delivery is blocked until these requirements "
                        f"are satisfied: {', '.join(unmet)}."
                    )
                    state.append_reasoning(note)
                    loop_state.worklog = (*loop_state.worklog, note)
                    yield ProxyReasoningDeltaEvent(note)
                    loop_state.current_move_rationale = None
                    loop_state.current_workroom_request = None
                    continue
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
            if plan.mode in {"act", "finish"}:
                loop_state.current_move_rationale = None
                loop_state.current_workroom_request = None
                return
            if not result_holder:
                loop_state.current_move_rationale = None
                loop_state.current_workroom_request = None
                return
            loop_state.absorb_result(result=_result(result_holder, loop_state))

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
            delivery_requirements=continuation.delivery_requirements,
            delivery_evidence=continuation.delivery_evidence,
            current_workroom_request=continuation.workroom_request,
        )


def _result_sink(
    holder: dict[str, object],
) -> Callable[[ProxyMoveResult], None]:
    def _capture(result: ProxyMoveResult) -> None:
        holder["result"] = result

    return _capture


def _result(
    holder: dict[str, object],
    loop_state: ProxyDecisionLoopState,
) -> ProxyMoveResult:
    value = holder.get("result")
    if isinstance(value, ProxyMoveResult):
        return value
    return ProxyMoveResult(
        worklog_lines=(),
        current_brief=loop_state.current_brief,
    )
