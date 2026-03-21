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
from ergon_studio.proxy.turn_state import ProxyTurnState
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
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Callable[..., Any] = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_runner = ProxyAgentRunner(
            registry,
            agent_builder=agent_builder,
        )
        self._tool_call_emitter = ProxyToolCallEmitter(self._agent_runner)
        self._workflow_support = ProxyWorkflowSupport(
            run_text_agent=self._agent_runner.run_text_agent,
        )
        self._turn_executor = ProxyTurnExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            run_text_agent=self._agent_runner.run_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
        )
        self._turn_planner = ProxyTurnPlanner(
            registry,
            run_text_agent=self._agent_runner.run_text_agent,
        )
        self._grouped_workflow_executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
        )
        self._group_chat_workflow_executor = ProxyGroupChatWorkflowExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
        )
        self._magentic_workflow_executor = ProxyMagenticWorkflowExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
            select_manager_agent=self._workflow_support.select_manager_agent,
        )
        self._handoff_workflow_executor = ProxyHandoffWorkflowExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
            select_handoff_target=self._workflow_support.select_handoff_target,
        )
        self._workflow_dispatcher = ProxyWorkflowDispatcher(
            registry,
            execute_grouped_workflow=self._grouped_workflow_executor.execute,
            execute_group_chat_workflow=self._group_chat_workflow_executor.execute,
            execute_magentic_workflow=self._magentic_workflow_executor.execute,
            execute_handoff_workflow=self._handoff_workflow_executor.execute,
        )
        self._workflow_request_executor = ProxyWorkflowRequestExecutor(
            self._workflow_dispatcher,
        )
        self._turn_router = ProxyTurnRouter(
            execute_direct=self._execute_direct,
            execute_delegation=self._execute_delegation,
            execute_workflow=self._workflow_request_executor.execute_workflow,
            execute_workflow_continuation=(
                self._workflow_request_executor.execute_workflow_continuation
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
                if pending is not None:
                    state.mode = pending.state.mode
                    async for event in self._turn_router.execute_continuation(
                        request=request,
                        pending=pending,
                        state=state,
                    ):
                        yield event
                else:
                    plan = await self._turn_planner.plan_turn(request)
                    state.mode = plan.mode
                    async for event in self._turn_router.execute_plan(
                        request=request,
                        plan=plan,
                        state=state,
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

    async def _execute_direct(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._turn_executor.execute_direct(
            request=request,
            state=state,
            pending=pending,
        ):
            yield event

    async def _execute_delegation(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        current_brief: str | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._turn_executor.execute_delegation(
            request=request,
            plan=plan,
            state=state,
            current_brief=current_brief,
            pending=pending,
        ):
            yield event
