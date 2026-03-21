from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_framework import ResponseStream

from ergon_studio.agent_factory import build_agent
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import (
    ProxyAgentRunner,
    ProxyToolChoice,
)
from ergon_studio.proxy.continuation import (
    ContinuationState,
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
    ProxyFunctionTool,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.tool_call_emitter import ProxyToolCallEmitter
from ergon_studio.proxy.turn_executor import ProxyTurnExecutor
from ergon_studio.proxy.turn_planner import ProxyTurnPlanner
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher
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
        self._workflow_dispatcher = ProxyWorkflowDispatcher(
            registry,
            execute_grouped_workflow=self._execute_grouped_workflow,
            execute_group_chat_workflow=self._execute_group_chat_workflow,
            execute_magentic_workflow=self._execute_magentic_workflow,
            execute_handoff_workflow=self._execute_handoff_workflow,
        )
        self._workflow_support = ProxyWorkflowSupport(
            run_text_agent=self._run_text_agent,
        )
        self._turn_executor = ProxyTurnExecutor(
            stream_text_agent=self._stream_text_agent,
            run_text_agent=self._run_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
        )
        self._turn_planner = ProxyTurnPlanner(
            registry,
            run_text_agent=self._run_text_agent,
        )
        self._grouped_workflow_executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=self._stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
        )
        self._group_chat_workflow_executor = ProxyGroupChatWorkflowExecutor(
            stream_text_agent=self._stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
        )
        self._magentic_workflow_executor = ProxyMagenticWorkflowExecutor(
            stream_text_agent=self._stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
            select_manager_agent=self._workflow_support.select_manager_agent,
        )
        self._handoff_workflow_executor = ProxyHandoffWorkflowExecutor(
            stream_text_agent=self._stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
            emit_workflow_summary=self._workflow_support.emit_summary,
            select_handoff_target=self._workflow_support.select_handoff_target,
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
                    async for event in self._execute_continuation(
                        request=request,
                        pending=pending,
                        created_at=created_at,
                        state=state,
                    ):
                        yield event
                else:
                    plan = await self._turn_planner.plan_turn(request)
                    state.mode = plan.mode
                    async for event in self._execute_plan(
                        request=request,
                        plan=plan,
                        created_at=created_at,
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

    async def _execute_plan(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        created_at: int,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        if plan.mode == "delegate" and plan.agent_id is not None:
            async for event in self._execute_delegation(
                request=request, plan=plan, created_at=created_at, state=state
            ):
                yield event
            return
        if plan.mode == "workflow" and plan.workflow_id is not None:
            async for event in self._execute_workflow(
                request=request, plan=plan, created_at=created_at, state=state
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request, created_at=created_at, state=state
        ):
            yield event

    async def _execute_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        pending: PendingContinuation,
        created_at: int,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        continuation = pending.state
        if continuation.mode == "workflow" and continuation.workflow_id is not None:
            async for event in self._execute_workflow_continuation(
                request=request,
                continuation=continuation,
                pending=pending,
                created_at=created_at,
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
                created_at=created_at,
                state=state,
                current_brief=continuation.current_brief,
                pending=pending,
            ):
                yield event
            return
        async for event in self._execute_direct(
            request=request, created_at=created_at, state=state, pending=pending
        ):
            yield event

    async def _execute_direct(
        self,
        *,
        request: ProxyTurnRequest,
        created_at: int,
        state: ProxyTurnState,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        del created_at
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
        created_at: int,
        state: ProxyTurnState,
        current_brief: str | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        del created_at
        async for event in self._turn_executor.execute_delegation(
            request=request,
            plan=plan,
            state=state,
            current_brief=current_brief,
            pending=pending,
        ):
            yield event

    async def _execute_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        created_at: int,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        del created_at
        async for event in self._workflow_dispatcher.execute_workflow(
            request=request,
            workflow_id=plan.workflow_id,
            goal=plan.goal or request.latest_user_text() or "",
            state=state,
        ):
            yield event

    async def _execute_workflow_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        pending: PendingContinuation,
        created_at: int,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        del created_at
        async for event in self._workflow_dispatcher.execute_workflow_continuation(
            request=request,
            continuation=continuation,
            pending=pending,
            state=state,
        ):
            yield event

    async def _execute_grouped_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for event in self._grouped_workflow_executor.execute(
            request=request,
            definition=definition,
            goal=goal,
            state=state,
            continuation=continuation,
            pending=pending,
        ):
            yield event

    async def _execute_group_chat_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for summary_event in self._group_chat_workflow_executor.execute(
            request=request,
            definition=definition,
            goal=goal,
            state=state,
            continuation=continuation,
            pending=pending,
        ):
            yield summary_event

    async def _execute_magentic_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for summary_event in self._magentic_workflow_executor.execute(
            request=request,
            definition=definition,
            goal=goal,
            state=state,
            continuation=continuation,
            pending=pending,
        ):
            yield summary_event

    async def _execute_handoff_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        async for summary_event in self._handoff_workflow_executor.execute(
            request=request,
            definition=definition,
            goal=goal,
            state=state,
            continuation=continuation,
            pending=pending,
        ):
            yield summary_event

    async def _run_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        preamble: str = "",
        pending_continuation: PendingContinuation | None = None,
    ) -> str | None:
        return await self._agent_runner.run_text_agent(
            agent_id=agent_id,
            prompt=prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            preamble=preamble,
            pending_continuation=pending_continuation,
        )

    async def _stream_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        preamble: str = "",
        host_tools: tuple[ProxyFunctionTool, ...] = (),
        tool_choice: ProxyToolChoice = None,
        parallel_tool_calls: bool | None = None,
        pending_continuation: PendingContinuation | None = None,
        final_response_sink: Callable[[Any], None] | None = None,
    ) -> AsyncIterator[str]:
        async for delta in self._agent_runner.stream_text_agent(
            agent_id=agent_id,
            prompt=prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            preamble=preamble,
            host_tools=host_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            pending_continuation=pending_continuation,
            final_response_sink=final_response_sink,
        ):
            yield delta
