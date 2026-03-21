from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

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
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyFunctionTool,
    ProxyOutputItemRef,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.planner import (
    ProxyTurnPlan,
    build_turn_planner_instructions,
    build_turn_planner_prompt,
    parse_turn_plan,
    summarize_conversation,
)
from ergon_studio.proxy.workflow_metadata import (
    workflow_finalizers_for_definition,
    workflow_handoffs_for_definition,
    workflow_max_rounds_for_definition,
    workflow_orchestration_for_definition,
    workflow_participants_for_definition,
    workflow_selection_sequence_for_definition,
    workflow_start_agent_for_definition,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)

@dataclass
class ProxyTurnState:
    content: str = ""
    reasoning: str = ""
    mode: str = "act"
    finish_reason: str = "stop"
    tool_calls: tuple[ProxyToolCall, ...] = ()
    output_items: tuple[ProxyOutputItemRef, ...] = field(default_factory=tuple)

    def append_content(self, delta: str) -> None:
        self.content += delta
        self.record_output_item("content")

    def set_content(self, content: str) -> None:
        self.content = content
        if content:
            self.record_output_item("content")

    def append_reasoning(self, delta: str) -> None:
        self.reasoning += delta
        self.record_output_item("reasoning")

    def record_output_item(self, kind: str, *, call_id: str | None = None) -> None:
        item = ProxyOutputItemRef(kind, call_id)
        if item in self.output_items:
            return
        self.output_items = (*self.output_items, item)


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
                    plan = await self._plan_turn(request)
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

    async def _plan_turn(self, request: ProxyTurnRequest) -> ProxyTurnPlan:
        planner_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=build_turn_planner_prompt(request),
            preamble=build_turn_planner_instructions(self.registry),
            session_id=f"proxy-planner-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not planner_text:
            return ProxyTurnPlan(mode="act")
        try:
            return parse_turn_plan(planner_text, registry=self.registry)
        except ValueError:
            return ProxyTurnPlan(mode="act")

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
        notice = "Orchestrator: handling this turn directly.\n"
        state.append_reasoning(notice)
        yield ProxyReasoningDeltaEvent(notice)
        prompt = _direct_reply_prompt(request)
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id="orchestrator",
            prompt=prompt,
            session_id=f"proxy-direct-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            pending_continuation=pending,
            final_response_sink=_response_holder_sink(response_holder),
        ):
            state.append_content(delta)
            yield ProxyContentDeltaEvent(delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                request=request,
                continuation=ContinuationState(mode="act", agent_id="orchestrator"),
                state=state,
            )
            for event in emitted:
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
        agent_id = plan.agent_id or "coder"
        intro = f"Orchestrator: delegating this turn to {agent_id}.\n"
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        specialist_prompt = _specialist_prompt(
            specialist_id=agent_id,
            request_text=plan.request or request.latest_user_text() or "",
            transcript_summary=summarize_conversation(request.messages),
            current_brief=current_brief,
        )
        specialist_text = ""
        first = True
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id=agent_id,
            prompt=specialist_prompt,
            session_id=f"proxy-delegate-{agent_id}-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            pending_continuation=pending,
            final_response_sink=_response_holder_sink(response_holder),
        ):
            specialist_text += delta
            reasoning_delta = f"{agent_id}: {delta}" if first else delta
            first = False
            state.append_reasoning(reasoning_delta)
            yield ProxyReasoningDeltaEvent(reasoning_delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                request=request,
                continuation=ContinuationState(
                    mode="delegate",
                    agent_id=agent_id,
                    request_text=plan.request or request.latest_user_text(),
                    current_brief=specialist_text.strip() or current_brief,
                ),
                state=state,
            )
            if emitted:
                for tool_event in emitted:
                    yield tool_event
                return
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_delegation_summary_prompt(
                request_text=request.latest_user_text() or "",
                specialist_id=agent_id,
                specialist_text=specialist_text,
            ),
            preamble=_summary_instructions(),
            session_id=f"proxy-delegation-summary-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not final_text:
            final_text = specialist_text.strip()
        state.set_content(final_text)
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def _execute_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        created_at: int,
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self.registry.workflow_definitions.get(plan.workflow_id or "")
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workflow: {plan.workflow_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: running workflow {definition.id}.\n"
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)

        goal = plan.goal or request.latest_user_text() or ""
        orchestration = workflow_orchestration_for_definition(definition)
        if orchestration in {"sequential", "grouped", "concurrent"}:
            async for event in self._execute_grouped_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
            ):
                yield event
            return
        if orchestration == "group_chat":
            async for event in self._execute_group_chat_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
            ):
                yield event
            return
        if orchestration == "magentic":
            async for event in self._execute_magentic_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
            ):
                yield event
            return
        if orchestration == "handoff":
            async for event in self._execute_handoff_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
            ):
                yield event
            return
        raise ValueError(f"unsupported workflow orchestration: {orchestration}")

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
        definition = self.registry.workflow_definitions.get(
            continuation.workflow_id or ""
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workflow: {continuation.workflow_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        agent_name = continuation.agent_id or "(unknown)"
        intro = (
            f"Orchestrator: continuing workflow {definition.id} with {agent_name}.\n"
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)

        goal = continuation.goal or request.latest_user_text() or ""
        orchestration = workflow_orchestration_for_definition(definition)
        if orchestration in {"sequential", "grouped", "concurrent"}:
            async for event in self._execute_grouped_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
            ):
                yield event
            return
        if orchestration == "group_chat":
            async for event in self._execute_group_chat_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
            ):
                yield event
            return
        if orchestration == "magentic":
            async for event in self._execute_magentic_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
            ):
                yield event
            return
        if orchestration == "handoff":
            async for event in self._execute_handoff_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
            ):
                yield event
            return
        raise ValueError(f"unsupported workflow orchestration: {orchestration}")

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
        step_groups = workflow_step_groups_for_definition(definition)
        start_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        start_agent_index = (
            continuation.agent_index
            if continuation and continuation.agent_index is not None
            else 0
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        for step_index in range(start_index, len(step_groups)):
            group = step_groups[step_index]
            group_start_index = start_agent_index if step_index == start_index else 0
            for agent_index in range(group_start_index, len(group)):
                agent_id = group[agent_index]
                specialist_prompt = _workflow_step_prompt(
                    workflow_id=definition.id,
                    agent_id=agent_id,
                    goal=goal,
                    current_brief=current_brief,
                    transcript_summary=summarize_conversation(request.messages),
                    prior_outputs=tuple(workflow_outputs),
                )
                agent_text = ""
                first = True
                response_holder: dict[str, Any] = {}
                async for delta in self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=specialist_prompt,
                    session_id=f"proxy-workflow-{definition.id}-{agent_id}-{uuid4().hex}",
                    model_id_override=request.model,
                    host_tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    pending_continuation=pending
                    if step_index == start_index and agent_index == group_start_index
                    else None,
                    final_response_sink=_response_holder_sink(response_holder),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_id}: {delta}" if first else delta
                    first = False
                    state.append_reasoning(reasoning_delta)
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                response = response_holder.get("response")
                if response is not None:
                    emitted = self._emit_tool_calls(
                        response=response,
                        request=request,
                        continuation=ContinuationState(
                            mode="workflow",
                            workflow_id=definition.id,
                            step_index=step_index,
                            agent_index=agent_index,
                            agent_id=agent_id,
                            goal=goal,
                            current_brief=agent_text.strip() or current_brief,
                            workflow_outputs=tuple(workflow_outputs),
                        ),
                        state=state,
                    )
                    if emitted:
                        for event in emitted:
                            yield event
                        return
                workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
                current_brief = agent_text.strip() or current_brief
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield summary_event

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
        participants = workflow_participants_for_definition(definition)
        sequence = workflow_selection_sequence_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(
            definition, default=max(len(sequence), len(participants), 1)
        )
        if not sequence:
            sequence = (
                tuple(
                    participants[index % len(participants)]
                    for index in range(max_rounds)
                )
                if participants
                else ()
            )
        else:
            sequence = sequence[:max_rounds]
        start_turn = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        for turn_index in range(start_turn, len(sequence)):
            agent_id = sequence[turn_index]
            prompt = _group_chat_turn_prompt(
                workflow_id=definition.id,
                agent_id=agent_id,
                goal=goal,
                transcript_summary=summarize_conversation(request.messages),
                current_brief=current_brief,
                prior_outputs=tuple(workflow_outputs),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=agent_id,
                prompt=prompt,
                session_id=f"proxy-group-chat-{definition.id}-{agent_id}-{uuid4().hex}",
                model_id_override=request.model,
                host_tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending if turn_index == start_turn else None,
                final_response_sink=_response_holder_sink(response_holder),
            ):
                agent_text += delta
                reasoning_delta = f"{agent_id}: {delta}" if first else delta
                first = False
                state.append_reasoning(reasoning_delta)
                yield ProxyReasoningDeltaEvent(reasoning_delta)
            response = response_holder.get("response")
            if response is not None:
                emitted = self._emit_tool_calls(
                    response=response,
                    request=request,
                    continuation=ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        step_index=turn_index,
                        agent_id=agent_id,
                        goal=goal,
                        current_brief=agent_text.strip() or current_brief,
                        workflow_outputs=tuple(workflow_outputs),
                    ),
                    state=state,
                )
                if emitted:
                    for tool_event in emitted:
                        yield tool_event
                    return
            workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
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
        participants = workflow_participants_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(
            definition, default=max(len(participants), 1)
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        round_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        while round_index < max_rounds:
            agent_id: str | None
            if continuation is not None and round_index == (
                continuation.step_index or 0
            ):
                agent_id = continuation.agent_id
            else:
                agent_id = await self._select_manager_agent(
                    workflow_id=definition.id,
                    goal=goal,
                    current_brief=current_brief,
                    participants=participants,
                    prior_outputs=tuple(workflow_outputs),
                    model_id_override=request.model,
                )
            if agent_id is None:
                break
            prompt = _workflow_step_prompt(
                workflow_id=definition.id,
                agent_id=agent_id,
                goal=goal,
                current_brief=current_brief,
                transcript_summary=summarize_conversation(request.messages),
                prior_outputs=tuple(workflow_outputs),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=agent_id,
                prompt=prompt,
                session_id=f"proxy-magentic-{definition.id}-{agent_id}-{uuid4().hex}",
                model_id_override=request.model,
                host_tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending
                if continuation is not None
                and round_index == (continuation.step_index or 0)
                else None,
                final_response_sink=_response_holder_sink(response_holder),
            ):
                agent_text += delta
                reasoning_delta = f"{agent_id}: {delta}" if first else delta
                first = False
                state.append_reasoning(reasoning_delta)
                yield ProxyReasoningDeltaEvent(reasoning_delta)
            response = response_holder.get("response")
            if response is not None:
                emitted = self._emit_tool_calls(
                    response=response,
                    request=request,
                    continuation=ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        step_index=round_index,
                        agent_id=agent_id,
                        goal=goal,
                        current_brief=agent_text.strip() or current_brief,
                        workflow_outputs=tuple(workflow_outputs),
                    ),
                    state=state,
                )
                if emitted:
                    for tool_event in emitted:
                        yield tool_event
                    return
            workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            round_index += 1
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
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
        participants = workflow_participants_for_definition(definition)
        finalizers = workflow_finalizers_for_definition(definition)
        handoffs = workflow_handoffs_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(
            definition, default=max(len(participants), 1)
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        round_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        current_agent: str | None = (
            continuation.agent_id
            if continuation is not None and continuation.agent_id is not None
            else workflow_start_agent_for_definition(definition)
            or (participants[0] if participants else "reviewer")
        )

        while round_index < max_rounds and current_agent:
            prompt = _workflow_step_prompt(
                workflow_id=definition.id,
                agent_id=current_agent,
                goal=goal,
                current_brief=current_brief,
                transcript_summary=summarize_conversation(request.messages),
                prior_outputs=tuple(workflow_outputs),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=current_agent,
                prompt=prompt,
                session_id=f"proxy-handoff-{definition.id}-{current_agent}-{uuid4().hex}",
                model_id_override=request.model,
                host_tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending
                if continuation is not None
                and round_index == (continuation.step_index or 0)
                else None,
                final_response_sink=_response_holder_sink(response_holder),
            ):
                agent_text += delta
                reasoning_delta = f"{current_agent}: {delta}" if first else delta
                first = False
                state.append_reasoning(reasoning_delta)
                yield ProxyReasoningDeltaEvent(reasoning_delta)
            response = response_holder.get("response")
            if response is not None:
                emitted = self._emit_tool_calls(
                    response=response,
                    request=request,
                    continuation=ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        step_index=round_index,
                        agent_id=current_agent,
                        goal=goal,
                        current_brief=agent_text.strip() or current_brief,
                        workflow_outputs=tuple(workflow_outputs),
                    ),
                    state=state,
                )
                if emitted:
                    for tool_event in emitted:
                        yield tool_event
                    return
            workflow_outputs.append(f"{current_agent}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            if current_agent in finalizers:
                break
            current_agent = await self._select_handoff_target(
                workflow_id=definition.id,
                current_agent=current_agent,
                goal=goal,
                current_brief=current_brief,
                prior_outputs=tuple(workflow_outputs),
                allowed=handoffs.get(
                    current_agent,
                    tuple(agent for agent in participants if agent != current_agent),
                ),
                model_id_override=request.model,
            )
            round_index += 1
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield summary_event

    async def _emit_workflow_summary(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        current_brief: str,
        workflow_outputs: tuple[str, ...],
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_workflow_summary_prompt(
                workflow_id=definition.id,
                goal=goal,
                outputs=workflow_outputs,
            ),
            preamble=_summary_instructions(),
            session_id=f"proxy-workflow-summary-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not final_text:
            final_text = current_brief.strip()
        state.set_content(final_text)
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def _select_manager_agent(
        self,
        *,
        workflow_id: str,
        goal: str,
        current_brief: str,
        participants: tuple[str, ...],
        prior_outputs: tuple[str, ...],
        model_id_override: str,
    ) -> str | None:
        raw = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=_workflow_manager_prompt(
                workflow_id=workflow_id,
                goal=goal,
                current_brief=current_brief,
                participants=participants,
                prior_outputs=prior_outputs,
            ),
            preamble=_workflow_manager_instructions(participants),
            session_id=f"proxy-workflow-manager-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return _parse_agent_selection(raw, participants=participants)

    async def _select_handoff_target(
        self,
        *,
        workflow_id: str,
        current_agent: str,
        goal: str,
        current_brief: str,
        prior_outputs: tuple[str, ...],
        allowed: tuple[str, ...],
        model_id_override: str,
    ) -> str | None:
        if not allowed:
            return None
        raw = await self._run_text_agent(
            agent_id=current_agent,
            prompt=_handoff_selection_prompt(
                workflow_id=workflow_id,
                current_agent=current_agent,
                goal=goal,
                current_brief=current_brief,
                prior_outputs=prior_outputs,
                allowed=allowed,
            ),
            preamble=_handoff_selection_instructions(allowed),
            session_id=f"proxy-handoff-select-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return _parse_agent_selection(raw, participants=allowed)

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

    def _emit_tool_calls(
        self,
        *,
        response: Any,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        state: ProxyTurnState,
    ) -> list[ProxyToolCallEvent]:
        encoded_calls, events = self._agent_runner.emit_tool_calls(
            response=response,
            request=request,
            continuation=continuation,
        )
        if not events:
            return []
        state.tool_calls = encoded_calls
        state.finish_reason = "tool_calls"
        for call in encoded_calls:
            state.record_output_item("tool_call", call_id=call.id)
        return events

def _direct_reply_prompt(request: ProxyTurnRequest) -> str:
    return "\n".join(
        [
            "You are responding to the host user in proxy mode.",
            "Use the full conversation transcript below as context.",
            "",
            summarize_conversation(request.messages, limit=12),
            "",
            "Latest user request:",
            request.latest_user_text() or "(none)",
        ]
    ).strip()


def _specialist_prompt(
    *,
    specialist_id: str,
    request_text: str,
    transcript_summary: str,
    current_brief: str | None = None,
) -> str:
    lines = [
        f"You are the {specialist_id} working inside the orchestration proxy.",
        "The orchestrator distilled the host conversation for you.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Assigned request:",
        request_text or "(none)",
    ]
    if current_brief:
        lines.extend(
            [
                "",
                "Current progress:",
                current_brief,
            ]
        )
    return "\n".join(lines).strip()


def _workflow_step_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    goal: str,
    current_brief: str,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} working inside workflow {workflow_id}.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Overall goal:",
        goal or "(none)",
        "",
        "Current brief:",
        current_brief or "(none)",
    ]
    if prior_outputs:
        lines.extend(
            [
                "",
                "Prior workflow outputs:",
                *prior_outputs[-6:],
            ]
        )
    return "\n".join(lines).strip()


def _group_chat_turn_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    goal: str,
    transcript_summary: str,
    current_brief: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} speaking in group chat workflow {workflow_id}.",
        "Respond to the current discussion and move the decision forward.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Goal:",
        goal or "(none)",
        "",
        "Current brief:",
        current_brief or "(none)",
    ]
    if prior_outputs:
        lines.extend(
            [
                "",
                "Discussion so far:",
                *prior_outputs[-8:],
            ]
        )
    return "\n".join(lines).strip()


def _workflow_manager_instructions(participants: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "You are selecting the next specialist for an adaptive workflow.",
            "Return JSON only.",
            f"Allowed agents: {', '.join(participants) or '(none)'}",
            (
                'Return {"agent_id":"<agent>" } to continue or '
                '{"agent_id":null} to finish.'
            ),
        ]
    )


def _workflow_manager_prompt(
    *,
    workflow_id: str,
    goal: str,
    current_brief: str,
    participants: tuple[str, ...],
    prior_outputs: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"Workflow: {workflow_id}",
            f"Goal: {goal or '(none)'}",
            f"Current brief: {current_brief or '(none)'}",
            f"Available specialists: {', '.join(participants) or '(none)'}",
            "",
            "Progress so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    ).strip()


def _handoff_selection_instructions(allowed: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "You are choosing the next specialist handoff.",
            "Return JSON only.",
            f"Allowed next agents: {', '.join(allowed) or '(none)'}",
            (
                'Return {"agent_id":"<agent>" } to continue or '
                '{"agent_id":null} to finish.'
            ),
        ]
    )


def _handoff_selection_prompt(
    *,
    workflow_id: str,
    current_agent: str,
    goal: str,
    current_brief: str,
    prior_outputs: tuple[str, ...],
    allowed: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"Workflow: {workflow_id}",
            f"You are {current_agent}.",
            f"Goal: {goal or '(none)'}",
            f"Current brief: {current_brief or '(none)'}",
            f"You may hand off to: {', '.join(allowed) or '(none)'}",
            "",
            "Work so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    ).strip()


def _parse_agent_selection(
    raw: str | None, *, participants: tuple[str, ...]
) -> str | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("agent_id")
    if candidate is None:
        return None
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    if not stripped or stripped.casefold() in {"none", "null", "finish", "done"}:
        return None
    if stripped not in participants:
        return None
    return stripped


def _summary_instructions() -> str:
    return "\n".join(
        [
            "Summarize the completed work for the host user.",
            "Be concise and concrete.",
            "State what was decided or produced.",
            "Do not mention hidden chain-of-thought.",
        ]
    )


def _delegation_summary_prompt(
    *, request_text: str, specialist_id: str, specialist_text: str
) -> str:
    return "\n".join(
        [
            f"The specialist {specialist_id} completed delegated work.",
            "",
            "Original request:",
            request_text or "(none)",
            "",
            "Specialist output:",
            specialist_text or "(none)",
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()


def _workflow_summary_prompt(
    *, workflow_id: str, goal: str, outputs: tuple[str, ...]
) -> str:
    return "\n".join(
        [
            f"The workflow {workflow_id} completed.",
            "",
            "Goal:",
            goal or "(none)",
            "",
            "Workflow outputs:",
            *(outputs or ("(none)",)),
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()


def _set_response_holder(scope: dict[str, Any], value: Any) -> None:
    scope["response"] = value


def _response_holder_sink(scope: dict[str, Any]) -> Callable[[Any], None]:
    return lambda value: _set_response_holder(scope, value)
