from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any
from uuid import uuid4

from agent_framework import Content, Message, ResponseStream

from ergon_studio.agent_factory import build_agent, provider_supports_tool_calling
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    continuation_result_map,
    continuation_tool_calls,
    decode_original_tool_call,
    encode_continuation_tool_call,
    latest_pending_continuation,
    original_tool_call_id,
)
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyOutputItemRef, ProxyReasoningDeltaEvent, ProxyToolCall, ProxyToolCallEvent, ProxyTurnResult
from ergon_studio.proxy.planner import ProxyTurnPlan, build_turn_planner_instructions, build_turn_planner_prompt, parse_turn_plan, summarize_conversation
from ergon_studio.proxy.tool_policy import resolve_agent_tool_policy
from ergon_studio.proxy.tool_passthrough import build_declaration_tools, extract_tool_calls
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


class ProxyOrchestrationCore:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Callable[..., Any] = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_builder = agent_builder

    def stream_turn(
        self,
        request,
        *,
        created_at: int | None = None,
    ) -> ResponseStream[ProxyReasoningDeltaEvent | ProxyContentDeltaEvent | ProxyToolCallEvent | ProxyFinishEvent, ProxyTurnResult]:
        if created_at is None:
            created_at = int(time.time())
        state: dict[str, Any] = {
            "content": "",
            "reasoning": "",
            "mode": "act",
            "finish_reason": "stop",
            "tool_calls": (),
            "output_items": (),
        }

        async def _events():
            try:
                pending = latest_pending_continuation(request.messages)
                if pending is not None:
                    state["mode"] = pending.state.mode
                    async for event in self._execute_continuation(
                        request=request,
                        pending=pending,
                        created_at=created_at,
                        state=state,
                    ):
                        yield event
                else:
                    plan = await self._plan_turn(request)
                    state["mode"] = plan.mode
                    async for event in self._execute_plan(
                        request=request,
                        plan=plan,
                        created_at=created_at,
                        state=state,
                    ):
                        yield event
            except ValueError as exc:
                state["finish_reason"] = "error"
                state["content"] = str(exc)
                yield ProxyContentDeltaEvent(state["content"])
            yield ProxyFinishEvent(state["finish_reason"])

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason=state["finish_reason"],
                content=state["content"],
                reasoning=state["reasoning"],
                mode=state["mode"],
                tool_calls=state["tool_calls"],
                output_items=state["output_items"],
            ),
        )

    async def _plan_turn(self, request) -> ProxyTurnPlan:
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

    async def _execute_plan(self, *, request, plan: ProxyTurnPlan, created_at: int, state: dict[str, Any]):
        if plan.mode == "delegate" and plan.agent_id is not None:
            async for event in self._execute_delegation(request=request, plan=plan, created_at=created_at, state=state):
                yield event
            return
        if plan.mode == "workflow" and plan.workflow_id is not None:
            async for event in self._execute_workflow(request=request, plan=plan, created_at=created_at, state=state):
                yield event
            return
        async for event in self._execute_direct(request=request, created_at=created_at, state=state):
            yield event

    async def _execute_continuation(
        self,
        *,
        request,
        pending: PendingContinuation,
        created_at: int,
        state: dict[str, Any],
    ):
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
        async for event in self._execute_direct(request=request, created_at=created_at, state=state, pending=pending):
            yield event

    async def _execute_direct(self, *, request, created_at: int, state: dict[str, Any], pending: PendingContinuation | None = None):
        notice = "Orchestrator: handling this turn directly.\n"
        state["reasoning"] += notice
        _record_output_item(state, "reasoning")
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
            final_response_sink=lambda value: _set_response_holder(response_holder, value),
        ):
            state["content"] += delta
            _record_output_item(state, "content")
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
        request,
        plan: ProxyTurnPlan,
        created_at: int,
        state: dict[str, Any],
        current_brief: str | None = None,
        pending: PendingContinuation | None = None,
    ):
        agent_id = plan.agent_id or "coder"
        intro = f"Orchestrator: delegating this turn to {agent_id}.\n"
        state["reasoning"] += intro
        _record_output_item(state, "reasoning")
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
            final_response_sink=lambda value: _set_response_holder(response_holder, value),
        ):
            specialist_text += delta
            reasoning_delta = f"{agent_id}: {delta}" if first else delta
            first = False
            state["reasoning"] += reasoning_delta
            _record_output_item(state, "reasoning")
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
                for event in emitted:
                    yield event
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
        state["content"] = final_text
        if final_text:
            _record_output_item(state, "content")
            yield ProxyContentDeltaEvent(final_text)

    async def _execute_workflow(self, *, request, plan: ProxyTurnPlan, created_at: int, state: dict[str, Any]):
        definition = self.registry.workflow_definitions.get(plan.workflow_id or "")
        if definition is None:
            state["finish_reason"] = "error"
            error_text = f"Unknown workflow: {plan.workflow_id or '(none)'}"
            state["content"] = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: running workflow {definition.id}.\n"
        state["reasoning"] += intro
        _record_output_item(state, "reasoning")
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
        request,
        continuation: ContinuationState,
        pending: PendingContinuation,
        created_at: int,
        state: dict[str, Any],
    ):
        definition = self.registry.workflow_definitions.get(continuation.workflow_id or "")
        if definition is None:
            state["finish_reason"] = "error"
            error_text = f"Unknown workflow: {continuation.workflow_id or '(none)'}"
            state["content"] = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: continuing workflow {definition.id} with {continuation.agent_id}.\n"
        state["reasoning"] += intro
        _record_output_item(state, "reasoning")
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
        request,
        definition,
        goal: str,
        state: dict[str, Any],
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ):
        step_groups = workflow_step_groups_for_definition(definition)
        start_index = continuation.step_index if continuation and continuation.step_index is not None else 0
        start_agent_index = continuation.agent_index if continuation and continuation.agent_index is not None else 0
        current_brief = continuation.current_brief if continuation and continuation.current_brief is not None else goal
        workflow_outputs: list[str] = list(continuation.workflow_outputs) if continuation is not None else []
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
                    pending_continuation=pending if step_index == start_index and agent_index == group_start_index else None,
                    final_response_sink=lambda value: _set_response_holder(response_holder, value),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_id}: {delta}" if first else delta
                    first = False
                    state["reasoning"] += reasoning_delta
                    _record_output_item(state, "reasoning")
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
        async for event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield event

    async def _execute_group_chat_workflow(
        self,
        *,
        request,
        definition,
        goal: str,
        state: dict[str, Any],
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ):
        participants = workflow_participants_for_definition(definition)
        sequence = workflow_selection_sequence_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(definition, default=max(len(sequence), len(participants), 1))
        if not sequence:
            sequence = tuple(participants[index % len(participants)] for index in range(max_rounds)) if participants else ()
        else:
            sequence = sequence[:max_rounds]
        start_turn = continuation.step_index if continuation and continuation.step_index is not None else 0
        current_brief = continuation.current_brief if continuation and continuation.current_brief is not None else goal
        workflow_outputs: list[str] = list(continuation.workflow_outputs) if continuation is not None else []
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
                final_response_sink=lambda value: _set_response_holder(response_holder, value),
            ):
                agent_text += delta
                reasoning_delta = f"{agent_id}: {delta}" if first else delta
                first = False
                state["reasoning"] += reasoning_delta
                _record_output_item(state, "reasoning")
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
                    for event in emitted:
                        yield event
                    return
            workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
        async for event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield event

    async def _execute_magentic_workflow(
        self,
        *,
        request,
        definition,
        goal: str,
        state: dict[str, Any],
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ):
        participants = workflow_participants_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(definition, default=max(len(participants), 1))
        current_brief = continuation.current_brief if continuation and continuation.current_brief is not None else goal
        workflow_outputs: list[str] = list(continuation.workflow_outputs) if continuation is not None else []
        round_index = continuation.step_index if continuation and continuation.step_index is not None else 0
        while round_index < max_rounds:
            if continuation is not None and round_index == (continuation.step_index or 0):
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
                pending_continuation=pending if continuation is not None and round_index == (continuation.step_index or 0) else None,
                final_response_sink=lambda value: _set_response_holder(response_holder, value),
            ):
                agent_text += delta
                reasoning_delta = f"{agent_id}: {delta}" if first else delta
                first = False
                state["reasoning"] += reasoning_delta
                _record_output_item(state, "reasoning")
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
                    for event in emitted:
                        yield event
                    return
            workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            round_index += 1
        async for event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield event

    async def _execute_handoff_workflow(
        self,
        *,
        request,
        definition,
        goal: str,
        state: dict[str, Any],
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
    ):
        participants = workflow_participants_for_definition(definition)
        finalizers = workflow_finalizers_for_definition(definition)
        handoffs = workflow_handoffs_for_definition(definition)
        max_rounds = workflow_max_rounds_for_definition(definition, default=max(len(participants), 1))
        current_brief = continuation.current_brief if continuation and continuation.current_brief is not None else goal
        workflow_outputs: list[str] = list(continuation.workflow_outputs) if continuation is not None else []
        round_index = continuation.step_index if continuation and continuation.step_index is not None else 0
        current_agent = continuation.agent_id if continuation is not None else (workflow_start_agent_for_definition(definition) or (participants[0] if participants else "reviewer"))

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
                pending_continuation=pending if continuation is not None and round_index == (continuation.step_index or 0) else None,
                final_response_sink=lambda value: _set_response_holder(response_holder, value),
            ):
                agent_text += delta
                reasoning_delta = f"{current_agent}: {delta}" if first else delta
                first = False
                state["reasoning"] += reasoning_delta
                _record_output_item(state, "reasoning")
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
                    for event in emitted:
                        yield event
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
                allowed=handoffs.get(current_agent, tuple(agent for agent in participants if agent != current_agent)),
                model_id_override=request.model,
            )
            round_index += 1
        async for event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield event

    async def _emit_workflow_summary(self, *, request, definition, goal: str, current_brief: str, workflow_outputs: tuple[str, ...], state: dict[str, Any]):
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
        state["content"] = final_text
        if final_text:
            _record_output_item(state, "content")
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
        full_prompt = _merge_preamble(preamble, prompt)
        response = await self._run_agent(
            agent_id=agent_id,
            prompt=full_prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            stream=False,
            pending_continuation=pending_continuation,
        )
        if response is None:
            return None
        response_text = getattr(response, "text", response)
        if not isinstance(response_text, str):
            return None
        return response_text.strip() or None

    async def _stream_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        preamble: str = "",
        host_tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
        pending_continuation: PendingContinuation | None = None,
        final_response_sink: Callable[[Any], None] | None = None,
    ):
        full_prompt = _merge_preamble(preamble, prompt)
        run_result = self._run_agent(
            agent_id=agent_id,
            prompt=full_prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            stream=True,
            host_tools=host_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            pending_continuation=pending_continuation,
        )
        if hasattr(run_result, "__aiter__") and hasattr(run_result, "get_final_response"):
            emitted = False
            async for update in run_result:
                delta = getattr(update, "text", "")
                if not delta:
                    continue
                emitted = True
                yield delta
            response = await run_result.get_final_response()
            if final_response_sink is not None:
                final_response_sink(response)
            final_text = getattr(response, "text", "")
            if final_text and not emitted:
                yield final_text
            return
        response = await run_result
        if final_response_sink is not None:
            final_response_sink(response)
        final_text = getattr(response, "text", "")
        if final_text:
            yield final_text

    def _run_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        stream: bool,
        host_tools=(),
        tool_choice=None,
        parallel_tool_calls=None,
        pending_continuation: PendingContinuation | None = None,
    ):
        agent = self._agent_builder(
            self.registry,
            agent_id,
            model_id_override=model_id_override,
        )
        allowed_tools, tool_options = resolve_agent_tool_policy(
            tools=tuple(host_tools),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        if allowed_tools and not provider_supports_tool_calling(self.registry, agent_id):
            if tool_options.get("tool_choice") not in (None, "auto", "none"):
                raise ValueError(f"provider for agent '{agent_id}' does not support tool calling")
            allowed_tools = ()
            tool_options.pop("tool_choice", None)
            tool_options.pop("allow_multiple_tool_calls", None)
        run_kwargs = {
            "session": agent.create_session(session_id=session_id),
            "stream": stream,
        }
        declaration_tools = build_declaration_tools(allowed_tools)
        if declaration_tools:
            run_kwargs["tools"] = declaration_tools
        run_kwargs.update(tool_options)
        result = agent.run(
            _build_agent_messages(prompt=prompt, pending_continuation=pending_continuation),
            **run_kwargs,
        )
        if stream:
            return result
        return result

    def _emit_tool_calls(
        self,
        *,
        response: Any,
        request,
        continuation: ContinuationState,
        state: dict[str, Any],
    ) -> list[ProxyToolCallEvent]:
        tool_calls = self._validated_tool_calls(
            extract_tool_calls(response),
            request=request,
        )
        if not tool_calls:
            return []
        encoded_calls = tuple(
            encode_continuation_tool_call(tool_call, state=continuation)
            for tool_call in tool_calls
        )
        state["tool_calls"] = encoded_calls
        state["finish_reason"] = "tool_calls"
        for call in encoded_calls:
            _record_output_item(state, "tool_call", call_id=call.id)
        return [ProxyToolCallEvent(call=call, index=index) for index, call in enumerate(encoded_calls)]

    def _validated_tool_calls(self, tool_calls: tuple[ProxyToolCall, ...], *, request) -> tuple[ProxyToolCall, ...]:
        if not tool_calls:
            return ()
        available_tool_names = {tool.name for tool in request.tools}
        for tool_call in tool_calls:
            if tool_call.name not in available_tool_names:
                raise ValueError(f"model requested unavailable host tool: {tool_call.name}")

        tool_choice = request.tool_choice
        if tool_choice == "none":
            raise ValueError("model requested tool calls despite tool_choice='none'")
        if isinstance(tool_choice, dict):
            required_name = tool_choice["function"]["name"]
            unexpected = [tool_call.name for tool_call in tool_calls if tool_call.name != required_name]
            if unexpected:
                raise ValueError(
                    f"model requested tool calls outside required tool '{required_name}': {', '.join(unexpected)}"
                )
        if request.parallel_tool_calls is False and len(tool_calls) > 1:
            raise ValueError("model requested multiple tool calls despite parallel_tool_calls=false")
        return tool_calls


def _merge_preamble(preamble: str, prompt: str) -> str:
    preamble = preamble.strip()
    prompt = prompt.strip()
    if preamble and prompt:
        return f"{preamble}\n\n{prompt}"
    return preamble or prompt


def _record_output_item(state: dict[str, Any], kind: str, *, call_id: str | None = None) -> None:
    current = state.get("output_items", ())
    if not isinstance(current, tuple):
        current = ()
    item = ProxyOutputItemRef(kind=kind, call_id=call_id)
    if item in current:
        return
    state["output_items"] = (*current, item)


def _direct_reply_prompt(request) -> str:
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


def _specialist_prompt(*, specialist_id: str, request_text: str, transcript_summary: str, current_brief: str | None = None) -> str:
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
            'Return {"agent_id":"<agent>" } to continue or {"agent_id":null} to finish.',
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
            'Return {"agent_id":"<agent>" } to continue or {"agent_id":null} to finish.',
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


def _parse_agent_selection(raw: str | None, *, participants: tuple[str, ...]) -> str | None:
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


def _build_agent_messages(*, prompt: str, pending_continuation: PendingContinuation | None) -> list[Message]:
    messages = [
        Message(
            role="user",
            text=prompt,
            author_name="proxy",
        )
    ]
    if pending_continuation is None:
        return messages

    assistant_contents: list[Content | str] = []
    assistant_message = pending_continuation.assistant_message
    if assistant_message is not None and assistant_message.content:
        assistant_contents.append(assistant_message.content)
    for tool_call in continuation_tool_calls(pending_continuation):
        assistant_contents.append(
            Content.from_function_call(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments_json,
            )
        )
    if assistant_contents:
        messages.append(
            Message(
                role="assistant",
                contents=assistant_contents,
                author_name=pending_continuation.state.agent_id,
            )
        )
    elif pending_continuation.assistant_message is None:
        synthetic_tool_calls = _synthetic_tool_calls_from_results(pending_continuation)
        if synthetic_tool_calls:
            messages.append(
                Message(
                    role="assistant",
                    contents=[
                        Content.from_function_call(
                            call_id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments_json,
                        )
                        for tool_call in synthetic_tool_calls
                    ],
                    author_name=pending_continuation.state.agent_id,
                )
            )

    for tool_call in continuation_tool_calls(pending_continuation):
        result_text = continuation_result_map(pending_continuation).get(tool_call.id, "")
        messages.append(
            Message(
                role="tool",
                contents=[Content.from_function_result(call_id=tool_call.id, result=result_text)],
                author_name=tool_call.name,
            )
        )
    if pending_continuation.assistant_message is None:
        synthetic_tool_calls_by_id = {
            tool_call.id: tool_call
            for tool_call in _synthetic_tool_calls_from_results(pending_continuation)
        }
        for tool_result in pending_continuation.tool_results:
            original_call_id = original_tool_call_id(tool_result.tool_call_id or "") or tool_result.tool_call_id or ""
            synthetic_tool_call = synthetic_tool_calls_by_id.get(tool_result.tool_call_id or "")
            messages.append(
                Message(
                    role="tool",
                    contents=[Content.from_function_result(call_id=original_call_id, result=tool_result.content)],
                    author_name=synthetic_tool_call.name if synthetic_tool_call is not None else "host_tool",
                )
            )
    return messages


def _synthetic_tool_calls_from_results(pending_continuation: PendingContinuation) -> list[ProxyToolCall]:
    return [
        tool_call
        for tool_result in pending_continuation.tool_results
        if (tool_call := decode_original_tool_call(tool_result.tool_call_id or "")) is not None
    ]


def _summary_instructions() -> str:
    return "\n".join(
        [
            "Summarize the completed work for the host user.",
            "Be concise and concrete.",
            "State what was decided or produced.",
            "Do not mention hidden chain-of-thought.",
        ]
    )


def _delegation_summary_prompt(*, request_text: str, specialist_id: str, specialist_text: str) -> str:
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


def _workflow_summary_prompt(*, workflow_id: str, goal: str, outputs: tuple[str, ...]) -> str:
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
