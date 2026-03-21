from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.delivery_requirements import delivery_evidence_for_agent
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import ProxyTurnPlan, summarize_conversation
from ergon_studio.proxy.prompts import (
    delegation_summary_prompt,
    direct_reply_prompt,
    finish_reply_prompt,
    specialist_prompt,
    summary_instructions,
)
from ergon_studio.proxy.response_sink import response_holder_sink
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
ProxyToolChoice = str | dict[str, object] | None


class ProxyTurnExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        run_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._run_text_agent = run_text_agent
        self._emit_tool_calls = emit_tool_calls

    async def execute_direct(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        pending: PendingContinuation | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        notice = _decision_notice(
            base="Orchestrator: handling this turn directly.",
            loop_state=loop_state,
        )
        state.append_reasoning(notice)
        yield ProxyReasoningDeltaEvent(notice)
        prompt = direct_reply_prompt(
            request,
            goal=loop_state.goal if loop_state is not None else None,
            current_brief=loop_state.current_brief if loop_state is not None else None,
            worklog=loop_state.worklog if loop_state is not None else (),
            move_rationale=(
                loop_state.current_move_rationale if loop_state is not None else None
            ),
            success_criteria=(
                loop_state.current_move_success_criteria
                if loop_state is not None
                else None
            ),
        )
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
            final_response_sink=response_holder_sink(response_holder),
        ):
            state.append_content(delta)
            yield ProxyContentDeltaEvent(delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                request=request,
                continuation=ContinuationState(
                    mode="act",
                    agent_id="orchestrator",
                    delivery_requirements=(
                        loop_state.delivery_requirements
                        if loop_state is not None
                        else ()
                    ),
                    delivery_evidence=(
                        loop_state.delivery_evidence if loop_state is not None else ()
                    ),
                    goal=loop_state.goal if loop_state is not None else None,
                    current_brief=(
                        loop_state.current_brief if loop_state is not None else None
                    ),
                    decision_history=(
                        loop_state.worklog if loop_state is not None else ()
                    ),
                ),
                state=state,
            )
            for event in emitted:
                yield event

    async def execute_finish(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        notice = _decision_notice(
            base="Orchestrator: delivering the current result.",
            loop_state=loop_state,
        )
        state.append_reasoning(notice)
        yield ProxyReasoningDeltaEvent(notice)
        prompt = finish_reply_prompt(
            request,
            goal=loop_state.goal if loop_state is not None else None,
            current_brief=loop_state.current_brief if loop_state is not None else None,
            worklog=loop_state.worklog if loop_state is not None else (),
            delivery_requirements=(
                loop_state.delivery_requirements if loop_state is not None else ()
            ),
            delivery_evidence=(
                loop_state.delivery_evidence if loop_state is not None else ()
            ),
            move_rationale=(
                loop_state.current_move_rationale if loop_state is not None else None
            ),
            success_criteria=(
                loop_state.current_move_success_criteria
                if loop_state is not None
                else None
            ),
        )
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id="orchestrator",
            prompt=prompt,
            session_id=f"proxy-finish-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            final_response_sink=response_holder_sink(response_holder),
        ):
            state.append_content(delta)
            yield ProxyContentDeltaEvent(delta)
        response = response_holder.get("response")
        if response is not None:
            emitted = self._emit_tool_calls(
                response=response,
                request=request,
                continuation=ContinuationState(
                    mode="finish",
                    agent_id="orchestrator",
                    delivery_requirements=(
                        loop_state.delivery_requirements
                        if loop_state is not None
                        else ()
                    ),
                    delivery_evidence=(
                        loop_state.delivery_evidence if loop_state is not None else ()
                    ),
                    goal=loop_state.goal if loop_state is not None else None,
                    current_brief=(
                        loop_state.current_brief if loop_state is not None else None
                    ),
                    decision_history=(
                        loop_state.worklog if loop_state is not None else ()
                    ),
                ),
                state=state,
            )
            for event in emitted:
                yield event

    async def execute_delegation(
        self,
        *,
        request: ProxyTurnRequest,
        plan: ProxyTurnPlan,
        state: ProxyTurnState,
        current_brief: str | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        agent_id = plan.agent_id or "coder"
        intro = _decision_notice(
            base=f"Orchestrator: delegating this turn to {agent_id}.",
            loop_state=loop_state,
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        prompt_text = specialist_prompt(
            specialist_id=agent_id,
            request_text=plan.request or request.latest_user_text() or "",
            transcript_summary=summarize_conversation(request.messages),
            current_brief=current_brief,
            move_rationale=plan.rationale,
            success_criteria=plan.success_criteria,
        )
        specialist_text = ""
        first = True
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id=agent_id,
            prompt=prompt_text,
            session_id=f"proxy-delegate-{agent_id}-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            pending_continuation=pending,
            final_response_sink=response_holder_sink(response_holder),
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
                    delivery_requirements=(
                        loop_state.delivery_requirements
                        if loop_state is not None
                        else ()
                    ),
                    delivery_evidence=(
                        loop_state.delivery_evidence if loop_state is not None else ()
                    ),
                    request_text=plan.request or request.latest_user_text(),
                    current_brief=specialist_text.strip() or current_brief,
                    goal=(
                        plan.goal
                        or (loop_state.goal if loop_state is not None else None)
                    ),
                    decision_history=(
                        loop_state.worklog if loop_state is not None else ()
                    ),
                ),
                state=state,
            )
            if emitted:
                for tool_event in emitted:
                    yield tool_event
                return
        final_text = specialist_text.strip() or current_brief or "(no output)"
        if result_sink is not None:
            result_sink(
                ProxyMoveResult(
                    worklog_lines=(f"{agent_id}: {final_text}",),
                    current_brief=final_text,
                    delivery_evidence=delivery_evidence_for_agent(agent_id),
                )
            )
            return
        summary_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=delegation_summary_prompt(
                request_text=request.latest_user_text() or "",
                specialist_id=agent_id,
                specialist_text=specialist_text,
            ),
            preamble=summary_instructions(),
            session_id=f"proxy-delegation-summary-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not summary_text:
            summary_text = final_text
        state.set_content(summary_text)
        if summary_text:
            yield ProxyContentDeltaEvent(summary_text)


def _decision_notice(
    *,
    base: str,
    loop_state: ProxyDecisionLoopState | None,
) -> str:
    lines = [base]
    if loop_state is not None and loop_state.current_move_rationale:
        lines.append(f"Why: {loop_state.current_move_rationale}")
    if loop_state is not None and loop_state.current_move_success_criteria:
        lines.append(f"Success target: {loop_state.current_move_success_criteria}")
    return "\n".join(lines) + "\n"
