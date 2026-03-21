from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.prompts import specialist_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.transcript import summarize_conversation
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


class ProxyTurnExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls

    async def execute_delegation(
        self,
        *,
        request: ProxyTurnRequest,
        agent_id: str,
        message: str,
        state: ProxyTurnState,
        current_brief: str | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        intro = f"Orchestrator: messaging specialist {agent_id}."
        state.append_reasoning(intro + "\n")
        yield ProxyReasoningDeltaEvent(intro + "\n")
        effective_brief = current_brief
        if effective_brief is None and loop_state is not None:
            effective_brief = loop_state.current_brief
        prompt_text = specialist_prompt(
            specialist_id=agent_id,
            message=message,
            transcript_summary=summarize_conversation(request.messages),
            current_brief=effective_brief,
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
                    message=message,
                    current_brief=specialist_text.strip() or effective_brief,
                    goal=loop_state.goal if loop_state is not None else None,
                    worklog=(
                        loop_state.worklog if loop_state is not None else ()
                    ),
                ),
                state=state,
            )
            if emitted:
                for tool_event in emitted:
                    yield tool_event
                return
        final_text = specialist_text.strip() or effective_brief or "(no output)"
        if result_sink is not None:
            result_sink(
                ProxyMoveResult(
                    worklog_lines=(f"{agent_id}: {final_text}",),
                    current_brief=final_text,
                )
            )
