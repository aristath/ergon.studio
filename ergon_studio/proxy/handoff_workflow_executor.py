from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import summarize_conversation
from ergon_studio.proxy.prompts import workflow_step_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_metadata import (
    workflow_finalizers_for_definition,
    workflow_handoffs_for_definition,
    workflow_max_rounds_for_definition,
    workflow_participants_for_definition,
    workflow_start_agent_for_definition,
)

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyHandoffWorkflowExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workflow_summary: Callable[..., AsyncIterator[ProxyEvent]],
        select_handoff_target: Callable[..., Any],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workflow_summary = emit_workflow_summary
        self._select_handoff_target = select_handoff_target

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
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
            prompt = workflow_step_prompt(
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
                final_response_sink=response_holder_sink(response_holder),
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
                        decision_history=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
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
        if result_sink is not None:
            result_sink(
                ProxyMoveResult(
                    worklog_lines=tuple(workflow_outputs),
                    current_brief=current_brief,
                )
            )
            return
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield summary_event
