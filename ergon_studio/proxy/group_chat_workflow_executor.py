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
from ergon_studio.proxy.prompts import group_chat_turn_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_metadata import (
    workflow_max_rounds_for_definition,
    workflow_participants_for_definition,
    workflow_selection_sequence_for_definition,
)

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyGroupChatWorkflowExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workflow_summary: Callable[..., AsyncIterator[ProxyEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workflow_summary = emit_workflow_summary

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
            prompt = group_chat_turn_prompt(
                workflow_id=definition.id,
                agent_id=agent_id,
                goal=goal,
                transcript_summary=summarize_conversation(request.messages),
                current_brief=current_brief,
                prior_outputs=tuple(workflow_outputs),
                move_rationale=(
                    loop_state.current_move_rationale
                    if loop_state is not None
                    else None
                ),
                success_criteria=(
                    loop_state.current_move_success_criteria
                    if loop_state is not None
                    else None
                ),
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
                final_response_sink=response_holder_sink(response_holder),
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
            workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            if result_sink is not None:
                next_turn = turn_index + 1
                workflow_progress = None
                if next_turn < len(sequence):
                    workflow_progress = ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        step_index=next_turn,
                        agent_id=agent_id,
                        goal=goal,
                        current_brief=current_brief,
                        decision_history=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
                        workflow_outputs=tuple(workflow_outputs),
                    )
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=(workflow_outputs[-1],),
                        current_brief=current_brief,
                        workflow_progress=workflow_progress,
                    )
                )
                return
        if result_sink is not None:
            result_sink(
                ProxyMoveResult(
                    worklog_lines=(),
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
