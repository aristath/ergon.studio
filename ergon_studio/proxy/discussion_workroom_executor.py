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
from ergon_studio.proxy.playbook_staffing import (
    expand_staffed_participants,
    expand_staffed_sequence,
    participant_by_label,
    participant_context,
)
from ergon_studio.proxy.prompts import discussion_turn_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.transcript import summarize_conversation
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workroom_metadata import (
    workroom_max_rounds_for_definition,
    workroom_participants_for_definition,
    workroom_turn_sequence_for_definition,
)

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyDiscussionWorkroomExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workroom_summary: Callable[..., AsyncIterator[ProxyEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workroom_summary = emit_workroom_summary

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workroom_request: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        staffed_specialists = (
            continuation.workroom_specialists
            if continuation is not None
            else specialists
        )
        staffed_specialist_counts = (
            continuation.workroom_specialist_counts
            if continuation is not None
            else specialist_counts
        )
        participants = expand_staffed_participants(
            workroom_participants_for_definition(definition),
            specialists=staffed_specialists,
            specialist_counts=staffed_specialist_counts,
        )
        sequence = expand_staffed_sequence(
            workroom_turn_sequence_for_definition(definition),
            participants=participants,
        )
        max_rounds = workroom_max_rounds_for_definition(
            definition, default=max(len(sequence), len(participants), 1)
        )
        if not sequence:
            sequence = (
                tuple(
                    participants[index % len(participants)].label
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
        workroom_request = (
            continuation.workroom_request
            if continuation is not None and continuation.workroom_request is not None
            else workroom_request
            if workroom_request is not None
            else (
                loop_state.current_workroom_request
                if loop_state is not None
                else None
            )
        )
        workroom_outputs: list[str] = (
            list(continuation.workroom_outputs) if continuation is not None else []
        )
        for turn_index in range(start_turn, len(sequence)):
            participant = participant_by_label(participants, sequence[turn_index])
            if participant is None:
                continue
            prompt = discussion_turn_prompt(
                workroom_id=definition.id,
                agent_id=participant.agent_id,
                role_instance_label=(
                    participant.label
                    if participant.label != participant.agent_id
                    else None
                ),
                role_instance_context=participant_context(participant),
                goal=goal,
                transcript_summary=summarize_conversation(request.messages),
                current_brief=current_brief,
                workroom_request=workroom_request,
                prior_outputs=tuple(workroom_outputs),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=participant.agent_id,
                prompt=prompt,
                session_id=(
                    f"proxy-group-chat-{definition.id}-{participant.label}-{uuid4().hex}"
                ),
                model_id_override=request.model,
                host_tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending if turn_index == start_turn else None,
                final_response_sink=response_holder_sink(response_holder),
            ):
                agent_text += delta
                reasoning_delta = (
                    f"{participant.label}: {delta}" if first else delta
                )
                first = False
                state.append_reasoning(reasoning_delta)
                yield ProxyReasoningDeltaEvent(reasoning_delta)
            response = response_holder.get("response")
            if response is not None:
                emitted = self._emit_tool_calls(
                    response=response,
                    request=request,
                    continuation=ContinuationState(
                        mode="workroom",
                        workroom_id=definition.id,
                        workroom_specialists=staffed_specialists,
                        workroom_specialist_counts=staffed_specialist_counts,
                        workroom_request=workroom_request,
                        step_index=turn_index,
                        agent_id=participant.agent_id,
                        participant_label=participant.label,
                        goal=goal,
                        current_brief=agent_text.strip() or current_brief,
                        decision_history=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
                        workroom_outputs=tuple(workroom_outputs),
                    ),
                    state=state,
                )
                if emitted:
                    for tool_event in emitted:
                        yield tool_event
                    return
            workroom_outputs.append(f"{participant.label}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            if result_sink is not None:
                next_turn = turn_index + 1
                workroom_progress = None
                if next_turn < len(sequence):
                    next_participant = participant_by_label(
                        participants,
                        sequence[next_turn],
                    )
                    workroom_progress = ContinuationState(
                        mode="workroom",
                        workroom_id=definition.id,
                        workroom_specialists=staffed_specialists,
                        workroom_specialist_counts=staffed_specialist_counts,
                        workroom_request=workroom_request,
                        step_index=next_turn,
                        agent_id=(
                            next_participant.agent_id
                            if next_participant is not None
                            else participant.agent_id
                        ),
                        participant_label=(
                            next_participant.label
                            if next_participant is not None
                            else None
                        ),
                        goal=goal,
                        current_brief=current_brief,
                        decision_history=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
                        workroom_outputs=tuple(workroom_outputs),
                    )
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=(workroom_outputs[-1],),
                        current_brief=current_brief,
                        workroom_progress=workroom_progress,
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
        async for summary_event in self._emit_workroom_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workroom_outputs=tuple(workroom_outputs),
            state=state,
        ):
            yield summary_event
