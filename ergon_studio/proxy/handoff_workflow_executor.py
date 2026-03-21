from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.delivery_requirements import (
    delivery_evidence_for_agent,
    merge_delivery_evidence,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import summarize_conversation
from ergon_studio.proxy.playbook_staffing import (
    StaffedParticipant,
    expand_staffed_participants,
    participant_by_label,
    participant_context,
    participant_for_agent,
    participant_labels_for_agents,
)
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
            workflow_participants_for_definition(definition),
            specialists=staffed_specialists,
            specialist_counts=staffed_specialist_counts,
        )
        finalizers = participant_labels_for_agents(
            participants,
            workflow_finalizers_for_definition(definition),
        )
        handoffs = _staffed_handoffs(definition, participants)
        max_rounds = workflow_max_rounds_for_definition(
            definition, default=max(len(participants), 1)
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
        round_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        if continuation is not None and pending is not None:
            current_participant = participant_by_label(
                participants,
                continuation.participant_label,
            ) or participant_for_agent(participants, continuation.agent_id)
        elif continuation is not None and continuation.agent_id is not None:
            current_participant = participant_by_label(
                participants,
                continuation.participant_label,
            ) or participant_for_agent(participants, continuation.agent_id)
            if current_participant is not None:
                next_label = await self._select_handoff_target(
                    workroom_id=definition.id,
                    current_agent=current_participant.label,
                    goal=goal,
                    current_brief=current_brief,
                    workroom_request=workroom_request,
                    prior_outputs=tuple(workroom_outputs),
                    allowed=handoffs.get(
                        current_participant.label,
                        tuple(
                            participant.label
                            for participant in participants
                            if participant.label != current_participant.label
                        ),
                    ),
                    move_rationale=(
                        loop_state.current_move_rationale
                        if loop_state is not None
                        else None
                    ),
                    model_id_override=request.model,
                )
                current_participant = participant_by_label(participants, next_label)
        else:
            start_agent = workflow_start_agent_for_definition(definition)
            if start_agent is None:
                current_participant = participants[0] if participants else None
            else:
                current_participant = participant_for_agent(participants, start_agent)
                if current_participant is None:
                    current_participant = participants[0] if participants else None

        while round_index < max_rounds and current_participant is not None:
            prompt = workflow_step_prompt(
                workroom_id=definition.id,
                agent_id=current_participant.agent_id,
                role_instance_label=(
                    current_participant.label
                    if current_participant.label != current_participant.agent_id
                    else None
                ),
                role_instance_context=participant_context(current_participant),
                goal=goal,
                current_brief=current_brief,
                workroom_request=workroom_request,
                transcript_summary=summarize_conversation(request.messages),
                prior_outputs=tuple(workroom_outputs),
                move_rationale=(
                    loop_state.current_move_rationale
                    if loop_state is not None
                    else None
                ),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=current_participant.agent_id,
                prompt=prompt,
                session_id=(
                    f"proxy-handoff-{definition.id}-"
                    f"{current_participant.label}-{uuid4().hex}"
                ),
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
                reasoning_delta = (
                    f"{current_participant.label}: {delta}" if first else delta
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
                        delivery_requirements=(
                            loop_state.delivery_requirements
                            if loop_state is not None
                            else ()
                        ),
                        delivery_evidence=(
                            loop_state.delivery_evidence
                            if loop_state is not None
                            else ()
                        ),
                        step_index=round_index,
                        agent_id=current_participant.agent_id,
                        participant_label=current_participant.label,
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
            workroom_outputs.append(
                f"{current_participant.label}: {agent_text.strip()}"
            )
            current_brief = agent_text.strip() or current_brief
            round_evidence = delivery_evidence_for_agent(current_participant.agent_id)
            if current_participant.label in finalizers:
                if result_sink is not None:
                    result_sink(
                        ProxyMoveResult(
                            worklog_lines=(workroom_outputs[-1],),
                            current_brief=current_brief,
                            delivery_evidence=round_evidence,
                        )
                    )
                    return
                break
            next_round = round_index + 1
            if result_sink is not None:
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=(workroom_outputs[-1],),
                        current_brief=current_brief,
                        delivery_evidence=round_evidence,
                        workroom_progress=ContinuationState(
                            mode="workroom",
                            workroom_id=definition.id,
                            workroom_specialists=staffed_specialists,
                            workroom_specialist_counts=staffed_specialist_counts,
                            workroom_request=workroom_request,
                            delivery_requirements=(
                                loop_state.delivery_requirements
                                if loop_state is not None
                                else ()
                            ),
                            delivery_evidence=merge_delivery_evidence(
                                (
                                    loop_state.delivery_evidence
                                    if loop_state is not None
                                    else ()
                                ),
                                round_evidence,
                            ),
                            step_index=next_round,
                            agent_id=current_participant.agent_id,
                            participant_label=current_participant.label,
                            goal=goal,
                            current_brief=current_brief,
                            decision_history=(
                                loop_state.worklog if loop_state is not None else ()
                            ),
                            workroom_outputs=tuple(workroom_outputs),
                        )
                        if next_round < max_rounds
                        else None,
                    )
                )
                return
            next_label = await self._select_handoff_target(
                workroom_id=definition.id,
                current_agent=current_participant.label,
                goal=goal,
                current_brief=current_brief,
                workroom_request=workroom_request,
                prior_outputs=tuple(workroom_outputs),
                allowed=handoffs.get(
                    current_participant.label,
                    tuple(
                        participant.label
                        for participant in participants
                        if participant.label != current_participant.label
                    ),
                ),
                move_rationale=(
                    loop_state.current_move_rationale
                    if loop_state is not None
                    else None
                ),
                model_id_override=request.model,
            )
            current_participant = participant_by_label(participants, next_label)
            round_index += 1
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
            workroom_outputs=tuple(workroom_outputs),
            state=state,
        ):
            yield summary_event


def _staffed_handoffs(
    definition: DefinitionDocument,
    participants: tuple[StaffedParticipant, ...],
) -> dict[str, tuple[str, ...]]:
    configured = workflow_handoffs_for_definition(definition)
    if not participants:
        return {}
    allowed_agents = {participant.agent_id for participant in participants}
    filtered: dict[str, tuple[str, ...]] = {}
    for current_participant in participants:
        targets = configured.get(current_participant.agent_id, ())
        if current_participant.agent_id not in allowed_agents:
            continue
        filtered_targets = tuple(
            participant.label
            for participant in participants
            if (
                participant.agent_id in targets
                and participant.label != current_participant.label
            )
        )
        if filtered_targets:
            filtered[current_participant.label] = filtered_targets
    return filtered
