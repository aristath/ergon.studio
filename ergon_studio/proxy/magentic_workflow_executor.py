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
    expand_staffed_participants,
    participant_by_label,
    participant_context,
    participant_for_agent,
)
from ergon_studio.proxy.prompts import workflow_step_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_metadata import (
    workflow_max_rounds_for_definition,
    workflow_participants_for_definition,
)

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyMagenticWorkflowExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workflow_summary: Callable[..., AsyncIterator[ProxyEvent]],
        select_manager_agent: Callable[..., Any],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workflow_summary = emit_workflow_summary
        self._select_manager_agent = select_manager_agent

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workflow_request: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        staffed_specialists = (
            continuation.workflow_specialists
            if continuation is not None
            else specialists
        )
        staffed_specialist_counts = (
            continuation.workflow_specialist_counts
            if continuation is not None
            else specialist_counts
        )
        participants = expand_staffed_participants(
            workflow_participants_for_definition(definition),
            specialists=staffed_specialists,
            specialist_counts=staffed_specialist_counts,
        )
        max_rounds = workflow_max_rounds_for_definition(
            definition, default=max(len(participants), 1)
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_request = (
            continuation.workflow_request
            if continuation is not None and continuation.workflow_request is not None
            else workflow_request
            if workflow_request is not None
            else (
                loop_state.current_playbook_request
                if loop_state is not None
                else None
            )
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
            participant = None
            if (
                continuation is not None
                and pending is not None
                and round_index == (continuation.step_index or 0)
            ):
                participant = participant_by_label(
                    participants,
                    continuation.participant_label,
                ) or participant_for_agent(participants, continuation.agent_id)
            else:
                participant_label = await self._select_manager_agent(
                    workflow_id=definition.id,
                    goal=goal,
                    current_brief=current_brief,
                    playbook_request=workflow_request,
                    participants=tuple(
                        participant.label for participant in participants
                    ),
                    prior_outputs=tuple(workflow_outputs),
                    move_rationale=(
                        loop_state.current_move_rationale
                        if loop_state is not None
                        else None
                    ),
                    model_id_override=request.model,
                )
                participant = participant_by_label(participants, participant_label)
            if participant is None:
                break
            prompt = workflow_step_prompt(
                workflow_id=definition.id,
                agent_id=participant.agent_id,
                role_instance_label=(
                    participant.label
                    if participant.label != participant.agent_id
                    else None
                ),
                role_instance_context=participant_context(participant),
                goal=goal,
                current_brief=current_brief,
                playbook_request=workflow_request,
                transcript_summary=summarize_conversation(request.messages),
                prior_outputs=tuple(workflow_outputs),
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
                agent_id=participant.agent_id,
                prompt=prompt,
                session_id=(
                    f"proxy-magentic-{definition.id}-{participant.label}-{uuid4().hex}"
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
                reasoning_delta = f"{participant.label}: {delta}" if first else delta
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
                        workflow_specialists=staffed_specialists,
                        workflow_specialist_counts=staffed_specialist_counts,
                        workflow_request=workflow_request,
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
                        agent_id=participant.agent_id,
                        participant_label=participant.label,
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
            workflow_outputs.append(f"{participant.label}: {agent_text.strip()}")
            current_brief = agent_text.strip() or current_brief
            round_index += 1
            if result_sink is not None:
                round_evidence = delivery_evidence_for_agent(participant.agent_id)
                workflow_progress = None
                if round_index < max_rounds:
                    workflow_progress = ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        workflow_specialists=staffed_specialists,
                        workflow_specialist_counts=staffed_specialist_counts,
                        workflow_request=workflow_request,
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
                        step_index=round_index,
                        agent_id=participant.agent_id,
                        participant_label=participant.label,
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
                        delivery_evidence=round_evidence,
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
