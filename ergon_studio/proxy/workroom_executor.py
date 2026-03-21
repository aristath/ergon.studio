from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial
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
from ergon_studio.proxy.prompts import workroom_round_prompt
from ergon_studio.proxy.tool_passthrough import extract_tool_calls
from ergon_studio.proxy.transcript import summarize_conversation
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workroom_staffing import (
    StaffedParticipant,
    expand_staffed_participants,
    participant_context,
)
from ergon_studio.workroom_layout import workroom_participants_for_definition

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


@dataclass(frozen=True)
class _AgentAttemptResult:
    participant: StaffedParticipant
    text: str
    response: Any


class ProxyWorkroomExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        participants: tuple[str, ...] = (),
        workroom_message: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None],
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        round_participants = _round_participants(
            definition=definition,
            participants=participants,
            continuation=continuation,
        )
        staffed_members = expand_staffed_participants(round_participants)
        start_index = (
            continuation.member_index
            if continuation and continuation.member_index is not None
            else 0
        )
        workroom_message = (
            continuation.workroom_message
            if continuation is not None and continuation.workroom_message is not None
            else workroom_message
            if workroom_message is not None
            else None
        )
        workroom_outputs: list[str] = (
            list(continuation.workroom_outputs) if continuation is not None else []
        )
        round_outputs: list[str] = []

        if self._should_try_parallel_round(
            staffed_members=staffed_members,
            pending=pending,
            start_index=start_index,
        ):
            parallel_results = await self._run_parallel_round(
                request=request,
                definition=definition,
                staffed_members=staffed_members,
                goal=goal,
                workroom_message=workroom_message,
            )
            if any(
                extract_tool_calls(result.response)
                for result in parallel_results
                if result.response is not None
            ):
                fallback_notice = (
                    "Orchestrator: parallel room round requested tool use; "
                    "rerunning this staffed group sequentially for safe "
                    "continuation.\n"
                )
                state.append_reasoning(fallback_notice)
                yield ProxyReasoningDeltaEvent(fallback_notice)
            else:
                for result in parallel_results:
                    reasoning_delta = f"{result.participant.label}: {result.text}"
                    state.append_reasoning(reasoning_delta)
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                    round_outputs.append(reasoning_delta)
                workroom_outputs.extend(round_outputs)
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=tuple(round_outputs),
                        active_workroom=_active_workroom_state(
                            definition=definition,
                            round_participants=round_participants,
                            workroom_message=workroom_message,
                            goal=goal,
                            loop_state=loop_state,
                            workroom_outputs=workroom_outputs,
                            staffed_members=staffed_members,
                        ),
                    )
                )
                return

        for member_index in range(start_index, len(staffed_members)):
            participant = staffed_members[member_index]
            prompt = workroom_round_prompt(
                workroom_id=definition.id,
                agent_id=participant.agent_id,
                role_instance_label=(
                    participant.label
                    if participant.label != participant.agent_id
                    else None
                ),
                role_instance_context=participant_context(participant),
                goal=goal,
                workroom_message=workroom_message,
                transcript_summary=summarize_conversation(request.messages),
                prior_outputs=tuple(workroom_outputs),
            )
            agent_text = ""
            first = True
            response_holder: dict[str, Any] = {}
            async for delta in self._stream_text_agent(
                agent_id=participant.agent_id,
                prompt=prompt,
                session_id=(
                    f"proxy-workroom-{definition.id}-{participant.label}-{uuid4().hex}"
                ),
                model_id_override=request.model,
                host_tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending if member_index == start_index else None,
                final_response_sink=partial(_store_response, response_holder),
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
                        mode="workroom",
                        workroom_id=definition.id,
                        workroom_participants=round_participants,
                        workroom_message=workroom_message,
                        member_index=member_index,
                        agent_id=participant.agent_id,
                        participant_label=participant.label,
                        goal=goal,
                        worklog=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
                        workroom_outputs=tuple(workroom_outputs),
                    ),
                    state=state,
                )
                if emitted:
                    for event in emitted:
                        yield event
                    return
            round_output = f"{participant.label}: {agent_text.strip()}"
            round_outputs.append(round_output)
            workroom_outputs.append(round_output)
        result_sink(
            ProxyMoveResult(
                worklog_lines=tuple(round_outputs),
                active_workroom=_active_workroom_state(
                    definition=definition,
                    round_participants=round_participants,
                    workroom_message=workroom_message,
                    goal=goal,
                    loop_state=loop_state,
                    workroom_outputs=workroom_outputs,
                    staffed_members=staffed_members,
                ),
            )
        )

    def _should_try_parallel_round(
        self,
        *,
        staffed_members: tuple[StaffedParticipant, ...],
        pending: PendingContinuation | None,
        start_index: int,
    ) -> bool:
        return (
            _is_parallel_round(staffed_members)
            and pending is None
            and start_index == 0
        )

    async def _run_parallel_round(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        staffed_members: tuple[StaffedParticipant, ...],
        goal: str,
        workroom_message: str | None,
    ) -> list[_AgentAttemptResult]:
        tasks = [
            asyncio.create_task(
                self._run_round_participant(
                    request=request,
                    definition=definition,
                    participant=participant,
                    goal=goal,
                    workroom_message=workroom_message,
                )
            )
            for participant in staffed_members
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_round_participant(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        participant: StaffedParticipant,
        goal: str,
        workroom_message: str | None,
    ) -> _AgentAttemptResult:
        prompt = workroom_round_prompt(
            workroom_id=definition.id,
            agent_id=participant.agent_id,
            role_instance_label=(
                participant.label
                if participant.label != participant.agent_id
                else None
            ),
            role_instance_context=participant_context(participant),
            goal=goal,
            workroom_message=workroom_message,
            transcript_summary=summarize_conversation(request.messages),
            prior_outputs=(),
        )
        text = ""
        response_holder: dict[str, Any] = {}
        async for delta in self._stream_text_agent(
            agent_id=participant.agent_id,
            prompt=prompt,
            session_id=f"proxy-workroom-{definition.id}-{participant.label}-{uuid4().hex}",
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            final_response_sink=partial(_store_response, response_holder),
        ):
            text += delta
        return _AgentAttemptResult(
            participant=participant,
            text=text.strip(),
            response=response_holder.get("response"),
        )


def _round_participants(
    *,
    definition: DefinitionDocument,
    participants: tuple[str, ...],
    continuation: ContinuationState | None,
) -> tuple[str, ...]:
    if continuation is not None and continuation.workroom_participants:
        return continuation.workroom_participants
    if participants:
        return participants
    return workroom_participants_for_definition(definition)


def _active_workroom_state(
    *,
    definition: DefinitionDocument,
    round_participants: tuple[str, ...],
    workroom_message: str | None,
    goal: str,
    loop_state: ProxyDecisionLoopState | None,
    workroom_outputs: list[str],
    staffed_members: tuple[StaffedParticipant, ...],
) -> ContinuationState | None:
    if not staffed_members:
        return None
    return ContinuationState(
        mode="workroom",
        agent_id=staffed_members[0].agent_id,
        participant_label=staffed_members[0].label,
        workroom_id=definition.id,
        workroom_participants=round_participants,
        workroom_message=workroom_message,
        goal=goal,
        worklog=loop_state.worklog if loop_state is not None else (),
        workroom_outputs=tuple(workroom_outputs),
    )
def _is_parallel_round(staffed_members: tuple[StaffedParticipant, ...]) -> bool:
    if len(staffed_members) <= 1:
        return False
    agent_ids = {participant.agent_id for participant in staffed_members}
    return len(agent_ids) == 1


def _store_response(holder: dict[str, Any], value: object) -> None:
    holder["response"] = value
