from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.orchestrator_tools import (
    WORKROOM_INTERNAL_TOOLS,
    is_internal_tool_name,
    parse_reply_lead_dev_message,
)
from ergon_studio.proxy.prompts import workroom_round_prompt
from ergon_studio.proxy.transcript import summarize_conversation
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workroom_staffing import (
    StaffedParticipant,
    expand_staffed_participants,
    participant_context,
)
from ergon_studio.response_stream import ResponseStream

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyToolCallEvent
)


@dataclass(frozen=True)
class _AgentAttemptResult:
    participant: StaffedParticipant
    text: str
    response: AgentRunResult


class ProxyWorkroomExecutor:
    _MAX_PARTICIPANT_MOVES = 8

    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., ResponseStream[str, AgentRunResult]],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls

    def execute(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_name: str,
        participants: tuple[str, ...] = (),
        workroom_message: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        worklog: tuple[str, ...] = (),
    ) -> ResponseStream[ProxyEvent, tuple[str, ...]]:
        round_participants = _round_participants(
            participants=participants,
            continuation=continuation,
        )
        user_request = request.latest_user_text() or ""
        staffed_members = expand_staffed_participants(round_participants)
        start_index = _continuation_start_index(
            staffed_members=staffed_members,
            continuation=continuation,
        )
        persisted_worklog = (
            continuation.worklog if continuation is not None else worklog
        )
        workroom_message = (
            continuation.workroom_message
            if continuation is not None and continuation.workroom_message is not None
            else workroom_message
            if workroom_message is not None
            else None
        )
        room_lines: list[str] = []

        async def _events() -> AsyncIterator[ProxyEvent]:
            if self._should_try_parallel_round(
                staffed_members=staffed_members,
                pending=pending,
                start_index=start_index,
            ):
                parallel_results = await self._run_parallel_round(
                    request=request,
                    workroom_name=workroom_name,
                    staffed_members=staffed_members,
                    user_request=user_request,
                    workroom_message=workroom_message,
                )
                if any(result.response.tool_calls for result in parallel_results):
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
                        room_lines.append(reasoning_delta)
                    return

            for member_index in range(start_index, len(staffed_members)):
                participant = staffed_members[member_index]
                participant_pending = pending if member_index == start_index else None
                participant_move_count = 0
                while True:
                    prompt = workroom_round_prompt(
                        workroom_name=workroom_name,
                        agent_id=participant.agent_id,
                        role_instance_label=(
                            participant.label
                            if participant.label != participant.agent_id
                            else None
                        ),
                        role_instance_context=participant_context(participant),
                        user_request=user_request,
                        workroom_message=workroom_message,
                        transcript_summary=summarize_conversation(request.messages),
                        prior_work=_prior_work(
                            worklog=persisted_worklog,
                            room_lines=room_lines,
                        ),
                    )
                    agent_text = ""
                    first = True
                    stream = self._stream_text_agent(
                        agent_id=participant.agent_id,
                        prompt=prompt,
                        model_id_override=request.model,
                        host_tools=request.tools,
                        extra_tools=WORKROOM_INTERNAL_TOOLS,
                        tool_choice=request.tool_choice,
                        parallel_tool_calls=request.parallel_tool_calls,
                        pending_continuation=participant_pending,
                    )
                    async for delta in stream:
                        agent_text += delta
                        reasoning_delta = (
                            f"{participant.label}: {delta}" if first else delta
                        )
                        first = False
                        state.append_reasoning(reasoning_delta)
                        yield ProxyReasoningDeltaEvent(reasoning_delta)
                    participant_pending = None
                    response = await stream.get_final_response()
                    internal_tool_calls = tuple(
                        tool_call
                        for tool_call in response.tool_calls
                        if is_internal_tool_name(tool_call.name)
                    )
                    host_tool_calls = tuple(
                        tool_call
                        for tool_call in response.tool_calls
                        if not is_internal_tool_name(tool_call.name)
                    )
                    if internal_tool_calls and host_tool_calls:
                        raise ValueError(
                            "workroom participants cannot mix internal actions with "
                            "host tool calls"
                        )
                    if len(internal_tool_calls) > 1:
                        raise ValueError(
                            "workroom participants must use at most one internal "
                            "action at a time"
                        )
                    if host_tool_calls:
                        emitted = self._emit_tool_calls(
                            tool_calls=host_tool_calls,
                            request=request,
                            continuation=ContinuationState(
                                workroom_name=workroom_name,
                                workroom_participants=round_participants,
                                workroom_message=workroom_message,
                                actor=participant.label,
                                worklog=(*persisted_worklog, *room_lines),
                            ),
                            state=state,
                        )
                        if emitted:
                            for event in emitted:
                                yield event
                            return
                    if internal_tool_calls:
                        message = parse_reply_lead_dev_message(
                            internal_tool_calls[0]
                        )
                        reasoning_delta = f"{participant.label}: {message}"
                        state.append_reasoning(reasoning_delta)
                        yield ProxyReasoningDeltaEvent(reasoning_delta)
                        room_lines.append(reasoning_delta)
                        break
                    text_summary = agent_text.strip()
                    if text_summary:
                        room_lines.append(f"{participant.label}: {text_summary}")
                    if (
                        len(staffed_members) == 1
                        and request.tools
                        and participant_move_count + 1 < self._MAX_PARTICIPANT_MOVES
                    ):
                        participant_move_count += 1
                        continue
                    break

        return ResponseStream(
            _events(),
            finalizer=lambda: tuple(room_lines),
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
        workroom_name: str,
        staffed_members: tuple[StaffedParticipant, ...],
        user_request: str,
        workroom_message: str | None,
    ) -> list[_AgentAttemptResult]:
        tasks = [
            asyncio.create_task(
                self._run_round_participant(
                    request=request,
                    workroom_name=workroom_name,
                    participant=participant,
                    user_request=user_request,
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
        workroom_name: str,
        participant: StaffedParticipant,
        user_request: str,
        workroom_message: str | None,
    ) -> _AgentAttemptResult:
        prompt = workroom_round_prompt(
            workroom_name=workroom_name,
            agent_id=participant.agent_id,
            role_instance_label=(
                participant.label
                if participant.label != participant.agent_id
                else None
            ),
            role_instance_context=participant_context(participant),
            user_request=user_request,
            workroom_message=workroom_message,
            transcript_summary=summarize_conversation(request.messages),
            prior_work=(),
        )
        text = ""
        stream = self._stream_text_agent(
            agent_id=participant.agent_id,
            prompt=prompt,
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
        )
        async for delta in stream:
            text += delta
        return _AgentAttemptResult(
            participant=participant,
            text=text.strip(),
            response=await stream.get_final_response(),
        )


def _round_participants(
    *,
    participants: tuple[str, ...],
    continuation: ContinuationState | None,
) -> tuple[str, ...]:
    if continuation is not None and continuation.workroom_participants:
        return continuation.workroom_participants
    return participants


def _continuation_start_index(
    *,
    staffed_members: tuple[StaffedParticipant, ...],
    continuation: ContinuationState | None,
) -> int:
    if continuation is None:
        return 0
    for index, participant in enumerate(staffed_members):
        if participant.label == continuation.actor:
            return index
    return 0

def _prior_work(
    *,
    worklog: tuple[str, ...],
    room_lines: list[str],
) -> tuple[str, ...]:
    prior_work = [
        *worklog,
        *room_lines,
    ]
    if not prior_work:
        return ()
    return tuple(prior_work[-6:])




def _is_parallel_round(staffed_members: tuple[StaffedParticipant, ...]) -> bool:
    if len(staffed_members) <= 1:
        return False
    agent_ids = {participant.agent_id for participant in staffed_members}
    return len(agent_ids) == 1
