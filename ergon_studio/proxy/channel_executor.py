from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.channel_staffing import (
    StaffedParticipant,
    expand_staffed_participants,
    participant_context,
)
from ergon_studio.proxy.channels import ChannelSnapshot, OpenChannel
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    pending_actors,
    pending_for_actor,
)
from ergon_studio.proxy.models import (
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.prompts import channel_message_prompt
from ergon_studio.proxy.transcript import summarize_conversation
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.response_stream import ResponseStream

ProxyEvent = ProxyReasoningDeltaEvent | ProxyToolCallEvent


class ProxyChannelExecutor:
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
        channel: OpenChannel,
        channels: dict[str, OpenChannel],
        channel_message: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        worklog: tuple[str, ...] = (),
    ) -> ResponseStream[ProxyEvent, tuple[str, ...]]:
        channel_participants = channel.participants
        staffed_members = expand_staffed_participants(channel_participants)
        persisted_worklog = (
            continuation.worklog if continuation is not None else worklog
        )
        persisted_transcript = tuple(channel.transcript)
        user_request = request.latest_user_text() or ""
        channel_lines: list[str] = []

        async def _events() -> AsyncIterator[ProxyEvent]:
            current_transcript = list(persisted_transcript)
            if pending is None and channel_message:
                current_transcript.append(f"orchestrator: {channel_message}")
            if pending is not None:
                actor_results = await asyncio.gather(
                    *[
                        self._run_channel_participant(
                            request=request,
                            channel_name=channel.name,
                            participant=_continuation_participant(
                                staffed_members=staffed_members,
                                actor=actor,
                            ),
                            user_request=user_request,
                            channel_transcript=tuple(current_transcript),
                            prior_work=persisted_worklog,
                            pending=_pending_for_actor_or_error(
                                pending=pending,
                                actor=actor,
                            ),
                        )
                        for actor in pending_actors(pending)
                    ]
                )
                emitted = _emit_participant_results(
                    actor_results=actor_results,
                    current_transcript=current_transcript,
                    channel_lines=channel_lines,
                    channel=channel,
                    channels=channels,
                    request=request,
                    state=state,
                    prior_work=persisted_worklog,
                    emit_tool_calls=self._emit_tool_calls,
                )
                for event in emitted:
                    yield event
                return

            results = await asyncio.gather(
                *[
                    self._run_channel_participant(
                        request=request,
                        channel_name=channel.name,
                        participant=participant,
                        user_request=user_request,
                        channel_transcript=tuple(current_transcript),
                        prior_work=persisted_worklog,
                    )
                    for participant in staffed_members
                ]
            )
            emitted = _emit_participant_results(
                actor_results=results,
                current_transcript=current_transcript,
                channel_lines=channel_lines,
                channel=channel,
                channels=channels,
                request=request,
                state=state,
                prior_work=persisted_worklog,
                emit_tool_calls=self._emit_tool_calls,
            )
            for event in emitted:
                yield event

        return ResponseStream(
            _events(),
            finalizer=lambda: tuple(channel_lines),
        )


    async def _run_channel_participant(
        self,
        *,
        request: ProxyTurnRequest,
        channel_name: str,
        participant: StaffedParticipant,
        user_request: str,
        channel_transcript: tuple[str, ...],
        prior_work: tuple[str, ...],
        pending: PendingContinuation | None = None,
    ) -> _ParticipantResult:
        prompt = channel_message_prompt(
            channel_name=channel_name,
            agent_id=participant.agent_id,
            role_instance_label=(
                participant.label if participant.label != participant.agent_id else None
            ),
            role_instance_context=participant_context(participant),
            user_request=user_request,
            transcript_summary=summarize_conversation(request.messages),
            channel_transcript=channel_transcript,
            prior_work=prior_work,
        )
        text = ""
        stream = self._stream_text_agent(
            agent_id=participant.agent_id,
            prompt=prompt,
            model_id_override=request.model,
            host_tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            pending_continuation=pending,
        )
        async for delta in stream:
            text += delta
        return _ParticipantResult(
            participant=participant,
            text=text,
            response=await stream.get_final_response(),
        )


class _ParticipantResult:
    def __init__(
        self,
        *,
        participant: StaffedParticipant,
        text: str,
        response: AgentRunResult,
    ) -> None:
        self.participant = participant
        self.text = text
        self.response = response


def _continuation_participant(
    *,
    staffed_members: tuple[StaffedParticipant, ...],
    actor: str,
) -> StaffedParticipant:
    for participant in staffed_members:
        if participant.label == actor:
            return participant
    return staffed_members[0]


def _snapshot_channels(
    *,
    channels: dict[str, OpenChannel],
    active_channel: OpenChannel,
    active_transcript: tuple[str, ...],
) -> tuple[ChannelSnapshot, ...]:
    snapshots = []
    for channel_id, channel in channels.items():
        if channel_id == active_channel.channel_id:
            snapshots.append(
                OpenChannel(
                    channel_id=channel.channel_id,
                    name=channel.name,
                    participants=channel.participants,
                    transcript=list(active_transcript),
                ).snapshot()
            )
            continue
        snapshots.append(channel.snapshot())
    return tuple(snapshots)


def _emit_participant_results(
    *,
    actor_results: list[_ParticipantResult] | tuple[_ParticipantResult, ...],
    current_transcript: list[str],
    channel_lines: list[str],
    channel: OpenChannel,
    channels: dict[str, OpenChannel],
    request: ProxyTurnRequest,
    state: ProxyTurnState,
    prior_work: tuple[str, ...],
    emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
) -> list[ProxyEvent]:
    events: list[ProxyEvent] = []
    tool_events: list[ProxyToolCallEvent] = []
    for result in actor_results:
        text_summary = result.text.strip()
        if not text_summary:
            continue
        line = f"{result.participant.label}: {text_summary}"
        channel_lines.append(line)
        state.append_reasoning(line)
        events.append(ProxyReasoningDeltaEvent(line))
        current_transcript.append(line)
    for result in actor_results:
        if not result.response.tool_calls:
            continue
        emitted = emit_tool_calls(
            tool_calls=result.response.tool_calls,
            request=request,
            continuation=ContinuationState(
                actor=result.participant.label,
                active_channel_id=channel.channel_id,
                channels=_snapshot_channels(
                    channels=channels,
                    active_channel=channel,
                    active_transcript=tuple(current_transcript),
                ),
                worklog=(*prior_work, *channel_lines),
            ),
            state=state,
        )
        tool_events.extend(emitted)
    if tool_events:
        tool_events = [
            ProxyToolCallEvent(call=event.call, index=index)
            for index, event in enumerate(tool_events)
        ]
        state.tool_calls = tuple(event.call for event in tool_events)
        state.finish_reason = "tool_calls"
    events.extend(tool_events)
    return events


def _pending_for_actor_or_error(
    *,
    pending: PendingContinuation,
    actor: str,
) -> PendingContinuation:
    actor_pending = pending_for_actor(pending, actor)
    if actor_pending is None:
        raise ValueError(f"missing pending tool results for actor: {actor}")
    return actor_pending
