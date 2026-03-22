from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.channel_staffing import (
    StaffedParticipant,
    expand_staffed_participants,
    participant_context,
    require_staffed_recipients,
)
from ergon_studio.proxy.channels import Channel, ChannelMessage
from ergon_studio.proxy.continuation import (
    PendingContinuation,
    PendingToolContext,
)
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.orchestrator_tools import (
    PARTICIPANT_INTERNAL_TOOLS,
    is_internal_tool_name,
    parse_message_channel_action,
)
from ergon_studio.proxy.prompts import channel_message_prompt
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.response_stream import ResponseStream

ProxyEvent = ProxyReasoningDeltaEvent | ProxyToolCallEvent


@dataclass(frozen=True)
class _ChannelDelivery:
    author: str
    message: str
    recipients: tuple[str, ...]


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
        session_id: str,
        channel: Channel,
        channel_message: str | None = None,
        recipients: tuple[str, ...] = (),
        state: ProxyTurnState,
        pending: PendingContinuation | None = None,
    ) -> ResponseStream[ProxyEvent, tuple[ChannelMessage, ...]]:
        all_staffed_members = expand_staffed_participants(channel.participants)
        channel_messages: list[ChannelMessage] = []

        async def _events() -> AsyncIterator[ProxyEvent]:
            current_transcript = list(channel.transcript)
            deliveries: deque[_ChannelDelivery] = deque()
            pending_tool_events: list[ProxyToolCallEvent] = []

            if channel_message:
                deliveries.append(
                    _ChannelDelivery(
                        author="orchestrator",
                        message=channel_message,
                        recipients=recipients,
                    )
                )

            if pending is not None:
                for actor in (item.actor for item in pending):
                    participant = next(
                        (
                            participant
                            for participant in all_staffed_members
                            if participant.label == actor
                        ),
                        None,
                    )
                    if participant is None:
                        raise ValueError(
                            f"pending actor is not staffed in this channel: {actor}"
                        )
                    actor_pending = next(
                        (item for item in pending if item.actor == actor),
                        None,
                    )
                    if actor_pending is None:
                        raise ValueError(
                            f"missing pending tool results for actor: {actor}"
                        )
                    result = await self._run_channel_participant(
                        request=request,
                        channel_name=channel.name,
                        participant=participant,
                        channel_transcript=tuple(current_transcript),
                        pending=actor_pending,
                    )
                    emitted, new_deliveries, new_tool_events = (
                        _process_participant_results(
                            participant=participant,
                            response=result,
                            request=request,
                            channel=channel,
                            current_transcript=current_transcript,
                            channel_messages=channel_messages,
                            state=state,
                            emit_tool_calls=self._emit_tool_calls,
                            session_id=session_id,
                        )
                    )
                    for reasoning_event in emitted:
                        yield reasoning_event
                    deliveries.extend(new_deliveries)
                    pending_tool_events.extend(new_tool_events)

            while deliveries:
                delivery = deliveries.popleft()
                delivery_line = ChannelMessage(
                    author=delivery.author,
                    content=delivery.message,
                )
                current_transcript.append(delivery_line)
                if delivery.author != "orchestrator":
                    channel_messages.append(delivery_line)
                    state.append_reasoning(delivery_line.render())
                    yield ProxyReasoningDeltaEvent(delivery_line.render())

                targets = require_staffed_recipients(
                    staffed_members=all_staffed_members,
                    recipients=delivery.recipients,
                )

                for participant in targets:
                    result = await self._run_channel_participant(
                        request=request,
                        channel_name=channel.name,
                        participant=participant,
                        channel_transcript=tuple(current_transcript),
                    )
                    emitted, new_deliveries, new_tool_events = (
                        _process_participant_results(
                            participant=participant,
                            response=result,
                            request=request,
                            channel=channel,
                            current_transcript=current_transcript,
                            channel_messages=channel_messages,
                            state=state,
                            emit_tool_calls=self._emit_tool_calls,
                            session_id=session_id,
                        )
                    )
                    for reasoning_event in emitted:
                        yield reasoning_event
                    deliveries.extend(new_deliveries)
                    pending_tool_events.extend(new_tool_events)

            if pending_tool_events:
                ordered = [
                    ProxyToolCallEvent(call=event.call, index=index)
                    for index, event in enumerate(pending_tool_events)
                ]
                state.tool_calls = tuple(event.call for event in ordered)
                state.finish_reason = "tool_calls"
                for tool_event in ordered:
                    yield tool_event

        return ResponseStream(
            _events(),
            finalizer=lambda: tuple(channel_messages),
        )

    async def _run_channel_participant(
        self,
        *,
        request: ProxyTurnRequest,
        channel_name: str,
        participant: StaffedParticipant,
        channel_transcript: tuple[ChannelMessage, ...],
        pending: PendingToolContext | None = None,
    ) -> AgentRunResult:
        prompt = channel_message_prompt(
            channel_name=channel_name,
            agent_id=participant.agent_id,
            role_instance_label=(
                participant.label if participant.label != participant.agent_id else None
            ),
            role_instance_context=participant_context(participant),
        )
        stream = self._stream_text_agent(
            agent_id=participant.agent_id,
            prompt=prompt,
            prompt_role="system",
            model_id_override=request.model,
            conversation_messages=_channel_conversation_messages(
                channel_transcript=channel_transcript,
                participant_label=participant.label,
            ),
            host_tools=request.tools,
            extra_tools=PARTICIPANT_INTERNAL_TOOLS,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            pending_continuation=pending,
        )
        async for _ in stream:
            pass
        return await stream.get_final_response()


def _channel_conversation_messages(
    *,
    channel_transcript: tuple[ChannelMessage, ...],
    participant_label: str,
) -> tuple[ProxyInputMessage, ...]:
    messages: list[ProxyInputMessage] = []
    for message in channel_transcript:
        if message.author == participant_label:
            messages.append(
                ProxyInputMessage(
                    role="assistant",
                    content=message.content,
                    name=message.author,
                )
            )
            continue
        messages.append(
            ProxyInputMessage(
                role="user",
                content=message.content,
                name=message.author,
            )
        )
    return tuple(messages)


def _process_participant_results(
    *,
    participant: StaffedParticipant,
    response: AgentRunResult,
    request: ProxyTurnRequest,
    channel: Channel,
    current_transcript: list[ChannelMessage],
    channel_messages: list[ChannelMessage],
    state: ProxyTurnState,
    emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
    session_id: str,
) -> tuple[
    list[ProxyReasoningDeltaEvent],
    list[_ChannelDelivery],
    list[ProxyToolCallEvent],
]:
    reasoning_events: list[ProxyReasoningDeltaEvent] = []
    deliveries: list[_ChannelDelivery] = []
    tool_events: list[ProxyToolCallEvent] = []

    internal_actions: list[_ChannelDelivery] = []
    host_tool_calls: list[ProxyToolCall] = []
    for tool_call in response.tool_calls:
        if is_internal_tool_name(tool_call.name):
            if tool_call.name != "message_channel":
                raise ValueError(
                    f"participants cannot use internal tool: {tool_call.name}"
                )
            action = parse_message_channel_action(
                tool_call,
                require_channel=False,
            )
            internal_actions.append(
                _ChannelDelivery(
                    author=participant.label,
                    message=action.message,
                    recipients=action.recipients,
                )
            )
            continue
        host_tool_calls.append(tool_call)

    if internal_actions:
        deliveries.extend(internal_actions)
    else:
        text = response.text.strip()
        if text:
            line = ChannelMessage(
                author=participant.label,
                content=text,
            )
            channel_messages.append(line)
            current_transcript.append(line)
            state.append_reasoning(line.render())
            reasoning_events.append(ProxyReasoningDeltaEvent(line.render()))

    if host_tool_calls:
        tool_events.extend(
            emit_tool_calls(
                tool_calls=tuple(host_tool_calls),
                request=request,
                session_id=session_id,
                actor=participant.label,
                active_channel_id=channel.channel_id,
                state=state,
            )
        )

    return reasoning_events, deliveries, tool_events
