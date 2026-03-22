from __future__ import annotations

import time
from collections.abc import AsyncIterator

from ergon_studio.proxy.agent_runner import AgentInvoker, ProxyAgentRunner
from ergon_studio.proxy.channel_executor import ProxyChannelExecutor
from ergon_studio.proxy.channels import (
    ChannelSnapshot,
    OpenChannel,
    describe_open_channels,
)
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    latest_pending_continuation,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.orchestrator_tools import (
    build_orchestrator_internal_tools,
    is_internal_tool_name,
    parse_close_channel_action,
    parse_message_channel_action,
    parse_open_channel_action,
)
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyOrchestrationCore:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_invoker: AgentInvoker | None = None,
    ) -> None:
        self.registry = registry
        self._agent_runner = ProxyAgentRunner(
            registry,
            invoker=agent_invoker,
        )
        channel_executor = ProxyChannelExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._agent_runner.emit_tool_call_events,
        )
        self._channel_executor = channel_executor

    def stream_turn(
        self,
        request: ProxyTurnRequest,
        *,
        created_at: int | None = None,
    ) -> ResponseStream[ProxyEvent, ProxyTurnResult]:
        if created_at is None:
            created_at = int(time.time())
        state = ProxyTurnState()

        async def _events() -> AsyncIterator[ProxyEvent]:
            try:
                pending = latest_pending_continuation(request.messages)
                worklog = list(pending.state.worklog if pending is not None else ())
                channels = _channels_from_pending(pending)
                next_channel_number = _next_channel_number(channels)
                if pending is not None:
                    if pending.state.active_channel_id is not None:
                        stream = self._message_channel(
                            request=request,
                            channels=channels,
                            channel_id=pending.state.active_channel_id,
                            continuation=pending.state,
                            pending=pending,
                            state=state,
                            worklog=tuple(worklog),
                        )
                        async for event in stream:
                            yield event
                        if state.finish_reason == "tool_calls":
                            return
                        worklog.extend(await stream.get_final_response())
                        async for event in self._run_orchestrator_loop(
                            request=request,
                            state=state,
                            worklog=worklog,
                            channels=channels,
                            next_channel_number=next_channel_number,
                        ):
                            yield event
                        return
                    async for event in self._run_orchestrator_loop(
                        request=request,
                        state=state,
                        worklog=worklog,
                        channels=channels,
                        next_channel_number=next_channel_number,
                        pending_orchestrator=pending,
                    ):
                        yield event
                    return

                async for event in self._run_orchestrator_loop(
                    request=request,
                    state=state,
                    worklog=worklog,
                    channels=channels,
                    next_channel_number=next_channel_number,
                ):
                    yield event
            except ValueError as exc:
                state.finish_reason = "error"
                state.content = str(exc)
                yield ProxyContentDeltaEvent(state.content)
            except Exception as exc:
                state.finish_reason = "error"
                state.content = f"{type(exc).__name__}: {exc}"
                yield ProxyContentDeltaEvent(state.content)
            yield ProxyFinishEvent(state.finish_reason)

        return ResponseStream(
            _events(),
            finalizer=lambda: ProxyTurnResult(
                finish_reason=state.finish_reason,
                content=state.content,
                reasoning=state.reasoning,
                tool_calls=state.tool_calls,
                output_items=state.output_items,
            ),
        )

    async def _run_orchestrator_loop(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        worklog: list[str],
        channels: dict[str, OpenChannel],
        next_channel_number: int,
        pending_orchestrator: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        pending = pending_orchestrator
        while True:
            internal_tools = build_orchestrator_internal_tools(self.registry)
            orchestrator_stream = self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
                    worklog=tuple(worklog),
                    open_channels=describe_open_channels(channels),
                ),
                model_id_override=request.model,
                host_tools=request.tools,
                extra_tools=internal_tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending,
            )
            async for delta in orchestrator_stream:
                state.append_content(delta)
                yield ProxyContentDeltaEvent(delta)
            pending = None
            response = await orchestrator_stream.get_final_response()
            tool_calls = response.tool_calls
            internal_tool_calls = tuple(
                tool_call
                for tool_call in tool_calls
                if is_internal_tool_name(tool_call.name)
            )
            host_tool_calls = tuple(
                tool_call
                for tool_call in tool_calls
                if not is_internal_tool_name(tool_call.name)
            )
            if internal_tool_calls and host_tool_calls:
                raise ValueError(
                    "orchestrator cannot mix internal actions with host tool calls"
                )
            if len(internal_tool_calls) > 1:
                raise ValueError(
                    "orchestrator must use at most one internal action at a time"
                )
            if host_tool_calls:
                for tool_event in self._agent_runner.emit_tool_call_events(
                    tool_calls=host_tool_calls,
                    request=request,
                    continuation=_orchestrator_continuation_state(
                        worklog=tuple(worklog),
                        channels=channels,
                    ),
                    state=state,
                ):
                    yield tool_event
                return
            if internal_tool_calls:
                tool_call = internal_tool_calls[0]
                if tool_call.name == "open_channel":
                    open_action = parse_open_channel_action(
                        tool_call,
                        registry=self.registry,
                    )
                    channel_id = f"channel-{next_channel_number}"
                    next_channel_number += 1
                    channel = _open_channel(
                        registry=self.registry,
                        channel_id=channel_id,
                        preset=open_action.preset,
                        participants=open_action.participants,
                    )
                    channels[channel.channel_id] = channel
                    channel_stream = self._message_channel(
                        request=request,
                        channels=channels,
                        channel_id=channel.channel_id,
                        message=open_action.message,
                        state=state,
                        worklog=tuple(worklog),
                    )
                    async for internal_event in channel_stream:
                        yield internal_event
                    if state.finish_reason == "tool_calls":
                        return
                    result = await channel_stream.get_final_response()
                    if result:
                        worklog.extend(result)
                    continue
                if tool_call.name == "message_channel":
                    message_action = parse_message_channel_action(tool_call)
                    channel_stream = self._message_channel(
                        request=request,
                        channels=channels,
                        channel_id=message_action.channel,
                        message=message_action.message,
                        state=state,
                        worklog=tuple(worklog),
                    )
                    async for internal_event in channel_stream:
                        yield internal_event
                    if state.finish_reason == "tool_calls":
                        return
                    result = await channel_stream.get_final_response()
                    if result:
                        worklog.extend(result)
                    continue
                if tool_call.name == "close_channel":
                    close_action = parse_close_channel_action(tool_call)
                    closed = channels.pop(close_action.channel, None)
                    if closed is None:
                        raise ValueError(f"unknown channel: {close_action.channel}")
                    notice = (
                        f"Orchestrator: closing channel {closed.channel_id} "
                        f"({closed.name}).\n"
                    )
                    state.append_reasoning(notice)
                    yield ProxyReasoningDeltaEvent(notice)
                    if closed.transcript:
                        worklog.extend(tuple(closed.transcript[-6:]))
                    continue
                raise ValueError(f"unsupported internal action: {tool_call.name}")

            if _requires_host_tool_result(request):
                raise ValueError("model ignored required host tool choice")
            return

    def _message_channel(
        self,
        *,
        request: ProxyTurnRequest,
        channels: dict[str, OpenChannel],
        channel_id: str,
        message: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        worklog: tuple[str, ...] = (),
    ) -> ResponseStream[ProxyEvent, tuple[str, ...]]:
        if continuation is not None:
            channel_id = continuation.active_channel_id or channel_id
        channel = channels.get(channel_id)
        if channel is None:
            state.finish_reason = "error"
            error_text = f"unknown channel: {channel_id}"
            state.content = error_text
            return ResponseStream(
                _single_event_stream(ProxyContentDeltaEvent(error_text)),
                finalizer=lambda: (),
            )
        if continuation is not None:
            intro = _channel_notice(
                f"Orchestrator: continuing channel {channel.channel_id} "
                f"({channel.name}) with "
                f"{continuation.actor}."
            )
        else:
            intro = _channel_notice(_channel_intro(channel))
        channel_stream = self._channel_executor.execute(
            request=request,
            channel=channel,
            channels=channels,
            channel_message=message,
            state=state,
            continuation=continuation,
            pending=pending,
            worklog=worklog,
        )
        channel_result: tuple[str, ...] = ()

        async def _events() -> AsyncIterator[ProxyEvent]:
            nonlocal channel_result
            state.append_reasoning(intro)
            yield ProxyReasoningDeltaEvent(intro)
            async for event in channel_stream:
                yield event
            channel_result = await channel_stream.get_final_response()
            if message and continuation is None:
                channel.transcript.append(f"orchestrator: {message}")
            channel.transcript.extend(channel_result)

        return ResponseStream(
            _events(),
            finalizer=lambda: channel_result,
        )


def _orchestrator_continuation_state(
    *,
    worklog: tuple[str, ...],
    channels: dict[str, OpenChannel],
) -> ContinuationState:
    return ContinuationState(
        actor="orchestrator",
        channels=tuple(_snapshot_open_channels(channels)),
        worklog=worklog,
    )


def _requires_host_tool_result(request: ProxyTurnRequest) -> bool:
    tool_choice = request.tool_choice
    if tool_choice == "required":
        return True
    return isinstance(tool_choice, dict)


async def _single_event_stream(event: ProxyEvent) -> AsyncIterator[ProxyEvent]:
    yield event


def _open_channel(
    *,
    registry: RuntimeRegistry,
    channel_id: str,
    preset: str | None,
    participants: tuple[str, ...],
) -> OpenChannel:
    if preset is not None:
        resolved_participants = registry.channel_presets.get(preset)
        if resolved_participants is None:
            raise ValueError(f"unknown channel preset: {preset}")
        return OpenChannel(
            channel_id=channel_id,
            name=preset,
            participants=participants or resolved_participants,
        )
    if participants:
        return OpenChannel(
            channel_id=channel_id,
            name="ad hoc",
            participants=participants,
        )
    raise ValueError("open_channel needs a preset or participants target")


def _channel_notice(base: str) -> str:
    return base + "\n"


def _channel_intro(channel: OpenChannel) -> str:
    roster = ", ".join(channel.participants)
    if channel.name == "ad hoc":
        return f"Orchestrator: opening channel {channel.channel_id} with {roster}."
    return (
        f"Orchestrator: opening channel {channel.channel_id} "
        f"({channel.name}) with {roster}."
    )


def _channels_from_pending(
    pending: PendingContinuation | None,
) -> dict[str, OpenChannel]:
    if pending is None:
        return {}
    return {
        channel.channel_id: OpenChannel(
            channel_id=channel.channel_id,
            name=channel.name,
            participants=channel.participants,
            transcript=list(channel.transcript),
        )
        for channel in pending.state.channels
    }


def _next_channel_number(channels: dict[str, OpenChannel]) -> int:
    highest = 0
    for channel_id in channels:
        if not channel_id.startswith("channel-"):
            continue
        try:
            highest = max(highest, int(channel_id.removeprefix("channel-")))
        except ValueError:
            continue
    return highest + 1


def _snapshot_open_channels(
    channels: dict[str, OpenChannel],
) -> tuple[ChannelSnapshot, ...]:
    return tuple(channel.snapshot() for channel in channels.values())
