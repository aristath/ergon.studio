from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from ergon_studio.proxy.agent_runner import AgentInvoker, ProxyAgentRunner
from ergon_studio.proxy.channel_executor import ProxyChannelExecutor
from ergon_studio.proxy.channel_staffing import (
    expand_staffed_participants,
    resolve_staffed_recipients,
)
from ergon_studio.proxy.channels import (
    Channel,
    ChannelMessage,
    ChannelStore,
    describe_open_channels,
)
from ergon_studio.proxy.continuation import (
    PendingContinuation,
    PendingToolContext,
    latest_pending_continuation,
    pending_actors,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
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
from ergon_studio.proxy.pending_store import PendingSeed, PendingStore
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
        self._pending_store = PendingStore()
        self._agent_runner = ProxyAgentRunner(
            registry,
            invoker=agent_invoker,
            pending_store=self._pending_store,
        )
        channel_executor = ProxyChannelExecutor(
            registry=registry,
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._agent_runner.emit_tool_call_events,
        )
        self._channel_executor = channel_executor
        self._channel_store = ChannelStore()

    def stream_turn(
        self,
        request: ProxyTurnRequest,
        *,
        session_id: str | None = None,
    ) -> ResponseStream[ProxyEvent, ProxyTurnResult]:
        state = ProxyTurnState()
        active_session_id: str | None = None
        channels: dict[str, Channel] = {}

        async def _events() -> AsyncIterator[ProxyEvent]:
            nonlocal active_session_id
            nonlocal channels
            try:
                pending = latest_pending_continuation(
                    request.messages,
                    pending_store=self._pending_store,
                )
                if pending is not None:
                    active_session_id = pending.session_id
                    channels = self._channel_store.get(active_session_id) or {}
                else:
                    active_session_id = session_id or f"session_{uuid4().hex}"
                    channels = self._channel_store.get(active_session_id) or {}
                if pending is not None:
                    async for event in self._resume_pending_groups(
                        request=request,
                        state=state,
                        channels=channels,
                        session_id=active_session_id,
                        pending=pending,
                    ):
                        yield event
                    if state.finish_reason != "tool_calls":
                        return
                    return

                async for event in self._run_orchestrator_loop(
                    request=request,
                    state=state,
                    channels=channels,
                    session_id=active_session_id,
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
            finalizer=lambda: self._finalize_turn(
                state=state,
                session_id=active_session_id,
                channels=channels,
            ),
        )

    async def _run_orchestrator_loop(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        channels: dict[str, Channel],
        session_id: str,
        pending_orchestrator: PendingToolContext | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        pending = pending_orchestrator
        while True:
            internal_tools = build_orchestrator_internal_tools(self.registry)
            orchestrator_stream = self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
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
            pending_host_tool_calls: list[ProxyToolCallEvent] = []
            processed_internal_action = False
            if response.tool_calls:
                host_tool_calls: list[ProxyToolCall] = []
                for tool_call in response.tool_calls:
                    if not is_internal_tool_name(tool_call.name):
                        host_tool_calls.append(tool_call)
                        continue
                    processed_internal_action = True
                    if tool_call.name == "open_channel":
                        open_action = parse_open_channel_action(
                            tool_call,
                            registry=self.registry,
                        )
                        channel_id = f"channel-{_next_channel_number(channels)}"
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
                            recipients=open_action.recipients,
                            state=state,
                            session_id=session_id,
                        )
                        async for internal_event in channel_stream:
                            yield internal_event
                        if state.finish_reason == "tool_calls":
                            return
                        await channel_stream.get_final_response()
                        continue
                    if tool_call.name == "message_channel":
                        message_action = parse_message_channel_action(
                            tool_call,
                            registry=self.registry,
                        )
                        channel_stream = self._message_channel(
                            request=request,
                            channels=channels,
                            channel_id=message_action.channel,
                            message=message_action.message,
                            recipients=message_action.recipients,
                            state=state,
                            session_id=session_id,
                        )
                        async for internal_event in channel_stream:
                            yield internal_event
                        if state.finish_reason == "tool_calls":
                            return
                        await channel_stream.get_final_response()
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
                        continue
                    raise ValueError(f"unsupported internal action: {tool_call.name}")
                if host_tool_calls:
                    pending_host_tool_calls.extend(
                        self._agent_runner.emit_tool_call_events(
                            tool_calls=tuple(host_tool_calls),
                            request=request,
                            continuation=PendingSeed(
                                session_id=session_id,
                                actor="orchestrator",
                            ),
                            state=state,
                        )
                    )
            if pending_host_tool_calls:
                for tool_event in pending_host_tool_calls:
                    yield tool_event
                return
            if processed_internal_action:
                continue

            if _requires_host_tool_result(request):
                raise ValueError("model ignored required host tool choice")
            return

    def _message_channel(
        self,
        *,
        request: ProxyTurnRequest,
        channels: dict[str, Channel],
        channel_id: str,
        message: str | None = None,
        recipients: tuple[str, ...] = (),
        state: ProxyTurnState,
        pending: PendingContinuation | None = None,
        session_id: str,
    ) -> ResponseStream[ProxyEvent, tuple[ChannelMessage, ...]]:
        if pending is not None and pending.items:
            channel_id = pending.items[0].active_channel_id or channel_id
        channel = channels.get(channel_id)
        if channel is None:
            state.finish_reason = "error"
            error_text = f"unknown channel: {channel_id}"
            state.content = error_text
            return ResponseStream(
                _single_event_stream(ProxyContentDeltaEvent(error_text)),
                finalizer=lambda: (),
            )
        if pending is None:
            _validate_channel_recipients(channel, recipients)
        if pending is not None:
            intro = _channel_notice(
                f"Orchestrator: continuing channel {channel.channel_id} "
                f"({channel.name}) with "
                f"{', '.join(pending_actors(pending))}."
            )
        elif channel.transcript:
            intro = _channel_notice(
                f"Orchestrator: continuing channel {channel.channel_id} "
                f"({channel.name})."
            )
        else:
            intro = _channel_notice(_channel_intro(channel))
        channel_stream = self._channel_executor.execute(
            request=request,
            session_id=session_id,
            channel=channel,
            channel_message=message,
            recipients=recipients,
            state=state,
            pending=pending,
        )
        channel_result: tuple[ChannelMessage, ...] = ()

        async def _events() -> AsyncIterator[ProxyEvent]:
            nonlocal channel_result
            state.append_reasoning(intro)
            yield ProxyReasoningDeltaEvent(intro)
            async for event in channel_stream:
                yield event
            channel_result = await channel_stream.get_final_response()
            if message and pending is None:
                channel.transcript.append(
                    ChannelMessage(author="orchestrator", content=message)
                )
            channel.transcript.extend(channel_result)

        return ResponseStream(
            _events(),
            finalizer=lambda: channel_result,
        )

    def _finalize_turn(
        self,
        *,
        state: ProxyTurnState,
        session_id: str | None,
        channels: dict[str, Channel],
    ) -> ProxyTurnResult:
        result = ProxyTurnResult(
            finish_reason=state.finish_reason,
            content=state.content,
            reasoning=state.reasoning,
            tool_calls=state.tool_calls,
            output_items=state.output_items,
        )
        self._persist_channels_for_result(
            result=result,
            session_id=session_id,
            channels=channels,
        )
        return result

    def _persist_channels_for_result(
        self,
        *,
        result: ProxyTurnResult,
        session_id: str | None,
        channels: dict[str, Channel],
    ) -> None:
        if session_id is None or result.finish_reason in {"error"}:
            return
        if result.finish_reason == "tool_calls":
            return
        if channels:
            self._channel_store.put(session_id, channels)
        else:
            self._channel_store.discard(session_id)

    async def _resume_pending_groups(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        channels: dict[str, Channel],
        session_id: str,
        pending: PendingContinuation,
    ) -> AsyncIterator[ProxyEvent]:
        channel_pending: dict[str, list[PendingToolContext]] = {}
        orchestrator_items: list[PendingToolContext] = []
        for item in pending.items:
            if item.active_channel_id is None:
                orchestrator_items.append(item)
                continue
            channel_pending.setdefault(item.active_channel_id, []).append(item)

        for channel_id, items in channel_pending.items():
            stream = self._message_channel(
                request=request,
                channels=channels,
                channel_id=channel_id,
                pending=PendingContinuation(
                    session_id=pending.session_id,
                    items=tuple(items),
                ),
                state=state,
                session_id=session_id,
            )
            async for event in stream:
                yield event
            if state.finish_reason == "tool_calls":
                return
            await stream.get_final_response()

        async for event in self._run_orchestrator_loop(
            request=request,
            state=state,
            channels=channels,
            session_id=session_id,
            pending_orchestrator=(
                orchestrator_items[0]
                if orchestrator_items
                else None
            ),
        ):
            yield event


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
) -> Channel:
    if preset is not None:
        resolved_participants = registry.channel_presets.get(preset)
        if resolved_participants is None:
            raise ValueError(f"unknown channel preset: {preset}")
        return Channel(
            channel_id=channel_id,
            name=preset,
            participants=participants or resolved_participants,
        )
    if participants:
        return Channel(
            channel_id=channel_id,
            name="ad hoc",
            participants=participants,
        )
    raise ValueError("open_channel needs a preset or participants target")


def _channel_notice(base: str) -> str:
    return base + "\n"


def _channel_intro(channel: Channel) -> str:
    roster = ", ".join(channel.participants)
    if channel.name == "ad hoc":
        return f"Orchestrator: opening channel {channel.channel_id} with {roster}."
    return (
        f"Orchestrator: opening channel {channel.channel_id} "
        f"({channel.name}) with {roster}."
    )


def _next_channel_number(channels: dict[str, Channel]) -> int:
    highest = 0
    for channel_id in channels:
        if not channel_id.startswith("channel-"):
            continue
        try:
            highest = max(highest, int(channel_id.removeprefix("channel-")))
        except ValueError:
            continue
    return highest + 1


def _validate_channel_recipients(
    channel: Channel,
    recipients: tuple[str, ...],
) -> None:
    staffed_members = expand_staffed_participants(channel.participants)
    resolved = resolve_staffed_recipients(
        staffed_members=staffed_members,
        recipients=recipients,
    )
    if len(resolved) == len(recipients):
        return

    available: set[str] = set()
    for participant in staffed_members:
        available.add(participant.agent_id)
        available.add(participant.label)
    invalid = [recipient for recipient in recipients if recipient not in available]
    if not invalid:
        invalid = list(recipients)
    if invalid:
        raise ValueError(
            "channel recipients are not staffed in this channel: "
            + ", ".join(sorted(invalid))
        )
