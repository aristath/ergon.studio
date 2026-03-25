from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from ergon_studio.debug_log import log_event
from ergon_studio.proxy.agent_runner import AgentInvoker, ProxyAgentRunner
from ergon_studio.proxy.channel_executor import ProxyChannelExecutor
from ergon_studio.proxy.channel_staffing import (
    expand_staffed_participants,
    require_staffed_recipients,
)
from ergon_studio.proxy.channels import (
    Channel,
    ChannelMessage,
    describe_open_channels,
)
from ergon_studio.proxy.continuation import (
    PendingContinuation,
    PendingToolContext,
    latest_pending_continuation,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.orchestrator_tools import (
    MalformedToolCallError,
    RunParallelAction,
    build_orchestrator_internal_tools,
    is_internal_tool_name,
    parse_close_channel_action,
    parse_message_channel_action,
    parse_open_channel_action,
    parse_run_parallel_action,
)
from ergon_studio.proxy.pending_store import PendingStore
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.session_overlay import SessionOverlay
from ergon_studio.proxy.subsession_executor import SubSessionExecutor
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
        overlay_root: Path | None = None,
    ) -> None:
        self.registry = registry
        self._overlay_root = overlay_root or (Path.home() / ".ergon-workspace")
        self._pending_store = PendingStore()
        self._agent_runner = ProxyAgentRunner(
            registry,
            invoker=agent_invoker,
            pending_store=self._pending_store,
        )
        self._channel_executor = ProxyChannelExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._agent_runner.emit_tool_call_events,
        )
        self._subsession_executor = SubSessionExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
        )
        self._channel_sessions: dict[str, dict[str, Channel]] = {}

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
                log_event(
                    "turn_start",
                    requested_session_id=session_id,
                    request=request,
                )
                pending = latest_pending_continuation(
                    request.messages,
                    pending_store=self._pending_store,
                )
                if pending is not None:
                    active_session_id = pending[0].session_id
                    channels = self._channel_sessions.get(active_session_id) or {}
                    log_event(
                        "turn_resume_pending",
                        session_id=active_session_id,
                        pending=pending,
                        open_channels=tuple(channels),
                    )
                else:
                    active_session_id = session_id or f"session_{uuid4().hex}"
                    channels = self._channel_sessions.get(active_session_id) or {}
                    log_event(
                        "turn_session_ready",
                        session_id=active_session_id,
                        open_channels=tuple(channels),
                    )
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
                log_event(
                    "turn_error",
                    session_id=active_session_id,
                    error=str(exc),
                    error_type="ValueError",
                )
                state.finish_reason = "error"
                state.content = str(exc)
                yield ProxyContentDeltaEvent(state.content)
            except Exception as exc:
                log_event(
                    "turn_error",
                    session_id=active_session_id,
                    error=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
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
        loop_history: list[ProxyInputMessage] = []
        retry_count = 0
        while True:
            log_event(
                "orchestrator_run_start",
                session_id=session_id,
                pending=bool(pending),
                open_channels=describe_open_channels(channels),
            )
            internal_tools = build_orchestrator_internal_tools(self.registry)
            orchestrator_stream = self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    open_channels=describe_open_channels(channels),
                ),
                prompt_role="system",
                model_id_override=request.model,
                conversation_messages=(*request.messages, *loop_history),
                host_tools=request.tools,
                extra_tools=internal_tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending,
            )
            buffered_deltas: list[str] = []
            async for delta in orchestrator_stream:
                log_event(
                    "orchestrator_delta",
                    session_id=session_id,
                    delta=delta,
                )
                buffered_deltas.append(delta)
            pending = None
            response = await orchestrator_stream.get_final_response()
            log_event(
                "orchestrator_run_result",
                session_id=session_id,
                response=response,
            )
            processed_internal_action = False
            host_tool_calls: list[ProxyToolCall] = []
            channel_results: list[tuple[str, str, tuple[ChannelMessage, ...]]] = []
            try:
                if response.tool_calls:
                    # Validation pre-pass: verify channel references before side effects
                    for tool_call in response.tool_calls:
                        if tool_call.name == "message_channel":
                            msg_action = parse_message_channel_action(tool_call)
                            if channels.get(msg_action.channel) is None:
                                raise ValueError(
                                    f"unknown channel: {msg_action.channel}"
                                )
                        elif tool_call.name == "close_channel":
                            close_action = parse_close_channel_action(tool_call)
                            if channels.get(close_action.channel) is None:
                                raise ValueError(
                                    f"unknown channel: {close_action.channel}"
                                )
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
                            highest_channel_number = 0
                            for existing_channel_id in channels:
                                if not existing_channel_id.startswith("channel-"):
                                    continue
                                try:
                                    highest_channel_number = max(
                                        highest_channel_number,
                                        int(
                                            existing_channel_id.removeprefix(
                                                "channel-"
                                            )
                                        ),
                                    )
                                except ValueError:
                                    continue
                            channel_id = f"channel-{highest_channel_number + 1}"
                            if open_action.preset is not None:
                                channel = Channel(
                                    channel_id=channel_id,
                                    name=open_action.preset,
                                    participants=self.registry.channel_presets[
                                        open_action.preset
                                    ],
                                )
                            elif open_action.participants:
                                channel = Channel(
                                    channel_id=channel_id,
                                    name="ad hoc",
                                    participants=open_action.participants,
                                )
                            else:
                                raise ValueError(
                                    "open_channel needs a preset or participants target"
                                )
                            channels[channel.channel_id] = channel
                            log_event(
                                "open_channel_action",
                                session_id=session_id,
                                channel=channel,
                                message=open_action.message,
                                recipients=open_action.recipients,
                            )
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
                                if buffered_deltas:
                                    text = "".join(buffered_deltas)
                                    state.append_reasoning(text)
                                    yield ProxyReasoningDeltaEvent(text)
                                return
                            channel_results.append(
                                (
                                    channel.channel_id,
                                    channel.name,
                                    await channel_stream.get_final_response(),
                                )
                            )
                            continue
                        if tool_call.name == "message_channel":
                            message_action = parse_message_channel_action(
                                tool_call,
                            )
                            log_event(
                                "message_channel_action",
                                session_id=session_id,
                                channel=message_action.channel,
                                message=message_action.message,
                                recipients=message_action.recipients,
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
                                if buffered_deltas:
                                    text = "".join(buffered_deltas)
                                    state.append_reasoning(text)
                                    yield ProxyReasoningDeltaEvent(text)
                                return
                            ch = channels.get(message_action.channel)
                            channel_results.append(
                                (
                                    message_action.channel,
                                    ch.name if ch else "?",
                                    await channel_stream.get_final_response(),
                                )
                            )
                            continue
                        if tool_call.name == "close_channel":
                            close_action = parse_close_channel_action(tool_call)
                            closed = channels.pop(close_action.channel, None)
                            if closed is None:
                                raise ValueError(
                                    f"unknown channel: {close_action.channel}"
                                )
                            notice = (
                                f"Orchestrator: closing channel {closed.channel_id} "
                                f"({closed.name}).\n"
                            )
                            log_event(
                                "close_channel_action",
                                session_id=session_id,
                                channel_id=closed.channel_id,
                                channel_name=closed.name,
                            )
                            state.append_reasoning(notice)
                            yield ProxyReasoningDeltaEvent(notice)
                            continue
                        if tool_call.name == "run_parallel":
                            parallel_action = parse_run_parallel_action(
                                tool_call,
                                registry=self.registry,
                            )
                            log_event(
                                "run_parallel_action",
                                session_id=session_id,
                                agent=parallel_action.agent,
                                count=parallel_action.count,
                            )
                            results, parallel_events = (
                                await self._run_parallel_subsessions(
                                    action=parallel_action,
                                    session_id=session_id,
                                    model_id=request.model,
                                )
                            )
                            for event in parallel_events:
                                state.append_reasoning(event.delta)
                                yield event
                            loop_history.append(
                                ProxyInputMessage(
                                    role="user",
                                    content=_format_parallel_results(
                                        parallel_action, results
                                    ),
                                )
                            )
                            continue
                        raise ValueError(
                            f"unsupported internal action: {tool_call.name}"
                        )
                    if host_tool_calls:
                        if buffered_deltas:
                            text = "".join(buffered_deltas)
                            if processed_internal_action:
                                state.append_reasoning(text)
                                yield ProxyReasoningDeltaEvent(text)
                            else:
                                state.append_content(text)
                                yield ProxyContentDeltaEvent(text)
                        for tool_event in self._agent_runner.emit_tool_call_events(
                            tool_calls=tuple(host_tool_calls),
                            request=request,
                            session_id=session_id,
                            actor="orchestrator",
                            state=state,
                        ):
                            yield tool_event
                        return
            except MalformedToolCallError:
                if retry_count < 2:
                    retry_count += 1
                    continue
                raise

            retry_count = 0

            if buffered_deltas:
                text = "".join(buffered_deltas)
                if processed_internal_action:
                    state.append_reasoning(text)
                    yield ProxyReasoningDeltaEvent(text)
                else:
                    state.append_content(text)
                    yield ProxyContentDeltaEvent(text)

            if processed_internal_action:
                text_this_iteration = "".join(buffered_deltas)
                if text_this_iteration:
                    loop_history.append(
                        ProxyInputMessage(role="assistant", content=text_this_iteration)
                    )
                for cid, cname, cresult in channel_results:
                    if cresult:
                        lines = "\n".join(
                            f"  {m.author}: {m.content}" for m in cresult
                        )
                        loop_history.append(
                            ProxyInputMessage(
                                role="user",
                                content=f"Channel {cid} ({cname}) completed:\n{lines}",
                            )
                        )
                continue

            if request.tool_choice == "required" or isinstance(
                request.tool_choice, dict
            ):
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
        if pending:
            pending_channel_id = pending[0].active_channel_id
            if pending_channel_id is None:
                raise ValueError(
                    "pending channel resume is missing an active channel id"
                )
            channel_id = pending_channel_id
        channel = channels.get(channel_id)
        if channel is None:
            raise ValueError(f"unknown channel: {channel_id}")
        if pending is None:
            require_staffed_recipients(
                staffed_members=expand_staffed_participants(channel.participants),
                recipients=recipients,
            )
        if pending is not None:
            intro = (
                f"Orchestrator: continuing channel {channel.channel_id} "
                f"({channel.name}) with "
                f"{', '.join(item.actor for item in pending)}."
            )
        elif channel.transcript:
            intro = (
                f"Orchestrator: continuing channel {channel.channel_id} "
                f"({channel.name})."
            )
        else:
            roster = ", ".join(channel.participants)
            if channel.name == "ad hoc":
                intro = (
                    f"Orchestrator: opening channel {channel.channel_id} with {roster}."
                )
            else:
                intro = (
                    f"Orchestrator: opening channel {channel.channel_id} "
                    f"({channel.name}) with {roster}."
                )
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
        log_event(
            "channel_stream_start",
            session_id=session_id,
            channel_id=channel.channel_id,
            channel_name=channel.name,
            message=message,
            recipients=recipients,
            pending_actors=tuple(item.actor for item in pending) if pending else (),
        )

        async def _events() -> AsyncIterator[ProxyEvent]:
            nonlocal channel_result
            intro_line = intro + "\n"
            state.append_reasoning(intro_line)
            yield ProxyReasoningDeltaEvent(intro_line)
            async for event in channel_stream:
                yield event
            channel_result = await channel_stream.get_final_response()
            if message and pending is None:
                channel.transcript.append(
                    ChannelMessage(author="orchestrator", content=message)
                )
            channel.transcript.extend(channel_result)
            log_event(
                "channel_stream_complete",
                session_id=session_id,
                channel_id=channel.channel_id,
                channel_name=channel.name,
                produced_messages=channel_result,
                finish_reason=state.finish_reason,
            )

        return ResponseStream(
            _events(),
            finalizer=lambda: channel_result,
        )

    async def _run_parallel_subsessions(
        self,
        *,
        action: RunParallelAction,
        session_id: str,
        model_id: str,
    ) -> tuple[list[str], list[ProxyReasoningDeltaEvent]]:
        all_events: list[ProxyReasoningDeltaEvent] = []
        streams = [
            self._subsession_executor.execute(
                agent_id=action.agent,
                task=action.task,
                session_id=f"{session_id}-parallel-{i}",
                overlay=SessionOverlay(
                    root=self._overlay_root / f"{session_id}-parallel-{i}"
                ),
                model_id=model_id,
            )
            for i in range(action.count)
        ]

        async def _drain(
            stream: ResponseStream[ProxyReasoningDeltaEvent, str],
        ) -> str:
            async for event in stream:
                all_events.append(event)
            return await stream.get_final_response()

        results: list[str] = list(await asyncio.gather(*(_drain(s) for s in streams)))
        return results, all_events

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
        if session_id is not None and result.finish_reason not in {
            "error",
            "tool_calls",
        }:
            if channels:
                self._channel_sessions[session_id] = channels
            else:
                self._channel_sessions.pop(session_id, None)
        log_event(
            "turn_result",
            session_id=session_id,
            result=result,
            open_channels=tuple(channels),
        )
        return result

    async def _resume_pending_groups(
        self,
        *,
        request: ProxyTurnRequest,
        state: ProxyTurnState,
        channels: dict[str, Channel],
        session_id: str,
        pending: PendingContinuation,
    ) -> AsyncIterator[ProxyEvent]:
        log_event(
            "pending_resume_start",
            session_id=session_id,
            pending=pending,
        )
        channel_pending: dict[str, list[PendingToolContext]] = {}
        orchestrator_items: list[PendingToolContext] = []
        for item in pending:
            if item.active_channel_id is None:
                if item.actor != "orchestrator":
                    raise ValueError(
                        "pending channel resume is missing an active channel id"
                    )
                orchestrator_items.append(item)
                continue
            channel_pending.setdefault(item.active_channel_id, []).append(item)

        for channel_id, items in channel_pending.items():
            stream = self._message_channel(
                request=request,
                channels=channels,
                channel_id=channel_id,
                pending=tuple(items),
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


def _format_parallel_results(
    action: RunParallelAction, results: list[str]
) -> str:
    header = f"run_parallel({action.agent!r}, count={action.count}) results:"
    body = "\n".join(
        f"\n[Sub-session {i + 1}]\n{result}" for i, result in enumerate(results)
    )
    return header + body
