from __future__ import annotations

import time
from collections.abc import AsyncIterator

from ergon_studio.proxy.agent_runner import AgentInvoker, ProxyAgentRunner
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
    parse_message_workroom_action,
)
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workroom_executor import ProxyWorkroomExecutor
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyOrchestrationCore:
    _MAX_INTERNAL_MOVES = 8

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
        workroom_executor = ProxyWorkroomExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._agent_runner.emit_tool_call_events,
        )
        self._workroom_executor = workroom_executor

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
                if pending is not None:
                    if pending.state.workroom_name is not None:
                        stream = self._message_workroom(
                            request=request,
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
                        ):
                            yield event
                        return
                    async for event in self._run_orchestrator_loop(
                        request=request,
                        state=state,
                        worklog=worklog,
                        pending_orchestrator=pending,
                    ):
                        yield event
                    return

                async for event in self._run_orchestrator_loop(
                    request=request,
                    state=state,
                    worklog=worklog,
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
        pending_orchestrator: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        pending = pending_orchestrator
        for _ in range(self._MAX_INTERNAL_MOVES):
            internal_tools = build_orchestrator_internal_tools(self.registry)
            orchestrator_stream = self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
                    worklog=tuple(worklog),
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
                    continuation=_orchestrator_continuation_state(tuple(worklog)),
                    state=state,
                ):
                    yield tool_event
                return
            if internal_tool_calls:
                action = parse_message_workroom_action(
                    internal_tool_calls[0],
                    registry=self.registry,
                )
                workroom_stream = self._message_workroom(
                    request=request,
                    workroom_name=action.preset,
                    participants=action.participants,
                    workroom_message=action.message,
                    state=state,
                    worklog=tuple(worklog),
                )
                async for internal_event in workroom_stream:
                    yield internal_event
                if state.finish_reason == "tool_calls":
                    return
                result = await workroom_stream.get_final_response()
                if result:
                    worklog.extend(result)
                else:
                    return
                continue

            if _requires_host_tool_result(request):
                raise ValueError("model ignored required host tool choice")
            return

        raise ValueError("orchestrator exceeded internal move limit")

    def _message_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_name: str | None = None,
        participants: tuple[str, ...] = (),
        workroom_message: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        worklog: tuple[str, ...] = (),
    ) -> ResponseStream[ProxyEvent, tuple[str, ...]]:
        if (
            continuation is None
            and workroom_name is None
            and not participants
        ):
            state.finish_reason = "error"
            error_text = "message_workroom needs a preset or participants target."
            state.content = error_text
            return ResponseStream(
                _single_event_stream(ProxyContentDeltaEvent(error_text)),
                finalizer=lambda: (),
            )
        if continuation is not None:
            assert continuation.workroom_name is not None
            workroom_name = continuation.workroom_name
            intro = _workroom_notice(
                f"Orchestrator: continuing workroom {workroom_name} with "
                f"{continuation.actor}."
            )
            participants = continuation.workroom_participants
            workroom_message = continuation.workroom_message
        else:
            resolved = _resolve_workroom_target(
                registry=self.registry,
                preset=workroom_name,
                participants=participants,
            )
            if resolved is None:
                error_text = f"Unknown workroom: {workroom_name or '(none)'}"
                state.finish_reason = "error"
                state.content = error_text
                return ResponseStream(
                    _single_event_stream(ProxyContentDeltaEvent(error_text)),
                    finalizer=lambda: (),
                )
            workroom_name, participants = resolved
            intro = _workroom_notice(_workroom_intro(workroom_name))
        workroom_stream = self._workroom_executor.execute(
            request=request,
            workroom_name=workroom_name,
            participants=participants,
            workroom_message=workroom_message,
            state=state,
            continuation=continuation,
            pending=pending,
            worklog=worklog,
        )
        workroom_result: tuple[str, ...] = ()

        async def _events() -> AsyncIterator[ProxyEvent]:
            nonlocal workroom_result
            state.append_reasoning(intro)
            yield ProxyReasoningDeltaEvent(intro)
            async for event in workroom_stream:
                yield event
            workroom_result = await workroom_stream.get_final_response()

        return ResponseStream(
            _events(),
            finalizer=lambda: workroom_result,
        )


def _orchestrator_continuation_state(
    worklog: tuple[str, ...],
) -> ContinuationState:
    return ContinuationState(
        actor="orchestrator",
        worklog=worklog,
    )


def _requires_host_tool_result(request: ProxyTurnRequest) -> bool:
    tool_choice = request.tool_choice
    if tool_choice == "required":
        return True
    return isinstance(tool_choice, dict)


async def _single_event_stream(event: ProxyEvent) -> AsyncIterator[ProxyEvent]:
    yield event


def _resolve_workroom_target(
    *,
    registry: RuntimeRegistry,
    preset: str | None,
    participants: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    if preset is not None:
        resolved_participants = registry.workroom_definitions.get(preset)
        if resolved_participants is None:
            return None
        return preset, participants or resolved_participants
    if participants:
        return "ad hoc", participants
    return None


def _workroom_notice(base: str) -> str:
    return base + "\n"


def _workroom_intro(workroom_name: str) -> str:
    if workroom_name == "ad hoc":
        return "Orchestrator: opening an ad hoc workroom."
    return f"Orchestrator: opening workroom {workroom_name}."
