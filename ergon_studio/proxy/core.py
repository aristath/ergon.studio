from __future__ import annotations

import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Any
from uuid import uuid4

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
    MessageWorkroomAction,
    build_orchestrator_internal_tools,
    is_internal_tool_name,
    parse_message_workroom_action,
)
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workroom_executor import ProxyWorkroomExecutor
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream
from ergon_studio.workroom_layout import workroom_participants_for_definition

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
        state = ProxyTurnState(mode="orchestrator")

        async def _events() -> AsyncIterator[ProxyEvent]:
            try:
                pending = latest_pending_continuation(request.messages)
                worklog = list(pending.state.worklog if pending is not None else ())
                if pending is not None:
                    if pending.state.workroom_name is not None:
                        result_holder: dict[str, object] = {}
                        async for event in self._resume_subtask(
                            request=request,
                            pending=pending,
                            state=state,
                            worklog=tuple(worklog),
                            result_holder=result_holder,
                        ):
                            yield event
                        if state.finish_reason == "tool_calls":
                            return
                        if result_holder:
                            worklog.extend(_result(result_holder))
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
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason=state.finish_reason,
                content=state.content,
                reasoning=state.reasoning,
                mode=state.mode,
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
            response_holder: dict[str, Any] = {}
            internal_tools = build_orchestrator_internal_tools(self.registry)
            async for delta in self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
                    worklog=tuple(worklog),
                ),
                session_id=f"proxy-orchestrator-{uuid4().hex}",
                model_id_override=request.model,
                host_tools=request.tools,
                extra_tools=internal_tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                pending_continuation=pending,
                final_response_sink=partial(_store_response, response_holder),
            ):
                state.mode = "orchestrator"
                state.append_content(delta)
                yield ProxyContentDeltaEvent(delta)
            pending = None
            response = response_holder.get("response")
            tool_calls = response.tool_calls if response is not None else ()
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
                state.mode = "orchestrator"
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
                result_holder: dict[str, object] = {}
                async for internal_event in self._execute_internal_action(
                    request=request,
                    action=action,
                    state=state,
                    worklog=tuple(worklog),
                    result_holder=result_holder,
                ):
                    yield internal_event
                if state.finish_reason == "tool_calls":
                    return
                if result_holder:
                    worklog.extend(_result(result_holder))
                else:
                    return
                continue

            if _requires_host_tool_result(request):
                raise ValueError("model ignored required host tool choice")
            state.mode = "orchestrator"
            return

        raise ValueError("orchestrator exceeded internal move limit")

    async def _execute_internal_action(
        self,
        *,
        request: ProxyTurnRequest,
        action: Any,
        state: ProxyTurnState,
        worklog: tuple[str, ...],
        result_holder: dict[str, object],
    ) -> AsyncIterator[ProxyEvent]:
        if isinstance(action, MessageWorkroomAction):
            state.mode = "workroom"
            if action.preset is None and not action.participants:
                state.finish_reason = "error"
                error_text = "message_workroom needs a preset or participants target."
                state.content = error_text
                yield ProxyContentDeltaEvent(error_text)
                return
            async for event in self._message_workroom(
                request=request,
                workroom_name=action.preset,
                participants=action.participants,
                workroom_message=action.message,
                state=state,
                result_sink=partial(_store_result, result_holder),
                worklog=worklog,
            ):
                yield event
            return
        raise ValueError(f"unsupported internal action: {action}")

    async def _resume_subtask(
        self,
        *,
        request: ProxyTurnRequest,
        pending: PendingContinuation,
        state: ProxyTurnState,
        worklog: tuple[str, ...],
        result_holder: dict[str, object],
    ) -> AsyncIterator[ProxyEvent]:
        continuation = pending.state
        if continuation.workroom_name is not None:
            state.mode = "workroom"
            async for event in self._message_workroom(
                request=request,
                continuation=continuation,
                pending=pending,
                state=state,
                result_sink=partial(_store_result, result_holder),
                worklog=worklog,
            ):
                yield event
            return
        raise ValueError("unsupported continuation state")

    async def _message_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_name: str | None = None,
        participants: tuple[str, ...] = (),
        workroom_message: str | None = None,
        state: ProxyTurnState,
        result_sink: Any,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        worklog: tuple[str, ...] = (),
    ) -> AsyncIterator[ProxyEvent]:
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
                yield ProxyContentDeltaEvent(error_text)
                return
            workroom_name, participants = resolved
            intro = _workroom_notice(_workroom_intro(workroom_name))
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._workroom_executor.execute(
            request=request,
            workroom_name=workroom_name,
            participants=participants,
            workroom_message=workroom_message,
            state=state,
            continuation=continuation,
            pending=pending,
            result_sink=result_sink,
            worklog=worklog,
        ):
            yield event


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


def _result(holder: dict[str, object]) -> tuple[str, ...]:
    value = holder.get("result")
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    return ()


def _store_response(holder: dict[str, Any], value: object) -> None:
    holder["response"] = value


def _store_result(holder: dict[str, object], result: tuple[str, ...]) -> None:
    holder["result"] = result


def _resolve_workroom_target(
    *,
    registry: RuntimeRegistry,
    preset: str | None,
    participants: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    if preset is not None:
        definition = registry.workroom_definitions.get(preset)
        if definition is None:
            return None
        resolved_participants = participants or workroom_participants_for_definition(
            definition
        )
        return preset, resolved_participants
    if participants:
        return "ad hoc", participants
    return None


def _workroom_notice(base: str) -> str:
    return base + "\n"


def _workroom_intro(workroom_name: str) -> str:
    if workroom_name == "ad hoc":
        return "Orchestrator: opening an ad hoc workroom."
    return f"Orchestrator: opening workroom {workroom_name}."
