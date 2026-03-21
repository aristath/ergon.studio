from __future__ import annotations

import time
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_framework import ResponseStream

from ergon_studio.agent_factory import build_agent
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import ProxyAgentRunner
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
    parse_internal_action,
)
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.tool_call_emitter import ProxyToolCallEmitter
from ergon_studio.proxy.tool_passthrough import extract_tool_calls
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workroom_executor import ProxyWorkroomExecutor
from ergon_studio.registry import RuntimeRegistry

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)

AD_HOC_WORKROOM_ID = "ad-hoc-workroom"


class ProxyOrchestrationCore:
    _MAX_INTERNAL_MOVES = 8

    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Any = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_runner = ProxyAgentRunner(
            registry,
            agent_builder=agent_builder,
        )
        self._tool_call_emitter = ProxyToolCallEmitter(self._agent_runner)
        workroom_executor = ProxyWorkroomExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
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
                loop_state = self._initial_loop_state(request, pending=pending)
                if pending is not None:
                    if pending.state.mode == "workroom":
                        result_holder: dict[str, object] = {}
                        async for event in self._resume_subtask(
                            request=request,
                            pending=pending,
                            state=state,
                            loop_state=loop_state,
                            result_holder=result_holder,
                        ):
                            yield event
                        if state.finish_reason == "tool_calls":
                            return
                        if result_holder:
                            loop_state.absorb_result(
                                result=_result(result_holder, loop_state)
                            )
                        async for event in self._run_orchestrator_loop(
                            request=request,
                            state=state,
                            loop_state=loop_state,
                        ):
                            yield event
                        return
                    async for event in self._run_orchestrator_loop(
                        request=request,
                        state=state,
                        loop_state=loop_state,
                        pending_orchestrator=pending,
                    ):
                        yield event
                    return

                async for event in self._run_orchestrator_loop(
                    request=request,
                    state=state,
                    loop_state=loop_state,
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
        loop_state: ProxyDecisionLoopState,
        pending_orchestrator: PendingContinuation | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        pending = pending_orchestrator
        for _ in range(self._MAX_INTERNAL_MOVES):
            response_holder: dict[str, Any] = {}
            buffered_deltas: list[str] = []
            internal_tools = build_orchestrator_internal_tools(self.registry)
            async for delta in self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
                    goal=loop_state.goal,
                    current_brief=loop_state.current_brief,
                    worklog=loop_state.worklog,
                    active_workroom_id=(
                        loop_state.active_workroom.workroom_id
                        if loop_state.active_workroom is not None
                        else None
                    ),
                    active_workroom_participants=(
                        loop_state.active_workroom.workroom_participants
                        if loop_state.active_workroom is not None
                        else ()
                    ),
                    active_workroom_message=(
                        loop_state.active_workroom.workroom_message
                        if loop_state.active_workroom is not None
                        else None
                    ),
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
                buffered_deltas.append(delta)
            pending = None
            response = response_holder.get("response")
            tool_calls = (
                extract_tool_calls(response)
                if response is not None
                else ()
            )
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
                for tool_event in self._tool_call_emitter.emit_tool_calls(
                    tool_calls=host_tool_calls,
                    request=request,
                    continuation=_orchestrator_continuation_state(loop_state),
                    state=state,
                ):
                    yield tool_event
                return
            if internal_tool_calls:
                action = parse_internal_action(
                    internal_tool_calls[0],
                    registry=self.registry,
                )
                result_holder: dict[str, object] = {}
                async for internal_event in self._execute_internal_action(
                    request=request,
                    action=action,
                    state=state,
                    loop_state=loop_state,
                    result_holder=result_holder,
                ):
                    yield internal_event
                if state.finish_reason == "tool_calls":
                    return
                if result_holder:
                    result = _result(result_holder, loop_state)
                    loop_state.absorb_result(result=result)
                else:
                    return
                continue

            if _requires_host_tool_result(request):
                raise ValueError("model ignored required host tool choice")
            state.mode = "orchestrator"
            if not buffered_deltas and response is not None:
                final_text = getattr(response, "text", "")
                if final_text:
                    buffered_deltas.append(final_text)
            for delta in buffered_deltas:
                state.append_content(delta)
                yield ProxyContentDeltaEvent(delta)
            return

        raise ValueError("orchestrator exceeded internal move limit")

    async def _execute_internal_action(
        self,
        *,
        request: ProxyTurnRequest,
        action: Any,
        state: ProxyTurnState,
        loop_state: ProxyDecisionLoopState,
        result_holder: dict[str, object],
    ) -> AsyncIterator[ProxyEvent]:
        if isinstance(action, MessageWorkroomAction):
            state.mode = "workroom"
            active_workroom = loop_state.active_workroom
            if (
                active_workroom is None
                and action.workroom_id is None
                and not action.participants
            ):
                state.finish_reason = "error"
                error_text = "message_workroom needs an active room or a room target."
                state.content = error_text
                yield ProxyContentDeltaEvent(error_text)
                return
            if _should_continue_active_workroom(
                action=action,
                active_workroom=active_workroom,
            ):
                if active_workroom is None:
                    state.finish_reason = "error"
                    error_text = "No active workroom is available to message."
                    state.content = error_text
                    yield ProxyContentDeltaEvent(error_text)
                    return
                continuation = _update_workroom_message(
                    active_workroom,
                    participants=action.participants,
                    message=action.message,
                )
                async for event in self._message_workroom(
                    request=request,
                    continuation=continuation,
                    pending=None,
                    state=state,
                    result_sink=partial(_store_result, result_holder),
                    loop_state=loop_state,
                ):
                    yield event
                return
            async for event in self._message_workroom(
                request=request,
                workroom_id=action.workroom_id,
                participants=action.participants,
                workroom_message=action.message,
                goal=loop_state.goal,
                state=state,
                result_sink=partial(_store_result, result_holder),
                loop_state=loop_state,
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
        loop_state: ProxyDecisionLoopState,
        result_holder: dict[str, object],
    ) -> AsyncIterator[ProxyEvent]:
        continuation = pending.state
        if continuation.mode == "workroom":
            state.mode = "workroom"
            async for event in self._message_workroom(
                request=request,
                continuation=continuation,
                pending=pending,
                state=state,
                result_sink=partial(_store_result, result_holder),
                loop_state=loop_state,
            ):
                yield event
            return
        raise ValueError(f"unsupported continuation mode: {continuation.mode}")

    async def _message_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_id: str | None = None,
        participants: tuple[str, ...] = (),
        workroom_message: str | None = None,
        goal: str | None = None,
        state: ProxyTurnState,
        result_sink: Any,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        if continuation is not None:
            definition = _resolve_workroom_definition(
                registry=self.registry,
                workroom_id=continuation.workroom_id,
                participants=continuation.workroom_participants,
            )
            if definition is None:
                error_text = (
                    f"Unknown workroom: {continuation.workroom_id or '(none)'}"
                )
                state.finish_reason = "error"
                state.content = error_text
                yield ProxyContentDeltaEvent(error_text)
                return
            intro = _workroom_notice(
                f"Orchestrator: continuing workroom {definition.id} with "
                f"{continuation.agent_id or '(unknown)'}."
            )
            goal = continuation.goal or request.latest_user_text() or ""
            participants = continuation.workroom_participants
            workroom_message = continuation.workroom_message
        else:
            definition = _resolve_workroom_definition(
                registry=self.registry,
                workroom_id=workroom_id,
                participants=participants,
            )
            if definition is None:
                error_text = f"Unknown workroom: {workroom_id or '(none)'}"
                state.finish_reason = "error"
                state.content = error_text
                yield ProxyContentDeltaEvent(error_text)
                return
            intro = _workroom_notice(_workroom_intro(definition))
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._workroom_executor.execute(
            request=request,
            definition=definition,
            goal=goal or request.latest_user_text() or "",
            participants=participants,
            workroom_message=workroom_message,
            state=state,
            continuation=continuation,
            pending=pending,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    def _initial_loop_state(
        self,
        request: ProxyTurnRequest,
        *,
        pending: PendingContinuation | None,
    ) -> ProxyDecisionLoopState:
        if pending is None:
            goal = request.latest_user_text() or ""
            return ProxyDecisionLoopState(
                goal=goal,
                current_brief=goal,
            )
        continuation = pending.state
        goal = continuation.goal or request.latest_user_text() or ""
        current_brief = continuation.current_brief or goal
        active_workroom = (
            continuation if continuation.workroom_id is not None else None
        )
        return ProxyDecisionLoopState(
            goal=goal,
            current_brief=current_brief,
            worklog=continuation.worklog,
            active_workroom=active_workroom,
        )


def _orchestrator_continuation_state(
    loop_state: ProxyDecisionLoopState,
) -> ContinuationState:
    active_workroom = loop_state.active_workroom
    return ContinuationState(
        mode="orchestrator",
        agent_id="orchestrator",
        workroom_id=(
            active_workroom.workroom_id if active_workroom is not None else None
        ),
        workroom_participants=(
            active_workroom.workroom_participants
            if active_workroom is not None
            else ()
        ),
        workroom_message=(
            active_workroom.workroom_message
            if active_workroom is not None
            else None
        ),
        goal=loop_state.goal,
        current_brief=loop_state.current_brief,
        worklog=loop_state.worklog,
    )


def _requires_host_tool_result(request: ProxyTurnRequest) -> bool:
    tool_choice = request.tool_choice
    if tool_choice == "required":
        return True
    return isinstance(tool_choice, dict)


def _update_workroom_message(
    continuation: ContinuationState,
    *,
    participants: tuple[str, ...],
    message: str,
) -> ContinuationState:
    next_participants = participants or continuation.workroom_participants
    if (
        message == continuation.workroom_message
        and next_participants == continuation.workroom_participants
    ):
        return continuation
    return ContinuationState(
        mode=continuation.mode,
        agent_id=continuation.agent_id,
        participant_label=continuation.participant_label,
        workroom_id=continuation.workroom_id,
        workroom_participants=next_participants,
        workroom_message=message,
        member_index=continuation.member_index,
        goal=continuation.goal,
        current_brief=continuation.current_brief,
        worklog=continuation.worklog,
        workroom_outputs=continuation.workroom_outputs,
    )


def _result(
    holder: dict[str, object],
    loop_state: ProxyDecisionLoopState,
) -> ProxyMoveResult:
    value = holder.get("result")
    if isinstance(value, ProxyMoveResult):
        return value
    return ProxyMoveResult(
        worklog_lines=(),
        current_brief=loop_state.current_brief,
        active_workroom=loop_state.active_workroom,
    )


def _should_continue_active_workroom(
    *,
    action: MessageWorkroomAction,
    active_workroom: ContinuationState | None,
) -> bool:
    if active_workroom is None:
        return False
    if action.workroom_id is None:
        return True
    return action.workroom_id == active_workroom.workroom_id


def _store_response(holder: dict[str, Any], value: object) -> None:
    holder["response"] = value


def _store_result(holder: dict[str, object], result: ProxyMoveResult) -> None:
    holder["result"] = result


def _resolve_workroom_definition(
    *,
    registry: RuntimeRegistry,
    workroom_id: str | None,
    participants: tuple[str, ...],
) -> DefinitionDocument | None:
    if workroom_id is None and participants:
        return _ad_hoc_workroom_definition(participants=participants)
    if _is_ad_hoc_workroom(workroom_id):
        if not participants:
            return None
        return _ad_hoc_workroom_definition(participants=participants)
    return registry.workroom_definitions.get(workroom_id or "")


def _workroom_notice(base: str) -> str:
    return base + "\n"


def _workroom_intro(definition: DefinitionDocument) -> str:
    if _is_ad_hoc_workroom(definition.id):
        return "Orchestrator: opening an ad hoc workroom."
    return f"Orchestrator: opening workroom {definition.id}."


def _ad_hoc_workroom_definition(
    *,
    participants: tuple[str, ...],
) -> DefinitionDocument:
    return DefinitionDocument(
        id=AD_HOC_WORKROOM_ID,
        path=Path(AD_HOC_WORKROOM_ID),
        metadata={
            "id": AD_HOC_WORKROOM_ID,
            "name": "Ad Hoc Workroom",
            "participants": list(participants),
        },
        body=(
            "## Purpose\n"
            "A temporary staffed workroom opened by the lead developer for "
            "natural-language collaboration."
        ),
        sections={
            "Purpose": (
                "A temporary staffed workroom opened by the lead developer for "
                "natural-language collaboration."
            )
        },
    )


def _is_ad_hoc_workroom(workroom_id: str | None) -> bool:
    return workroom_id == AD_HOC_WORKROOM_ID
