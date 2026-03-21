from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agent_framework import ResponseStream

from ergon_studio.agent_factory import build_agent
from ergon_studio.proxy.agent_runner import ProxyAgentRunner
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    latest_pending_continuation,
)
from ergon_studio.proxy.discussion_workroom_executor import (
    ProxyDiscussionWorkroomExecutor,
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
    ContinueWorkroomAction,
    MessageSpecialistAction,
    OpenWorkroomAction,
    build_orchestrator_internal_tools,
    is_internal_tool_name,
    parse_internal_action,
)
from ergon_studio.proxy.prompts import orchestrator_turn_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.staged_workroom_executor import ProxyStagedWorkroomExecutor
from ergon_studio.proxy.tool_call_emitter import ProxyToolCallEmitter
from ergon_studio.proxy.tool_passthrough import extract_tool_calls
from ergon_studio.proxy.turn_executor import ProxyTurnExecutor
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workroom_dispatcher import ProxyWorkroomDispatcher
from ergon_studio.registry import RuntimeRegistry

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
        agent_builder: Any = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_runner = ProxyAgentRunner(
            registry,
            agent_builder=agent_builder,
        )
        self._tool_call_emitter = ProxyToolCallEmitter(self._agent_runner)
        self._turn_executor = ProxyTurnExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
        )
        staged_workroom_executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
        )
        discussion_workroom_executor = ProxyDiscussionWorkroomExecutor(
            stream_text_agent=self._agent_runner.stream_text_agent,
            emit_tool_calls=self._tool_call_emitter.emit_tool_calls,
        )
        self._workroom_dispatcher = ProxyWorkroomDispatcher(
            registry,
            execute_staged_workroom=staged_workroom_executor.execute,
            execute_discussion_workroom=discussion_workroom_executor.execute,
        )

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
                    if pending.state.mode in {"delegate", "workroom"}:
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
            internal_tools = build_orchestrator_internal_tools(
                self.registry,
                has_active_workroom=loop_state.workroom_progress is not None,
            )
            async for delta in self._agent_runner.stream_text_agent(
                agent_id="orchestrator",
                prompt=orchestrator_turn_prompt(
                    request,
                    goal=loop_state.goal,
                    current_brief=loop_state.current_brief,
                    worklog=loop_state.worklog,
                    active_workroom_id=(
                        loop_state.workroom_progress.workroom_id
                        if loop_state.workroom_progress is not None
                        else None
                    ),
                    active_workroom_request=(
                        loop_state.workroom_progress.workroom_request
                        if loop_state.workroom_progress is not None
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
                final_response_sink=response_holder_sink(response_holder),
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
                    if (
                        isinstance(action, MessageSpecialistAction)
                        and result.workroom_progress is None
                        and loop_state.workroom_progress is not None
                    ):
                        result = ProxyMoveResult(
                            worklog_lines=result.worklog_lines,
                            current_brief=result.current_brief,
                            workroom_progress=loop_state.workroom_progress,
                        )
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
        if isinstance(action, MessageSpecialistAction):
            state.mode = "delegate"
            async for event in self._turn_executor.execute_delegation(
                request=request,
                agent_id=action.agent_id,
                message=action.message,
                state=state,
                current_brief=loop_state.current_brief,
                result_sink=_result_sink(result_holder),
                loop_state=loop_state,
            ):
                yield event
            return
        if isinstance(action, OpenWorkroomAction):
            state.mode = "workroom"
            async for event in self._workroom_dispatcher.execute_workroom(
                request=request,
                workroom_id=action.workroom_id,
                participants=action.participants,
                workroom_request=action.message,
                goal=loop_state.goal,
                state=state,
                result_sink=_result_sink(result_holder),
                loop_state=loop_state,
            ):
                yield event
            return
        if isinstance(action, ContinueWorkroomAction):
            state.mode = "workroom"
            active_workroom = loop_state.workroom_progress
            if active_workroom is None:
                state.finish_reason = "error"
                error_text = "No active workroom is available to continue."
                state.content = error_text
                yield ProxyContentDeltaEvent(error_text)
                return
            async for event in self._workroom_dispatcher.execute_workroom_continuation(
                request=request,
                continuation=_update_workroom_request(
                    active_workroom,
                    message=action.message,
                ),
                pending=None,
                state=state,
                result_sink=_result_sink(result_holder),
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
        if continuation.mode == "delegate":
            state.mode = "delegate"
            async for event in self._turn_executor.execute_delegation(
                request=request,
                agent_id=continuation.agent_id,
                message=continuation.message or request.latest_user_text() or "",
                state=state,
                current_brief=continuation.current_brief,
                pending=pending,
                result_sink=_result_sink(result_holder),
                loop_state=loop_state,
            ):
                yield event
            return
        if continuation.mode == "workroom":
            state.mode = "workroom"
            async for event in (
                self._workroom_dispatcher.execute_workroom_continuation(
                    request=request,
                    continuation=continuation,
                    pending=pending,
                    state=state,
                    result_sink=_result_sink(result_holder),
                    loop_state=loop_state,
                )
            ):
                yield event
            return
        raise ValueError(f"unsupported continuation mode: {continuation.mode}")

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
        workroom_progress = (
            continuation if continuation.workroom_id is not None else None
        )
        return ProxyDecisionLoopState(
            goal=goal,
            current_brief=current_brief,
            worklog=continuation.worklog,
            workroom_progress=workroom_progress,
            current_workroom_request=continuation.workroom_request,
        )


def _orchestrator_continuation_state(
    loop_state: ProxyDecisionLoopState,
) -> ContinuationState:
    workroom_progress = loop_state.workroom_progress
    return ContinuationState(
        mode="orchestrator",
        agent_id="orchestrator",
        workroom_id=(
            workroom_progress.workroom_id if workroom_progress is not None else None
        ),
        workroom_participants=(
            workroom_progress.workroom_participants
            if workroom_progress is not None
            else ()
        ),
        workroom_request=(
            workroom_progress.workroom_request
            if workroom_progress is not None
            else loop_state.current_workroom_request
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


def _result_sink(holder: dict[str, object]) -> Any:
    def _capture(result: ProxyMoveResult) -> None:
        holder["result"] = result

    return _capture


def _update_workroom_request(
    continuation: ContinuationState,
    *,
    message: str,
) -> ContinuationState:
    if message == continuation.workroom_request:
        return continuation
    return ContinuationState(
        mode=continuation.mode,
        agent_id=continuation.agent_id,
        participant_label=continuation.participant_label,
        workroom_id=continuation.workroom_id,
        workroom_participants=continuation.workroom_participants,
        workroom_request=message,
        last_stage_outputs=continuation.last_stage_outputs,
        last_stage_parallel_attempts=continuation.last_stage_parallel_attempts,
        progress_index=continuation.progress_index,
        member_index=continuation.member_index,
        message=continuation.message,
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
        workroom_progress=loop_state.workroom_progress,
    )
