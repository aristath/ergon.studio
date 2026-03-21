from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import summarize_conversation
from ergon_studio.proxy.prompts import workflow_step_prompt
from ergon_studio.proxy.response_sink import response_holder_sink
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyGroupedWorkflowExecutor:
    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., Any],
        emit_tool_calls: Callable[..., list[ProxyToolCallEvent]],
        emit_workflow_summary: Callable[..., AsyncIterator[ProxyEvent]],
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._emit_tool_calls = emit_tool_calls
        self._emit_workflow_summary = emit_workflow_summary

    async def execute(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        step_groups = workflow_step_groups_for_definition(definition)
        start_index = (
            continuation.step_index
            if continuation and continuation.step_index is not None
            else 0
        )
        start_agent_index = (
            continuation.agent_index
            if continuation and continuation.agent_index is not None
            else 0
        )
        current_brief = (
            continuation.current_brief
            if continuation and continuation.current_brief is not None
            else goal
        )
        workflow_outputs: list[str] = (
            list(continuation.workflow_outputs) if continuation is not None else []
        )
        for step_index in range(start_index, len(step_groups)):
            group = step_groups[step_index]
            group_start_index = start_agent_index if step_index == start_index else 0
            for agent_index in range(group_start_index, len(group)):
                agent_id = group[agent_index]
                prompt = workflow_step_prompt(
                    workflow_id=definition.id,
                    agent_id=agent_id,
                    goal=goal,
                    current_brief=current_brief,
                    transcript_summary=summarize_conversation(request.messages),
                    prior_outputs=tuple(workflow_outputs),
                )
                agent_text = ""
                first = True
                response_holder: dict[str, Any] = {}
                async for delta in self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=prompt,
                    session_id=f"proxy-workflow-{definition.id}-{agent_id}-{uuid4().hex}",
                    model_id_override=request.model,
                    host_tools=request.tools,
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    pending_continuation=pending
                    if step_index == start_index and agent_index == group_start_index
                    else None,
                    final_response_sink=response_holder_sink(response_holder),
                ):
                    agent_text += delta
                    reasoning_delta = f"{agent_id}: {delta}" if first else delta
                    first = False
                    state.append_reasoning(reasoning_delta)
                    yield ProxyReasoningDeltaEvent(reasoning_delta)
                response = response_holder.get("response")
                if response is not None:
                    emitted = self._emit_tool_calls(
                        response=response,
                        request=request,
                        continuation=ContinuationState(
                            mode="workflow",
                            workflow_id=definition.id,
                            step_index=step_index,
                            agent_index=agent_index,
                            agent_id=agent_id,
                            goal=goal,
                            current_brief=agent_text.strip() or current_brief,
                            decision_history=(
                                loop_state.worklog if loop_state is not None else ()
                            ),
                            workflow_outputs=tuple(workflow_outputs),
                        ),
                        state=state,
                    )
                    if emitted:
                        for event in emitted:
                            yield event
                        return
                workflow_outputs.append(f"{agent_id}: {agent_text.strip()}")
                current_brief = agent_text.strip() or current_brief
            if result_sink is not None:
                next_step_index = step_index + 1
                workflow_progress = None
                if next_step_index < len(step_groups):
                    next_group = step_groups[next_step_index]
                    workflow_progress = ContinuationState(
                        mode="workflow",
                        workflow_id=definition.id,
                        step_index=next_step_index,
                        agent_index=0,
                        agent_id=next_group[0],
                        goal=goal,
                        current_brief=current_brief,
                        decision_history=(
                            loop_state.worklog if loop_state is not None else ()
                        ),
                        workflow_outputs=tuple(workflow_outputs),
                    )
                result_sink(
                    ProxyMoveResult(
                        worklog_lines=tuple(
                            workflow_outputs[-len(group) :] if group else ()
                        ),
                        current_brief=current_brief,
                        workflow_progress=workflow_progress,
                    )
                )
                return
        async for summary_event in self._emit_workflow_summary(
            request=request,
            definition=definition,
            goal=goal,
            current_brief=current_brief,
            workflow_outputs=tuple(workflow_outputs),
            state=state,
        ):
            yield summary_event
