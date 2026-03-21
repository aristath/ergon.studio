from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)
from ergon_studio.proxy.workflow_metadata import workflow_orchestration_for_definition
from ergon_studio.registry import RuntimeRegistry

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)

WorkflowHandler = Callable[
    ...,
    AsyncIterator[ProxyEvent],
]


class ProxyWorkflowDispatcher:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        execute_grouped_workflow: WorkflowHandler,
        execute_group_chat_workflow: WorkflowHandler,
        execute_magentic_workflow: WorkflowHandler,
        execute_handoff_workflow: WorkflowHandler,
    ) -> None:
        self.registry = registry
        self._execute_grouped_workflow = execute_grouped_workflow
        self._execute_group_chat_workflow = execute_group_chat_workflow
        self._execute_magentic_workflow = execute_magentic_workflow
        self._execute_handoff_workflow = execute_handoff_workflow

    async def execute_workflow(
        self,
        *,
        request: ProxyTurnRequest,
        workflow_id: str | None,
        goal: str,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self.registry.workflow_definitions.get(workflow_id or "")
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workflow: {workflow_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = f"Orchestrator: running workflow {definition.id}.\n"
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workflow(
            request=request,
            definition=definition,
            goal=goal,
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def execute_workflow_continuation(
        self,
        *,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        pending: PendingContinuation | None,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self.registry.workflow_definitions.get(
            continuation.workflow_id or ""
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workflow: {continuation.workflow_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        agent_name = continuation.agent_id or "(unknown)"
        intro = (
            f"Orchestrator: continuing workflow {definition.id} with {agent_name}.\n"
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workflow(
            request=request,
            definition=definition,
            goal=continuation.goal or request.latest_user_text() or "",
            state=state,
            continuation=continuation,
            pending=pending,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def _dispatch_workflow(
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
        orchestration = workflow_orchestration_for_definition(definition)
        if orchestration in {"sequential", "grouped", "concurrent"}:
            async for event in self._execute_grouped_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if orchestration == "group_chat":
            async for event in self._execute_group_chat_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if orchestration == "magentic":
            async for event in self._execute_magentic_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if orchestration == "handoff":
            async for event in self._execute_handoff_workflow(
                request=request,
                definition=definition,
                goal=goal,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        raise ValueError(f"unsupported workflow orchestration: {orchestration}")
