from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

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
from ergon_studio.proxy.workflow_metadata import (
    workflow_orchestration_for_definition,
)
from ergon_studio.proxy.workroom import AD_HOC_WORKROOM_ID, is_ad_hoc_workroom
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
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workflow_request: str | None = None,
        goal: str,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self._resolve_workroom_definition(
            workflow_id=workflow_id,
            specialists=specialists,
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workroom: {workflow_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = _workflow_notice(
            base=_workroom_intro(definition),
            loop_state=loop_state,
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workflow(
            request=request,
            definition=definition,
            goal=goal,
            specialists=specialists,
            specialist_counts=specialist_counts,
            workflow_request=workflow_request,
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
        definition = self._resolve_workroom_definition(
            workflow_id=continuation.workflow_id,
            specialists=continuation.workflow_specialists,
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = (
                f"Unknown workroom: {continuation.workflow_id or '(none)'}"
            )
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        agent_name = continuation.agent_id or "(unknown)"
        intro = _workflow_notice(
            base=(
                f"Orchestrator: continuing workroom {definition.id} with "
                f"{agent_name}."
            ),
            loop_state=loop_state,
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workflow(
            request=request,
            definition=definition,
            goal=continuation.goal or request.latest_user_text() or "",
            specialists=continuation.workflow_specialists,
            specialist_counts=continuation.workflow_specialist_counts,
            workflow_request=continuation.workflow_request,
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
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workflow_request: str | None = None,
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
                specialists=specialists,
                specialist_counts=specialist_counts,
                workflow_request=workflow_request,
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
                specialists=specialists,
                specialist_counts=specialist_counts,
                workflow_request=workflow_request,
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
                specialists=specialists,
                specialist_counts=specialist_counts,
                workflow_request=workflow_request,
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
                specialists=specialists,
                specialist_counts=specialist_counts,
                workflow_request=workflow_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        raise ValueError(f"unsupported workflow orchestration: {orchestration}")

    def _resolve_workroom_definition(
        self,
        *,
        workflow_id: str | None,
        specialists: tuple[str, ...],
    ) -> DefinitionDocument | None:
        if is_ad_hoc_workroom(workflow_id):
            if not specialists:
                return None
            return _ad_hoc_workroom_definition(specialists)
        return self.registry.workflow_definitions.get(workflow_id or "")


def _workflow_notice(
    *,
    base: str,
    loop_state: ProxyDecisionLoopState | None,
) -> str:
    lines = [base]
    if loop_state is not None and loop_state.current_move_rationale:
        lines.append(f"Why: {loop_state.current_move_rationale}")
    return "\n".join(lines) + "\n"


def _workroom_intro(definition: DefinitionDocument) -> str:
    if is_ad_hoc_workroom(definition.id):
        return "Orchestrator: opening an ad hoc workroom."
    return f"Orchestrator: opening workroom template {definition.id}."


def _ad_hoc_workroom_definition(
    specialists: tuple[str, ...],
) -> DefinitionDocument:
    return DefinitionDocument(
        id=AD_HOC_WORKROOM_ID,
        path=Path(AD_HOC_WORKROOM_ID),
        metadata={
            "id": AD_HOC_WORKROOM_ID,
            "name": "Ad Hoc Workroom",
            "orchestration": "magentic",
            "steps": list(specialists),
            "max_rounds": max(len(specialists) * 2, 2),
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
