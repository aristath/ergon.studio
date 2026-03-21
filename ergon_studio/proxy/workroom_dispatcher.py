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
from ergon_studio.proxy.workroom import AD_HOC_WORKROOM_ID, is_ad_hoc_workroom
from ergon_studio.proxy.workroom_metadata import (
    workroom_shape_for_definition,
)
from ergon_studio.registry import RuntimeRegistry

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)

WorkroomHandler = Callable[
    ...,
    AsyncIterator[ProxyEvent],
]


class ProxyWorkroomDispatcher:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        execute_grouped_workroom: WorkroomHandler,
        execute_group_chat_workroom: WorkroomHandler,
        execute_magentic_workroom: WorkroomHandler,
        execute_handoff_workroom: WorkroomHandler,
    ) -> None:
        self.registry = registry
        self._execute_grouped_workroom = execute_grouped_workroom
        self._execute_group_chat_workroom = execute_group_chat_workroom
        self._execute_magentic_workroom = execute_magentic_workroom
        self._execute_handoff_workroom = execute_handoff_workroom

    async def execute_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_id: str | None,
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workroom_request: str | None = None,
        goal: str,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self._resolve_workroom_definition(
            workroom_id=workroom_id,
            specialists=specialists,
            specialist_counts=specialist_counts,
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workroom: {workroom_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = _workroom_notice(
            base=_workroom_intro(definition),
            loop_state=loop_state,
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workroom(
            request=request,
            definition=definition,
            goal=goal,
            specialists=specialists,
            specialist_counts=specialist_counts,
            workroom_request=workroom_request,
            state=state,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def execute_workroom_continuation(
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
            workroom_id=continuation.workroom_id,
            specialists=continuation.workroom_specialists,
            specialist_counts=continuation.workroom_specialist_counts,
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = (
                f"Unknown workroom: {continuation.workroom_id or '(none)'}"
            )
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        agent_name = continuation.agent_id or "(unknown)"
        intro = _workroom_notice(
            base=(
                f"Orchestrator: continuing workroom {definition.id} with "
                f"{agent_name}."
            ),
            loop_state=loop_state,
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workroom(
            request=request,
            definition=definition,
            goal=continuation.goal or request.latest_user_text() or "",
            specialists=continuation.workroom_specialists,
            specialist_counts=continuation.workroom_specialist_counts,
            workroom_request=continuation.workroom_request,
            state=state,
            continuation=continuation,
            pending=pending,
            result_sink=result_sink,
            loop_state=loop_state,
        ):
            yield event

    async def _dispatch_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        specialists: tuple[str, ...] = (),
        specialist_counts: tuple[tuple[str, int], ...] = (),
        workroom_request: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        shape = workroom_shape_for_definition(definition)
        if shape in {"sequential", "grouped", "concurrent"}:
            async for event in self._execute_grouped_workroom(
                request=request,
                definition=definition,
                goal=goal,
                specialists=specialists,
                specialist_counts=specialist_counts,
                workroom_request=workroom_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if shape == "group_chat":
            async for event in self._execute_group_chat_workroom(
                request=request,
                definition=definition,
                goal=goal,
                specialists=specialists,
                specialist_counts=specialist_counts,
                workroom_request=workroom_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if shape == "magentic":
            async for event in self._execute_magentic_workroom(
                request=request,
                definition=definition,
                goal=goal,
                specialists=specialists,
                specialist_counts=specialist_counts,
                workroom_request=workroom_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if shape == "handoff":
            async for event in self._execute_handoff_workroom(
                request=request,
                definition=definition,
                goal=goal,
                specialists=specialists,
                specialist_counts=specialist_counts,
                workroom_request=workroom_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        raise ValueError(f"unsupported workroom shape: {shape}")

    def _resolve_workroom_definition(
        self,
        *,
        workroom_id: str | None,
        specialists: tuple[str, ...],
        specialist_counts: tuple[tuple[str, int], ...],
    ) -> DefinitionDocument | None:
        if is_ad_hoc_workroom(workroom_id):
            if not specialists:
                return None
            return _ad_hoc_workroom_definition(
                specialists=specialists,
                specialist_counts=specialist_counts,
            )
        return self.registry.workroom_definitions.get(workroom_id or "")


def _workroom_notice(
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
    return f"Orchestrator: opening workroom {definition.id}."


def _ad_hoc_workroom_definition(
    *,
    specialists: tuple[str, ...],
    specialist_counts: tuple[tuple[str, int], ...],
) -> DefinitionDocument:
    count_map = {agent_id: count for agent_id, count in specialist_counts}
    expanded_staffing = tuple(
        agent_id
        for agent_id in specialists
        for _ in range(count_map.get(agent_id, 1))
    )
    if not expanded_staffing:
        expanded_staffing = specialists
    unique_roles = {agent_id for agent_id in expanded_staffing}
    if len(expanded_staffing) > 1 and len(unique_roles) == 1:
        shape = "grouped"
        steps = list(expanded_staffing)
        max_rounds = len(expanded_staffing)
    else:
        shape = "group_chat"
        steps = list(specialists)
        max_rounds = max(len(expanded_staffing), 2)
    return DefinitionDocument(
        id=AD_HOC_WORKROOM_ID,
        path=Path(AD_HOC_WORKROOM_ID),
        metadata={
            "id": AD_HOC_WORKROOM_ID,
            "name": "Ad Hoc Workroom",
            "shape": shape,
            "steps": steps,
            "max_rounds": max_rounds,
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
