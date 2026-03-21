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
        execute_staged_workroom: WorkroomHandler,
        execute_discussion_workroom: WorkroomHandler,
    ) -> None:
        self.registry = registry
        self._execute_staged_workroom = execute_staged_workroom
        self._execute_discussion_workroom = execute_discussion_workroom

    async def execute_workroom(
        self,
        *,
        request: ProxyTurnRequest,
        workroom_id: str | None,
        participants: tuple[str, ...] = (),
        workroom_request: str | None = None,
        goal: str,
        state: ProxyTurnState,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        definition = self._resolve_workroom_definition(
            workroom_id=workroom_id,
            participants=participants,
        )
        if definition is None:
            state.finish_reason = "error"
            error_text = f"Unknown workroom: {workroom_id or '(none)'}"
            state.content = error_text
            yield ProxyContentDeltaEvent(error_text)
            return
        intro = _workroom_notice(
            _workroom_intro(definition),
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workroom(
            request=request,
            definition=definition,
            goal=goal,
            participants=participants,
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
            participants=continuation.workroom_participants,
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
            (
                f"Orchestrator: continuing workroom {definition.id} with "
                f"{agent_name}."
            ),
        )
        state.append_reasoning(intro)
        yield ProxyReasoningDeltaEvent(intro)
        async for event in self._dispatch_workroom(
            request=request,
            definition=definition,
            goal=continuation.goal or request.latest_user_text() or "",
            participants=continuation.workroom_participants,
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
        participants: tuple[str, ...] = (),
        workroom_request: str | None = None,
        state: ProxyTurnState,
        continuation: ContinuationState | None = None,
        pending: PendingContinuation | None = None,
        result_sink: Callable[[ProxyMoveResult], None] | None = None,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> AsyncIterator[ProxyEvent]:
        shape = workroom_shape_for_definition(definition)
        if shape == "staged":
            async for event in self._execute_staged_workroom(
                request=request,
                definition=definition,
                goal=goal,
                participants=participants,
                workroom_request=workroom_request,
                state=state,
                continuation=continuation,
                pending=pending,
                result_sink=result_sink,
                loop_state=loop_state,
            ):
                yield event
            return
        if shape == "discussion":
            async for event in self._execute_discussion_workroom(
                request=request,
                definition=definition,
                goal=goal,
                participants=participants,
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
        participants: tuple[str, ...],
    ) -> DefinitionDocument | None:
        if is_ad_hoc_workroom(workroom_id):
            if not participants:
                return None
            return _ad_hoc_workroom_definition(
                participants=participants,
            )
        return self.registry.workroom_definitions.get(workroom_id or "")


def _workroom_notice(base: str) -> str:
    return base + "\n"


def _workroom_intro(definition: DefinitionDocument) -> str:
    if is_ad_hoc_workroom(definition.id):
        return "Orchestrator: opening an ad hoc workroom."
    return f"Orchestrator: opening workroom {definition.id}."


def _ad_hoc_workroom_definition(
    *,
    participants: tuple[str, ...],
) -> DefinitionDocument:
    expanded_staffing = participants
    unique_roles = {agent_id for agent_id in expanded_staffing}
    if len(expanded_staffing) > 1 and len(unique_roles) == 1:
        shape = "staged"
        metadata: dict[str, object] = {"stages": list(expanded_staffing)}
    else:
        shape = "discussion"
        ordered_roles: list[str] = []
        for agent_id in expanded_staffing:
            if agent_id not in ordered_roles:
                ordered_roles.append(agent_id)
        metadata = {"turns": ordered_roles}
    return DefinitionDocument(
        id=AD_HOC_WORKROOM_ID,
        path=Path(AD_HOC_WORKROOM_ID),
        metadata={
            "id": AD_HOC_WORKROOM_ID,
            "name": "Ad Hoc Workroom",
            "shape": shape,
            **metadata,
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
