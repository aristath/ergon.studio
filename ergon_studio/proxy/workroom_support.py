from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.prompts import (
    handoff_selection_instructions,
    handoff_selection_prompt,
    parse_agent_selection,
    summary_instructions,
    workroom_manager_instructions,
    workroom_manager_prompt,
    workroom_summary_prompt,
)
from ergon_studio.proxy.turn_state import ProxyTurnState

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyWorkroomSupport:
    def __init__(
        self,
        *,
        run_text_agent: Callable[..., Any],
    ) -> None:
        self._run_text_agent = run_text_agent

    async def emit_summary(
        self,
        *,
        request: ProxyTurnRequest,
        definition: DefinitionDocument,
        goal: str,
        current_brief: str,
        workroom_outputs: tuple[str, ...],
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=workroom_summary_prompt(
                workroom_id=definition.id,
                goal=goal,
                outputs=workroom_outputs,
            ),
            preamble=summary_instructions(),
            session_id=f"proxy-workroom-summary-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not final_text:
            final_text = current_brief.strip()
        state.set_content(final_text)
        if final_text:
            yield ProxyContentDeltaEvent(final_text)

    async def select_manager_agent(
        self,
        *,
        workroom_id: str,
        goal: str,
        current_brief: str,
        workroom_request: str | None = None,
        participants: tuple[str, ...],
        prior_outputs: tuple[str, ...],
        move_rationale: str | None = None,
        model_id_override: str,
    ) -> str | None:
        raw = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=workroom_manager_prompt(
                workroom_id=workroom_id,
                goal=goal,
                current_brief=current_brief,
                workroom_request=workroom_request,
                participants=participants,
                prior_outputs=prior_outputs,
                move_rationale=move_rationale,
            ),
            preamble=workroom_manager_instructions(participants),
            session_id=f"proxy-workroom-manager-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return parse_agent_selection(raw, participants=participants)

    async def select_handoff_target(
        self,
        *,
        workroom_id: str,
        current_agent: str,
        goal: str,
        current_brief: str,
        workroom_request: str | None = None,
        prior_outputs: tuple[str, ...],
        allowed: tuple[str, ...],
        move_rationale: str | None = None,
        model_id_override: str,
    ) -> str | None:
        if not allowed:
            return None
        raw = await self._run_text_agent(
            agent_id=current_agent,
            prompt=handoff_selection_prompt(
                workroom_id=workroom_id,
                current_agent=current_agent,
                goal=goal,
                current_brief=current_brief,
                workroom_request=workroom_request,
                prior_outputs=prior_outputs,
                allowed=allowed,
                move_rationale=move_rationale,
            ),
            preamble=handoff_selection_instructions(allowed),
            session_id=f"proxy-handoff-select-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return parse_agent_selection(raw, participants=allowed)
