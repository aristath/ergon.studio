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
    comparison_outcome_instructions,
    comparison_outcome_prompt,
    handoff_selection_instructions,
    handoff_selection_prompt,
    parse_agent_selection,
    parse_comparison_outcome,
    summary_instructions,
    workflow_manager_instructions,
    workflow_manager_prompt,
    workflow_summary_prompt,
)
from ergon_studio.proxy.selection_outcome import ProxySelectionOutcome
from ergon_studio.proxy.turn_state import ProxyTurnState

ProxyEvent = (
    ProxyReasoningDeltaEvent
    | ProxyContentDeltaEvent
    | ProxyToolCallEvent
    | ProxyFinishEvent
)


class ProxyWorkflowSupport:
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
        workflow_outputs: tuple[str, ...],
        state: ProxyTurnState,
    ) -> AsyncIterator[ProxyEvent]:
        final_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=workflow_summary_prompt(
                workflow_id=definition.id,
                goal=goal,
                outputs=workflow_outputs,
            ),
            preamble=summary_instructions(),
            session_id=f"proxy-workflow-summary-{uuid4().hex}",
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
        workflow_id: str,
        goal: str,
        current_brief: str,
        playbook_request: str | None,
        participants: tuple[str, ...],
        prior_outputs: tuple[str, ...],
        move_rationale: str | None,
        success_criteria: str | None,
        model_id_override: str,
    ) -> str | None:
        raw = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=workflow_manager_prompt(
                workflow_id=workflow_id,
                goal=goal,
                current_brief=current_brief,
                playbook_request=playbook_request,
                participants=participants,
                prior_outputs=prior_outputs,
                move_rationale=move_rationale,
                success_criteria=success_criteria,
            ),
            preamble=workflow_manager_instructions(participants),
            session_id=f"proxy-workflow-manager-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return parse_agent_selection(raw, participants=participants)

    async def select_handoff_target(
        self,
        *,
        workflow_id: str,
        current_agent: str,
        goal: str,
        current_brief: str,
        playbook_request: str | None,
        prior_outputs: tuple[str, ...],
        allowed: tuple[str, ...],
        move_rationale: str | None,
        success_criteria: str | None,
        model_id_override: str,
    ) -> str | None:
        if not allowed:
            return None
        raw = await self._run_text_agent(
            agent_id=current_agent,
            prompt=handoff_selection_prompt(
                workflow_id=workflow_id,
                current_agent=current_agent,
                goal=goal,
                current_brief=current_brief,
                playbook_request=playbook_request,
                prior_outputs=prior_outputs,
                allowed=allowed,
                move_rationale=move_rationale,
                success_criteria=success_criteria,
            ),
            preamble=handoff_selection_instructions(allowed),
            session_id=f"proxy-handoff-select-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return parse_agent_selection(raw, participants=allowed)

    async def select_comparison_outcome(
        self,
        *,
        workflow_id: str,
        goal: str,
        comparison_mode: str | None,
        comparison_candidates: tuple[str, ...],
        stage_outputs: tuple[str, ...],
        comparison_criteria: str | None,
        move_rationale: str | None,
        success_criteria: str | None,
        model_id_override: str,
    ) -> ProxySelectionOutcome | None:
        if (
            comparison_mode is None
            or not comparison_candidates
            or not stage_outputs
        ):
            return None
        raw = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=comparison_outcome_prompt(
                workflow_id=workflow_id,
                goal=goal,
                comparison_mode=comparison_mode,
                comparison_candidates=comparison_candidates,
                stage_outputs=stage_outputs,
                comparison_criteria=comparison_criteria,
                move_rationale=move_rationale,
                success_criteria=success_criteria,
            ),
            preamble=comparison_outcome_instructions(
                candidate_count=len(comparison_candidates)
            ),
            session_id=f"proxy-compare-{uuid4().hex}",
            model_id_override=model_id_override,
        )
        return parse_comparison_outcome(
            raw,
            comparison_mode=comparison_mode,
            comparison_candidates=comparison_candidates,
        )
