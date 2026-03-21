from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any
from uuid import uuid4

from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.planner import (
    ProxyTurnPlan,
    build_turn_planner_instructions,
    build_turn_planner_prompt,
    parse_turn_plan,
)
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState
from ergon_studio.registry import RuntimeRegistry


class ProxyTurnPlanner:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        run_text_agent: Callable[..., Any],
    ) -> None:
        self._registry = registry
        self._run_text_agent = run_text_agent

    async def plan_turn(
        self,
        request: ProxyTurnRequest,
        *,
        loop_state: ProxyDecisionLoopState | None = None,
    ) -> ProxyTurnPlan:
        planner_text = await self._run_text_agent(
            agent_id="orchestrator",
            prompt=build_turn_planner_prompt(
                request,
                goal=loop_state.goal if loop_state is not None else None,
                current_brief=(
                    loop_state.current_brief if loop_state is not None else None
                ),
                worklog=loop_state.worklog if loop_state is not None else (),
                active_workroom_id=(
                    loop_state.workroom_progress.workroom_id
                    if loop_state is not None
                    and loop_state.workroom_progress is not None
                    else None
                ),
                active_specialists=(
                    loop_state.workroom_progress.workroom_specialists
                    if loop_state is not None
                    and loop_state.workroom_progress is not None
                    else ()
                ),
                active_specialist_counts=(
                    loop_state.workroom_progress.workroom_specialist_counts
                    if loop_state is not None
                    and loop_state.workroom_progress is not None
                    else ()
                ),
                active_workroom_request=(
                    loop_state.workroom_progress.workroom_request
                    if loop_state is not None
                    and loop_state.workroom_progress is not None
                    else None
                ),
                active_delivery_requirements=(
                    loop_state.delivery_requirements
                    if loop_state is not None
                    else ()
                ),
                satisfied_delivery_evidence=(
                    loop_state.delivery_evidence
                    if loop_state is not None
                    else ()
                ),
            ),
            preamble=build_turn_planner_instructions(self._registry),
            session_id=f"proxy-planner-{uuid4().hex}",
            model_id_override=request.model,
        )
        if not planner_text:
            return ProxyTurnPlan(mode="act")
        try:
            plan = parse_turn_plan(planner_text, registry=self._registry)
        except ValueError:
            return ProxyTurnPlan(mode="act")
        return self._normalize_plan(plan, loop_state=loop_state)

    def _normalize_plan(
        self,
        plan: ProxyTurnPlan,
        *,
        loop_state: ProxyDecisionLoopState | None,
    ) -> ProxyTurnPlan:
        active_workroom_id = (
            loop_state.workroom_progress.workroom_id
            if loop_state is not None
            and loop_state.workroom_progress is not None
            else None
        )
        if active_workroom_id is None:
            if plan.mode == "continue_workroom":
                return replace(plan, mode="act", workroom_id=None)
            return plan

        if plan.mode == "continue_workroom":
            if plan.workroom_id is None:
                return replace(plan, workroom_id=active_workroom_id)
            if plan.workroom_id != active_workroom_id:
                return replace(plan, mode="workroom")
            return plan

        if plan.mode == "workroom" and plan.workroom_id == active_workroom_id:
            return replace(plan, mode="continue_workroom")
        return plan
