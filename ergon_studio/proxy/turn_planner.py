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
                active_workflow_id=(
                    loop_state.workflow_progress.workflow_id
                    if loop_state is not None
                    and loop_state.workflow_progress is not None
                    else None
                ),
                selection_outcome=(
                    loop_state.latest_selection_outcome
                    if loop_state is not None
                    else None
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
        active_workflow_id = (
            loop_state.workflow_progress.workflow_id
            if loop_state is not None
            and loop_state.workflow_progress is not None
            else None
        )
        if active_workflow_id is None:
            if plan.mode == "continue_playbook":
                return replace(plan, mode="act", workflow_id=None)
            return plan

        if plan.mode == "continue_playbook":
            if plan.workflow_id is None:
                return replace(plan, workflow_id=active_workflow_id)
            if plan.workflow_id != active_workflow_id:
                return replace(plan, mode="workflow")
            return plan

        if plan.mode == "workflow" and plan.workflow_id == active_workflow_id:
            return replace(plan, mode="continue_playbook")
        return plan
