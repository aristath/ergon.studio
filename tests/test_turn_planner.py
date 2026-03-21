from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.proxy.turn_planner import ProxyTurnPlanner
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class TurnPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_turn_returns_act_when_planner_returns_invalid_json(
        self,
    ) -> None:
        async def _run_text_agent(**_kwargs):
            return "not-json"

        planner = ProxyTurnPlanner(_registry(), run_text_agent=_run_text_agent)
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        plan = await planner.plan_turn(request)

        self.assertEqual(plan.mode, "act")

    async def test_plan_turn_parses_valid_workflow_plan(self) -> None:
        async def _run_text_agent(**_kwargs):
            return (
                '{"mode":"workflow","workflow_id":"standard-build",'
                '"goal":"Build calculator"}'
            )

        planner = ProxyTurnPlanner(_registry(), run_text_agent=_run_text_agent)
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        plan = await planner.plan_turn(request)

        self.assertEqual(plan.mode, "workflow")
        self.assertEqual(plan.workflow_id, "standard-build")
        self.assertEqual(plan.goal, "Build calculator")


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
        agent_definitions={},
        workflow_definitions={
            "standard-build": DefinitionDocument(
                id="standard-build",
                path=Path("standard-build.md"),
                metadata={
                    "id": "standard-build",
                    "orchestration": "sequential",
                    "steps": ["architect", "coder"],
                },
                body="## Purpose\nBuild.",
                sections={"Purpose": "Build."},
            )
        },
    )
