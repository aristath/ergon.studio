from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall, ProxyTurnRequest
from ergon_studio.proxy.planner import (
    build_turn_planner_prompt,
    parse_turn_plan,
    resolve_workflow_reference,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class ProxyPlannerTests(unittest.TestCase):
    def test_build_turn_planner_prompt_includes_transcript(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="system", content="rules"),
                ProxyInputMessage(role="user", content="Build a calculator"),
                ProxyInputMessage(role="assistant", content="I will help"),
            ),
        )

        prompt = build_turn_planner_prompt(request)

        self.assertIn("Build a calculator", prompt)
        self.assertIn("assistant: I will help", prompt)

    def test_build_turn_planner_prompt_includes_tool_names(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect the file"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ProxyToolCall(
                            id="call_1",
                            name="read_file",
                            arguments_json='{"path":"main.py"}',
                        ),
                    ),
                ),
                ProxyInputMessage(
                    role="tool",
                    name="read_file",
                    content="print('hello')",
                    tool_call_id="call_1",
                ),
            ),
        )

        prompt = build_turn_planner_prompt(request)

        self.assertIn("assistant: [tool_calls read_file]", prompt)
        self.assertIn("tool_result[call_1]<read_file>: print('hello')", prompt)

    def test_build_turn_planner_prompt_includes_active_staffing(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Keep going"),),
        )

        prompt = build_turn_planner_prompt(
            request,
            active_workflow_id="best-of-n",
            active_specialists=("coder", "reviewer"),
            active_specialist_counts=(("coder", 3),),
        )

        self.assertIn("Playbook currently in progress:", prompt)
        self.assertIn("best-of-n", prompt)
        self.assertIn("Currently staffed specialists:", prompt)
        self.assertIn("coder, reviewer", prompt)
        self.assertIn("Current role instance counts:", prompt)
        self.assertIn("coder x3", prompt)

    def test_build_turn_planner_prompt_includes_active_playbook_request(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Keep going"),),
        )

        prompt = build_turn_planner_prompt(
            request,
            active_workflow_id="best-of-n",
            active_playbook_request="Compare the two alternatives and pick one.",
        )

        self.assertIn("Current playbook round assignment:", prompt)
        self.assertIn("Compare the two alternatives and pick one.", prompt)

    def test_parse_turn_plan_resolves_known_workflow(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"mode":"workflow","workflow_id":"standard-build",'
                '"goal":"Build it","rationale":"This needs staged work",'
                '"comparison_mode":"select_best",'
                '"comparison_criteria":"Prefer the safest implementation",'
                '"success_criteria":"Have a reviewed implementation",'
                '"deliverable_expected":true}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "workflow")
        self.assertEqual(plan.workflow_id, "standard-build")
        self.assertEqual(plan.rationale, "This needs staged work")
        self.assertEqual(plan.comparison_mode, "select_best")
        self.assertEqual(
            plan.comparison_criteria,
            "Prefer the safest implementation",
        )
        self.assertEqual(plan.success_criteria, "Have a reviewed implementation")
        self.assertTrue(plan.deliverable_expected)

    def test_parse_turn_plan_drops_unknown_agent(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"mode":"delegate","agent_id":"ghost","request":"Do it"}',
            registry=registry,
        )

        self.assertEqual(plan.mode, "delegate")
        self.assertIsNone(plan.agent_id)

    def test_parse_turn_plan_resolves_workflow_by_name(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"mode":"workflow","workflow_id":"Standard Build"}',
            registry=registry,
        )

        self.assertEqual(plan.workflow_id, "standard-build")

    def test_parse_turn_plan_resolves_workflow_by_selection_hint(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"mode":"workflow","workflow_id":"staged_delivery"}',
            registry=registry,
        )

        self.assertEqual(plan.workflow_id, "standard-build")

    def test_parse_turn_plan_normalizes_staffed_specialists(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"mode":"workflow","workflow_id":"standard-build",'
                '"specialists":["coder","orchestrator","ghost","coder"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.specialists, ("coder",))

    def test_parse_turn_plan_normalizes_specialist_counts(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"mode":"workflow","workflow_id":"best-of-n",'
                '"specialist_counts":{"coder":3,"architect":1,"ghost":2,'
                '"orchestrator":5,"reviewer":true,"coder_bad":"4"}}'
            ),
            registry=registry,
        )

        self.assertEqual(
            plan.specialist_counts,
            (("coder", 3), ("architect", 1)),
        )

    def test_parse_turn_plan_parses_playbook_request(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"mode":"continue_playbook","workflow_id":"best-of-n",'
                '"playbook_request":"Compare the current options and pick one."}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "continue_playbook")
        self.assertEqual(
            plan.playbook_request,
            "Compare the current options and pick one.",
        )

    def test_resolve_workflow_reference_returns_none_for_ambiguous_hint(self) -> None:
        registry = _make_registry()
        registry.workflow_definitions["other-build"] = DefinitionDocument(
            id="other-build",
            path=Path("other-build.md"),
            metadata={
                "id": "other-build",
                "name": "Other Build",
                "selection_hints": ["staged_delivery"],
                "steps": ["coder"],
            },
            body="## Purpose\nBuild something else.",
            sections={"Purpose": "Build something else."},
        )

        self.assertIsNone(resolve_workflow_reference(registry, "staged_delivery"))


def _make_registry():
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
            "coder": DefinitionDocument(
                id="coder",
                path=Path("coder.md"),
                metadata={"id": "coder", "role": "coder"},
                body="## Identity\nCoder.",
                sections={"Identity": "Coder."},
            ),
        },
        workflow_definitions={
            "best-of-n": DefinitionDocument(
                id="best-of-n",
                path=Path("best-of-n.md"),
                metadata={
                    "id": "best-of-n",
                    "name": "Best Of N",
                    "orchestration": "grouped",
                    "selection_hints": ["multiple_attempts"],
                    "step_groups": [["coder", "coder"], ["reviewer"]],
                },
                body="## Purpose\nCompare multiple attempts.",
                sections={"Purpose": "Compare multiple attempts."},
            ),
            "standard-build": DefinitionDocument(
                id="standard-build",
                path=Path("standard-build.md"),
                metadata={
                    "id": "standard-build",
                    "name": "Standard Build",
                    "orchestration": "sequential",
                    "delivery_candidate": True,
                    "selection_hints": ["staged_delivery"],
                    "steps": ["architect", "coder"],
                },
                body="## Purpose\nBuild something.",
                sections={"Purpose": "Build something."},
            )
        },
    )
