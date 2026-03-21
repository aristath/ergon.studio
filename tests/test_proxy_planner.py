from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall, ProxyTurnRequest
from ergon_studio.proxy.planner import (
    build_turn_planner_prompt,
    parse_turn_plan,
    resolve_workroom_reference,
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
            active_workroom_id="best-of-n",
            active_specialists=("coder", "reviewer"),
            active_specialist_counts=(("coder", 3),),
        )

        self.assertIn("Workroom currently in progress:", prompt)
        self.assertIn("best-of-n", prompt)
        self.assertIn("Currently staffed specialists:", prompt)
        self.assertIn("coder, reviewer", prompt)
        self.assertIn("Current role instance counts:", prompt)
        self.assertIn("coder x3", prompt)

    def test_build_turn_planner_prompt_includes_active_workroom_request(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Keep going"),),
        )

        prompt = build_turn_planner_prompt(
            request,
            active_workroom_id="best-of-n",
            active_workroom_request="Compare the two alternatives and pick one.",
        )

        self.assertIn("Current workroom assignment:", prompt)
        self.assertIn("Compare the two alternatives and pick one.", prompt)

    def test_build_turn_planner_prompt_includes_delivery_requirements(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Ship it"),),
        )

        prompt = build_turn_planner_prompt(
            request,
            active_delivery_requirements=("review", "verify"),
            satisfied_delivery_evidence=("review",),
        )

        self.assertIn("Current delivery requirements:", prompt)
        self.assertIn("review, verify", prompt)
        self.assertIn("Satisfied delivery evidence:", prompt)
        self.assertIn("Still missing before delivery:", prompt)
        self.assertIn("verify", prompt)

    def test_parse_turn_plan_requires_action(self) -> None:
        registry = _make_registry()

        with self.assertRaisesRegex(ValueError, "must include action"):
            parse_turn_plan(
                '{"mode":"workflow","workroom_id":"standard-build"}',
                registry=registry,
            )

    def test_parse_turn_plan_parses_compact_delegate_action(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"delegate","target":"coder",'
                '"assignment":"Implement the change","rationale":"Keep this focused"}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "delegate")
        self.assertEqual(plan.agent_id, "coder")
        self.assertEqual(plan.request, "Implement the change")
        self.assertEqual(plan.rationale, "Keep this focused")

    def test_parse_turn_plan_parses_compact_workroom_action(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"open_workroom","target":"standard-build",'
                '"assignment":"Build the feature safely","staffing":['
                '"coder","coder","reviewer"],'
                '"delivery_requirements":["review"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "workroom")
        self.assertEqual(plan.workroom_id, "standard-build")
        self.assertEqual(plan.workroom_request, "Build the feature safely")
        self.assertEqual(plan.specialists, ("coder", "reviewer"))
        self.assertEqual(plan.specialist_counts, (("coder", 2),))
        self.assertEqual(plan.delivery_requirements, ("review",))

    def test_parse_turn_plan_parses_compact_continue_workroom_action(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"continue_workroom","target":"current",'
                '"assignment":"Run one more review pass","staffing":["reviewer"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "continue_workroom")
        self.assertIsNone(plan.workroom_id)
        self.assertEqual(plan.workroom_request, "Run one more review pass")
        self.assertEqual(plan.specialists, ("reviewer",))

    def test_parse_turn_plan_drops_unknown_agent(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"action":"delegate","target":"ghost","assignment":"Do it"}',
            registry=registry,
        )

        self.assertEqual(plan.mode, "delegate")
        self.assertIsNone(plan.agent_id)

    def test_parse_turn_plan_resolves_workroom_by_name(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"action":"open_workroom","target":"Standard Build"}',
            registry=registry,
        )

        self.assertEqual(plan.workroom_id, "standard-build")

    def test_parse_turn_plan_resolves_workroom_by_selection_hint(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"action":"open_workroom","target":"staged_delivery"}',
            registry=registry,
        )

        self.assertEqual(plan.workroom_id, "standard-build")

    def test_parse_turn_plan_normalizes_staffing_list(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"open_workroom","target":"standard-build",'
                '"staffing":["coder","orchestrator","ghost","coder"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.specialists, ("coder",))
        self.assertEqual(plan.specialist_counts, (("coder", 2),))

    def test_parse_turn_plan_opens_ad_hoc_workroom_from_staffing(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"open_workroom","assignment":"Brainstorm a safe plan",'
                '"staffing":["architect","coder","critic"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "workroom")
        self.assertEqual(plan.workroom_id, "ad-hoc-workroom")
        self.assertEqual(plan.specialists, ("architect", "coder", "critic"))

    def test_parse_turn_plan_allows_dense_staffing_list(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"open_workroom","target":"best-of-n",'
                '"staffing":["coder","coder","coder","architect"]}'
            ),
            registry=registry,
        )

        self.assertEqual(
            plan.specialist_counts,
            (("coder", 3),),
        )

    def test_parse_turn_plan_parses_workroom_request(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"continue_workroom","target":"current",'
                '"assignment":"Compare the current options and pick one."}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "continue_workroom")
        self.assertEqual(
            plan.workroom_request,
            "Compare the current options and pick one.",
        )

    def test_parse_turn_plan_parses_delivery_requirements(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            (
                '{"action":"deliver","delivery_requirements":['
                '"reviewed","verification","ghost","review"]}'
            ),
            registry=registry,
        )

        self.assertEqual(plan.mode, "finish")
        self.assertEqual(plan.delivery_requirements, ("review", "verify"))

    def test_resolve_workroom_reference_returns_none_for_ambiguous_hint(self) -> None:
        registry = _make_registry()
        registry.workroom_definitions["other-build"] = DefinitionDocument(
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

        self.assertIsNone(resolve_workroom_reference(registry, "staged_delivery"))


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
            "reviewer": DefinitionDocument(
                id="reviewer",
                path=Path("reviewer.md"),
                metadata={"id": "reviewer", "role": "reviewer"},
                body="## Identity\nReviewer.",
                sections={"Identity": "Reviewer."},
            ),
            "critic": DefinitionDocument(
                id="critic",
                path=Path("critic.md"),
                metadata={"id": "critic", "role": "critic"},
                body="## Identity\nCritic.",
                sections={"Identity": "Critic."},
            ),
        },
        workroom_definitions={
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
