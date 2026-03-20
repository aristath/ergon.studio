from __future__ import annotations

import unittest

from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.proxy.planner import build_turn_planner_prompt, parse_turn_plan, resolve_workflow_reference
from ergon_studio.registry import RuntimeRegistry


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

    def test_parse_turn_plan_resolves_known_workflow(self) -> None:
        registry = _make_registry()

        plan = parse_turn_plan(
            '{"mode":"workflow","workflow_id":"standard-build","goal":"Build it","deliverable_expected":true}',
            registry=registry,
        )

        self.assertEqual(plan.mode, "workflow")
        self.assertEqual(plan.workflow_id, "standard-build")
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
        config={
            "providers": {
                "local": {
                    "type": "openai_chat",
                    "base_url": "http://localhost:8080/v1",
                    "api_key": "not-needed",
                    "model": "qwen2.5-coder",
                }
            },
            "role_assignments": {
                "orchestrator": "local",
                "architect": "local",
                "coder": "local",
            },
        },
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


if __name__ == "__main__":
    unittest.main()
