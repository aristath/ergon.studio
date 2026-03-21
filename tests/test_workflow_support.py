from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workflow_support import ProxyWorkflowSupport


class WorkflowSupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_emit_summary_uses_orchestrator_summary_and_sets_content(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        async def _run_text_agent(**kwargs):
            captured.update(kwargs)
            return "Final workflow answer"

        support = ProxyWorkflowSupport(run_text_agent=_run_text_agent)
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in support.emit_summary(
                request=request,
                definition=_definition(),
                goal="Build it",
                current_brief="Coder done",
                workflow_outputs=("architect: plan", "coder: built"),
                state=state,
            )
        ]

        self.assertEqual(captured["agent_id"], "orchestrator")
        self.assertEqual(captured["model_id_override"], "qwen")
        self.assertEqual(state.content, "Final workflow answer")
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ProxyContentDeltaEvent)

    async def test_select_manager_agent_parses_orchestrator_choice(self) -> None:
        async def _run_text_agent(**_kwargs):
            return '{"agent_id":"coder"}'

        support = ProxyWorkflowSupport(run_text_agent=_run_text_agent)

        selected = await support.select_manager_agent(
            workflow_id="dynamic-open-ended",
            goal="Build it",
            current_brief="Need implementation",
            participants=("architect", "coder"),
            prior_outputs=("architect: plan",),
            move_rationale="Architecture is done; implementation should start.",
            success_criteria="Pick the next specialist who can advance delivery.",
            model_id_override="qwen",
        )

        self.assertEqual(selected, "coder")

    async def test_select_handoff_target_returns_none_when_no_allowed_agents(
        self,
    ) -> None:
        async def _run_text_agent(**_kwargs):
            raise AssertionError("should not be called")

        support = ProxyWorkflowSupport(run_text_agent=_run_text_agent)

        selected = await support.select_handoff_target(
            workflow_id="specialist-handoff",
            current_agent="coder",
            goal="Ship it",
            current_brief="Done",
            prior_outputs=("coder: done",),
            allowed=(),
            move_rationale="The handoff is likely complete.",
            success_criteria="Only continue if another specialist is truly needed.",
            model_id_override="qwen",
        )

        self.assertIsNone(selected)

    async def test_select_comparison_outcome_parses_structured_result(self) -> None:
        async def _run_text_agent(**_kwargs):
            return (
                '{"selected_candidate_number":2,'
                '"summary":"Candidate 2 is safer and clearer.",'
                '"next_refinement":"Polish the selected candidate."}'
            )

        support = ProxyWorkflowSupport(run_text_agent=_run_text_agent)

        outcome = await support.select_comparison_outcome(
            workflow_id="best-of-n",
            goal="Pick the best candidate",
            comparison_mode="select_best",
            comparison_candidates=("coder[1]: Idea A", "coder[2]: Idea B"),
            stage_outputs=("reviewer: Candidate 2 wins",),
            comparison_criteria="Prefer the safer and simpler approach.",
            move_rationale="We need a clear winner before polishing.",
            success_criteria="Choose the best candidate and explain why.",
            model_id_override="qwen",
        )

        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.mode, "select_best")
        self.assertEqual(outcome.selected_candidate_index, 1)
        self.assertEqual(outcome.selected_candidate_text, "coder[2]: Idea B")
        self.assertEqual(
            outcome.next_refinement,
            "Polish the selected candidate.",
        )

    async def test_select_comparison_outcome_returns_none_without_candidates(
        self,
    ) -> None:
        async def _run_text_agent(**_kwargs):
            raise AssertionError("should not be called")

        support = ProxyWorkflowSupport(run_text_agent=_run_text_agent)

        outcome = await support.select_comparison_outcome(
            workflow_id="best-of-n",
            goal="Pick the best candidate",
            comparison_mode="select_best",
            comparison_candidates=(),
            stage_outputs=("reviewer: Candidate 2 wins",),
            comparison_criteria=None,
            move_rationale=None,
            success_criteria=None,
            model_id_override="qwen",
        )

        self.assertIsNone(outcome)


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
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
