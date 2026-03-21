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
