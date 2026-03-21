from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.magentic_workflow_executor import (
    ProxyMagenticWorkflowExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class MagenticWorkflowExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_selects_agents_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
        manager_calls: list[tuple[str, tuple[str, ...]]] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Plan", "coder": "Ship"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Delivered")

        async def _select_manager_agent(**kwargs):
            manager_calls.append((kwargs["current_brief"], kwargs["prior_outputs"]))
            return "architect" if not kwargs["prior_outputs"] else "coder"

        executor = ProxyMagenticWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_manager_agent=_select_manager_agent,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Build it",
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "coder"])
        self.assertEqual(
            manager_calls,
            [("Build it", ()), ("Plan", ("architect: Plan",))],
        )
        self.assertEqual(
            summary_calls,
            [("Ship", ("architect: Plan", "coder: Ship"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[2], ProxyContentDeltaEvent)


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="dynamic-open-ended",
        path=Path("dynamic-open-ended.md"),
        metadata={
            "id": "dynamic-open-ended",
            "orchestration": "magentic",
            "steps": ["architect", "coder"],
            "max_rounds": 2,
        },
        body="## Purpose\nAdapt.",
        sections={"Purpose": "Adapt."},
    )
