from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.handoff_workflow_executor import (
    ProxyHandoffWorkflowExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class HandoffWorkflowExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_hands_off_until_finalizer(self) -> None:
        streamed_agents: list[str] = []
        handoff_calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Plan", "coder": "Built"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Done")

        async def _select_handoff_target(**kwargs):
            handoff_calls.append(
                (
                    kwargs["current_agent"],
                    kwargs["prior_outputs"],
                    kwargs["allowed"],
                )
            )
            return "coder"

        executor = ProxyHandoffWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_handoff_target=_select_handoff_target,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Ship it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Ship it",
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "coder"])
        self.assertEqual(
            handoff_calls,
            [("architect", ("architect: Plan",), ("coder",))],
        )
        self.assertEqual(
            summary_calls,
            [("Built", ("architect: Plan", "coder: Built"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[2], ProxyContentDeltaEvent)

    async def test_execute_supports_repeated_staffed_handoff_instances(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []
        handoff_calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            yield {"architect": "Plan", "coder": "Built"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Done")

        async def _select_handoff_target(**kwargs):
            handoff_calls.append(
                (
                    kwargs["current_agent"],
                    kwargs["prior_outputs"],
                    kwargs["allowed"],
                )
            )
            return (
                "architect[2]"
                if kwargs["current_agent"] == "architect[1]"
                else "coder"
            )

        executor = ProxyHandoffWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
            select_handoff_target=_select_handoff_target,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Ship it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_repeated_handoff_definition(),
                goal="Ship it",
                specialists=("architect", "coder"),
                specialist_counts=(("architect", 2),),
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "architect", "coder"])
        self.assertIn("Current staffed instance: architect[1]", streamed_prompts[0])
        self.assertIn("Current staffed instance: architect[2]", streamed_prompts[1])
        self.assertEqual(
            handoff_calls,
            [
                ("architect[1]", ("architect[1]: Plan",), ("architect[2]", "coder")),
                (
                    "architect[2]",
                    ("architect[1]: Plan", "architect[2]: Plan"),
                    ("architect[1]", "coder"),
                ),
            ],
        )
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect[1]: Plan", reasoning)
        self.assertIn("architect[2]: Plan", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(
            summary_calls,
            [
                (
                    "Built",
                    ("architect[1]: Plan", "architect[2]: Plan", "coder: Built"),
                )
            ],
        )


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="specialist-handoff",
        path=Path("specialist-handoff.md"),
        metadata={
            "id": "specialist-handoff",
            "orchestration": "handoff",
            "steps": ["architect", "coder"],
            "start_agent": "architect",
            "handoffs": {"architect": ["coder"]},
            "finalizers": ["coder"],
        },
        body="## Purpose\nHand off.",
        sections={"Purpose": "Hand off."},
    )


def _repeated_handoff_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="specialist-handoff",
        path=Path("specialist-handoff.md"),
        metadata={
            "id": "specialist-handoff",
            "orchestration": "handoff",
            "steps": ["architect", "coder"],
            "start_agent": "architect",
            "handoffs": {"architect": ["architect", "coder"]},
            "finalizers": ["coder"],
            "max_rounds": 3,
        },
        body="## Purpose\nHand off.",
        sections={"Purpose": "Hand off."},
    )
