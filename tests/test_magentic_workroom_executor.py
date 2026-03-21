from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.magentic_workroom_executor import (
    ProxyMagenticWorkroomExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class MagenticWorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_selects_agents_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
        manager_calls: list[tuple[str, tuple[str, ...]]] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Plan", "coder": "Ship"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workroom_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workroom_outputs"])
            )
            yield ProxyContentDeltaEvent("Delivered")

        async def _select_manager_agent(**kwargs):
            manager_calls.append((kwargs["current_brief"], kwargs["prior_outputs"]))
            return "architect" if not kwargs["prior_outputs"] else "coder"

        executor = ProxyMagenticWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workroom_summary=_emit_workroom_summary,
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

    async def test_execute_allows_repeated_staffed_instances(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []
        manager_calls: list[tuple[str, ...]] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            yield "Ship"

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workroom_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workroom_outputs"])
            )
            yield ProxyContentDeltaEvent("Delivered")

        async def _select_manager_agent(**kwargs):
            manager_calls.append(kwargs["participants"])
            return "coder[2]"

        executor = ProxyMagenticWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workroom_summary=_emit_workroom_summary,
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
                definition=_single_round_definition(),
                goal="Build it",
                specialists=("coder",),
                specialist_counts=(("coder", 2),),
                state=state,
            )
        ]

        self.assertEqual(manager_calls, [("coder[1]", "coder[2]")])
        self.assertEqual(streamed_agents, ["coder"])
        self.assertIn("Current staffed instance: coder[2]", streamed_prompts[0])
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder[2]: Ship", reasoning)
        self.assertEqual(summary_calls, [("Ship", ("coder[2]: Ship",))])


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


def _single_round_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="dynamic-open-ended",
        path=Path("dynamic-open-ended.md"),
        metadata={
            "id": "dynamic-open-ended",
            "orchestration": "magentic",
            "steps": ["coder"],
            "max_rounds": 1,
        },
        body="## Purpose\nAdapt.",
        sections={"Purpose": "Adapt."},
    )
