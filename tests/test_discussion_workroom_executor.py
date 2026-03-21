from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.discussion_workroom_executor import (
    ProxyDiscussionWorkroomExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class DiscussionWorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_sequence_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workroom_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workroom_outputs"])
            )
            yield ProxyContentDeltaEvent("Shared result")

        executor = ProxyDiscussionWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workroom_summary=_emit_workroom_summary,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Discuss it",
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "reviewer"])
        self.assertEqual(
            summary_calls,
            [("Refine", ("architect: Idea", "reviewer: Refine"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[2], ProxyContentDeltaEvent)

    async def test_execute_expands_repeated_staffed_instances(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            yield {
                "architect": "Idea",
                "reviewer": "Challenge",
            }[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workroom_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workroom_outputs"])
            )
            yield ProxyContentDeltaEvent("Shared result")

        executor = ProxyDiscussionWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workroom_summary=_emit_workroom_summary,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Debate it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Debate it",
                specialists=("architect", "reviewer"),
                specialist_counts=(("reviewer", 2),),
                state=state,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "reviewer", "reviewer"])
        self.assertIn("Current staffed instance: reviewer[1]", streamed_prompts[1])
        self.assertIn("instance 1 of 2 staffed reviewers", streamed_prompts[1])
        self.assertIn("Current staffed instance: reviewer[2]", streamed_prompts[2])
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Idea", reasoning)
        self.assertIn("reviewer[1]: Challenge", reasoning)
        self.assertIn("reviewer[2]: Challenge", reasoning)
        self.assertEqual(
            summary_calls,
            [
                (
                    "Challenge",
                    (
                        "architect: Idea",
                        "reviewer[1]: Challenge",
                        "reviewer[2]: Challenge",
                    ),
                )
            ],
        )


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="debate",
        path=Path("debate.md"),
        metadata={
            "id": "debate",
            "shape": "discussion",
            "steps": ["architect", "reviewer"],
        },
        body="## Purpose\nDebate.",
        sections={"Purpose": "Debate."},
    )
