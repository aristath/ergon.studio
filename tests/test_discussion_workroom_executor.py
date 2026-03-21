from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.discussion_workroom_executor import (
    ProxyDiscussionWorkroomExecutor,
)
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyMoveResult, ProxyTurnState


class DiscussionWorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_one_full_discussion_round_and_keeps_room_open(
        self,
    ) -> None:
        streamed_agents: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyDiscussionWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Discuss it",
                state=state,
                result_sink=captured.append,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "reviewer"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(
            captured[0].worklog_lines,
            ("architect: Idea", "reviewer: Refine"),
        )
        self.assertEqual(captured[0].current_brief, "Refine")
        self.assertIsNotNone(captured[0].workroom_progress)
        if captured[0].workroom_progress is None:
            raise AssertionError("expected discussion room to stay active")
        self.assertEqual(captured[0].workroom_progress.workroom_id, "debate")
        self.assertIsNone(captured[0].workroom_progress.progress_index)
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Idea", reasoning)
        self.assertIn("reviewer: Refine", reasoning)

    async def test_execute_expands_repeated_staffed_instances(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            yield {
                "architect": "Idea",
                "reviewer": "Challenge",
            }[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyDiscussionWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Debate it"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Debate it",
                participants=("architect", "reviewer", "reviewer"),
                state=state,
                result_sink=captured.append,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "reviewer", "reviewer"])
        self.assertIn("Current staffed instance: reviewer[1]", streamed_prompts[1])
        self.assertIn("instance 1 of 2 staffed reviewers", streamed_prompts[1])
        self.assertIn("Current staffed instance: reviewer[2]", streamed_prompts[2])
        self.assertEqual(
            captured[0].worklog_lines,
            (
                "architect: Idea",
                "reviewer[1]: Challenge",
                "reviewer[2]: Challenge",
            ),
        )
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Idea", reasoning)
        self.assertIn("reviewer[1]: Challenge", reasoning)
        self.assertIn("reviewer[2]: Challenge", reasoning)


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="debate",
        path=Path("debate.md"),
        metadata={
            "id": "debate",
            "shape": "discussion",
            "turns": ["architect", "reviewer"],
        },
        body="## Purpose\nDebate.",
        sections={"Purpose": "Debate."},
    )
