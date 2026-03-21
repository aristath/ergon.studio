from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.group_chat_workflow_executor import (
    ProxyGroupChatWorkflowExecutor,
)
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class GroupChatWorkflowExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_sequence_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Shared result")

        executor = ProxyGroupChatWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
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


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="debate",
        path=Path("debate.md"),
        metadata={
            "id": "debate",
            "orchestration": "group_chat",
            "steps": ["architect", "reviewer"],
            "participants": ["architect", "reviewer"],
            "selection_sequence": ["architect", "reviewer"],
        },
        body="## Purpose\nDebate.",
        sections={"Purpose": "Debate."},
    )
