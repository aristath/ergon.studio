from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.grouped_workflow_executor import ProxyGroupedWorkflowExecutor
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class GroupedWorkflowExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_steps_and_emits_summary(self) -> None:
        streamed_agents: list[str] = []
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
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
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
            summary_calls,
            [("Built", ("architect: Plan", "coder: Built"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[2], ProxyContentDeltaEvent)

    async def test_execute_labels_parallel_role_instances(self) -> None:
        streamed_prompts: list[str] = []
        summary_calls: list[tuple[str, tuple[str, ...]]] = []
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        async def _emit_workflow_summary(**kwargs):
            summary_calls.append(
                (kwargs["current_brief"], kwargs["workflow_outputs"])
            )
            yield ProxyContentDeltaEvent("Final summary")

        executor = ProxyGroupedWorkflowExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
            emit_workflow_summary=_emit_workflow_summary,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_best_of_n_definition(),
                goal="Try a few approaches",
                state=state,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder[1]: Idea A", reasoning)
        self.assertIn("coder[2]: Idea B", reasoning)
        self.assertIn("coder[3]: Chosen", reasoning)
        self.assertEqual(
            summary_calls,
            [
                (
                    "Chosen",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
                )
            ],
        )
        self.assertIn("Current staffed instance: coder[1]", streamed_prompts[0])
        self.assertIn("instance 2 of 3", streamed_prompts[1])


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


def _best_of_n_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "orchestration": "grouped",
            "step_groups": [["coder", "coder", "coder"]],
        },
        body="## Purpose\nCompare attempts.",
        sections={"Purpose": "Compare attempts."},
    )
