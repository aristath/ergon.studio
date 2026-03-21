from __future__ import annotations

import asyncio
import time
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
                    "coder[1]: Idea A\ncoder[2]: Idea B\ncoder[3]: Chosen",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
                )
            ],
        )
        self.assertIn("Current staffed instance: coder[1]", streamed_prompts[0])
        self.assertIn("instance 2 of 3", streamed_prompts[1])

    async def test_execute_runs_parallel_attempt_group_concurrently(self) -> None:
        summary_calls: list[tuple[str, tuple[str, ...]]] = []
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**_kwargs):
            await asyncio.sleep(0.05)
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

        started_at = time.perf_counter()
        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_definition(),
            goal="Try a few approaches",
            state=state,
        )]
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.12)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Idea A\ncoder[2]: Idea B\ncoder[3]: Chosen",
                    ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
                )
            ],
        )

    async def test_execute_falls_back_to_sequential_when_parallel_attempt_uses_tools(
        self,
    ) -> None:
        call_count = 0
        summary_calls: list[tuple[str, tuple[str, ...]]] = []

        async def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                kwargs["final_response_sink"](
                    _fake_response_with_tool_call(kwargs["agent_id"], call_count)
                )
                yield f"Draft {call_count}"
                return
            yield f"Final {call_count}"

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
            tools=(_host_tool("read_file"),),
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
        self.assertIn("rerunning this staffed group sequentially", reasoning)
        self.assertEqual(call_count, 6)
        self.assertEqual(
            summary_calls,
            [
                (
                    "coder[1]: Final 4\ncoder[2]: Final 5\ncoder[3]: Final 6",
                    ("coder[1]: Final 4", "coder[2]: Final 5", "coder[3]: Final 6"),
                )
            ],
        )


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


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )


def _fake_response_with_tool_call(agent_id: str, call_index: int):
    class _FakeContent:
        def __init__(self) -> None:
            self.type = "function_call"
            self.call_id = f"call_{call_index}"
            self.name = "read_file"
            self.arguments = '{"path":"main.py"}'

    class _FakeMessage:
        def __init__(self) -> None:
            self.contents = [_FakeContent()]
            self.author_name = agent_id

    class _FakeResponse:
        def __init__(self) -> None:
            self.messages = [_FakeMessage()]

    return _FakeResponse()
