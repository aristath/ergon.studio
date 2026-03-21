from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.staged_workroom_executor import ProxyStagedWorkroomExecutor
from ergon_studio.proxy.turn_state import ProxyMoveResult, ProxyTurnState


class StagedWorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_first_stage_and_returns_next_progress(self) -> None:
        streamed_agents: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Plan", "coder": "Built"}[kwargs["agent_id"]]

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Build it",
                state=state,
                result_sink=captured.append,
            )
        ]

        self.assertEqual(streamed_agents, ["architect"])
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].worklog_lines, ("architect: Plan",))
        self.assertEqual(captured[0].current_brief, "Plan")
        self.assertIsNotNone(captured[0].workroom_progress)
        if captured[0].workroom_progress is None:
            raise AssertionError("expected staged workroom to advance")
        self.assertEqual(captured[0].workroom_progress.progress_index, 1)
        self.assertEqual(captured[0].workroom_progress.agent_id, "coder")
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)

    async def test_execute_labels_parallel_role_instances(self) -> None:
        streamed_prompts: list[str] = []
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_best_of_n_definition(),
                goal="Try a few approaches",
                state=state,
                result_sink=captured.append,
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
            captured[0].worklog_lines,
            ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
        )
        self.assertIn("Current staffed instance: coder[1]", streamed_prompts[0])
        self.assertIn("instance 2 of 3", streamed_prompts[1])

    async def test_execute_expands_repeated_role_instances_from_staffing_plan(
        self,
    ) -> None:
        coder_outputs = iter(["Idea A", "Idea B", "Idea C"])

        async def _stream_text_agent(**_kwargs):
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_definition(),
                goal="Try a few approaches",
                participants=("coder", "coder", "coder"),
                state=state,
                result_sink=captured.append,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertNotIn("architect:", reasoning)
        self.assertIn("coder[1]: Idea A", reasoning)
        self.assertIn("coder[2]: Idea B", reasoning)
        self.assertIn("coder[3]: Idea C", reasoning)
        self.assertEqual(
            captured[0].worklog_lines,
            ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Idea C"),
        )

    async def test_execute_runs_parallel_attempt_group_concurrently(self) -> None:
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**_kwargs):
            await asyncio.sleep(0.05)
            yield next(coder_outputs)

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        started_at = time.perf_counter()
        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_definition(),
            goal="Try a few approaches",
            state=state,
            result_sink=captured.append,
        )]
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.12)
        self.assertEqual(
            captured[0].worklog_lines,
            ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
        )

    async def test_execute_falls_back_to_sequential_when_parallel_attempt_uses_tools(
        self,
    ) -> None:
        call_count = 0

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

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
            tools=(_host_tool("read_file"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_best_of_n_definition(),
                goal="Try a few approaches",
                state=state,
                result_sink=captured.append,
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
            captured[0].worklog_lines,
            ("coder[1]: Final 4", "coder[2]: Final 5", "coder[3]: Final 6"),
        )

    async def test_next_stage_receives_prior_workroom_outputs(self) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Selected best option"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_review_definition(),
            goal="Pick the best one",
            state=state,
            result_sink=lambda _result: None,
            continuation=ContinuationState(
                mode="workroom",
                workroom_id="best-of-n",
                workroom_participants=("coder", "coder", "reviewer"),
                progress_index=1,
                member_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workroom_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
        )]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Prior workroom outputs:", reviewer_prompt)
        self.assertIn("coder[1]: Idea A", reviewer_prompt)
        self.assertIn("coder[2]: Idea B", reviewer_prompt)

    async def test_stage_prompt_receives_workroom_assignment(self) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Reviewer chose candidate 2"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyStagedWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [event async for event in executor.execute(
            request=request,
            definition=_best_of_n_review_definition(),
            goal="Pick the best one",
            workroom_request="Compare the two alternatives and choose one winner.",
            state=state,
            result_sink=lambda _result: None,
            continuation=ContinuationState(
                mode="workroom",
                workroom_id="best-of-n",
                workroom_participants=("coder", "coder", "reviewer"),
                progress_index=1,
                member_index=0,
                agent_id="reviewer",
                goal="Pick the best one",
                current_brief="coder[1]: Idea A\ncoder[2]: Idea B",
                workroom_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
            ),
        )]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Current workroom assignment:", reviewer_prompt)
        self.assertIn(
            "Compare the two alternatives and choose one winner.",
            reviewer_prompt,
        )


def _definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="standard-build",
        path=Path("standard-build.md"),
        metadata={
            "id": "standard-build",
            "stages": ["architect", "coder"],
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
            "stages": [["coder", "coder", "coder"]],
        },
        body="## Purpose\nCompare attempts.",
        sections={"Purpose": "Compare attempts."},
    )


def _best_of_n_review_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "stages": [["coder", "coder"], ["reviewer"]],
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
