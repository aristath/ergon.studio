from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyMoveResult, ProxyTurnState
from ergon_studio.proxy.workroom_executor import ProxyWorkroomExecutor


class WorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_one_full_round_and_keeps_room_open(self) -> None:
        streamed_agents: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            yield {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
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
                definition=_ordered_definition(),
                state=state,
                result_sink=captured.append,
            )
        ]

        self.assertEqual(streamed_agents, ["architect", "reviewer"])
        self.assertEqual(
            captured[0].worklog_lines,
            ("architect: Idea", "reviewer: Refine"),
        )
        self.assertIsNotNone(captured[0].active_workroom)
        assert captured[0].active_workroom is not None
        self.assertEqual(captured[0].active_workroom.workroom_id, "debate")
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

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
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
                definition=_ordered_definition(),
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

    async def test_execute_runs_parallel_same_role_round_concurrently(self) -> None:
        coder_outputs = iter(["Idea A", "Idea B", "Chosen"])

        async def _stream_text_agent(**_kwargs):
            await asyncio.sleep(0.05)
            yield next(coder_outputs)

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        started_at = time.perf_counter()
        [
            event
            async for event in executor.execute(
                request=request,
                definition=_parallel_definition(),
                state=state,
                result_sink=captured.append,
            )
        ]
        elapsed = time.perf_counter() - started_at

        self.assertLess(elapsed, 0.12)
        self.assertEqual(
            captured[0].worklog_lines,
            ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
        )

    async def test_execute_falls_back_to_sequential_when_parallel_round_uses_tools(
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

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
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
                definition=_parallel_definition(),
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

    async def test_execute_keeps_single_tool_worker_until_it_replies_to_lead_dev(
        self,
    ) -> None:
        call_count = 0

        async def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Checked README.md and found the current intro."
                return
            kwargs["final_response_sink"](
                AgentRunResult(
                    text="",
                    tool_calls=(
                        _internal_reply_tool_call(
                            "Updated the README intro and added setup notes."
                        ),
                    ),
                )
            )
            if False:
                yield ""

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(
                ProxyInputMessage(
                    role="user", content="Update the README and improve setup docs"
                ),
            ),
            tools=(_host_tool("read_file"), _host_tool("write_file")),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute(
                request=request,
                definition=_ad_hoc_solo_definition(),
                state=state,
                result_sink=captured.append,
            )
        ]

        self.assertEqual(call_count, 2)
        self.assertEqual(
            captured[0].worklog_lines,
            (
                "coder: Checked README.md and found the current intro.",
                "coder: Updated the README intro and added setup notes.",
            ),
        )
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder: Checked README.md", reasoning)

    async def test_round_prompt_receives_workroom_assignment_and_prior_outputs(
        self,
    ) -> None:
        streamed_prompts: list[str] = []

        async def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            yield "Reviewer chose candidate 2"

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()

        [
            event
            async for event in executor.execute(
                request=request,
                definition=_ordered_definition(),
                workroom_message="Choose one clear direction.",
                participants=("reviewer",),
                state=state,
                result_sink=lambda _result: None,
                continuation=_continuation_state(),
            )
        ]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Latest lead-dev message to this workroom:", reviewer_prompt)
        self.assertIn("Choose one clear direction.", reviewer_prompt)
        self.assertIn("Relevant team work so far:", reviewer_prompt)
        self.assertIn("coder[1]: Idea A", reviewer_prompt)
        self.assertIn("coder[2]: Idea B", reviewer_prompt)


def _ordered_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="debate",
        path=Path("debate.md"),
        metadata={
            "id": "debate",
            "participants": ["architect", "reviewer"],
        },
        body="## Purpose\nDebate.",
        sections={"Purpose": "Debate."},
    )


def _parallel_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="best-of-n",
        path=Path("best-of-n.md"),
        metadata={
            "id": "best-of-n",
            "participants": ["coder", "coder", "coder"],
        },
        body="## Purpose\nCompare attempts.",
        sections={"Purpose": "Compare attempts."},
    )


def _ad_hoc_solo_definition() -> DefinitionDocument:
    return DefinitionDocument(
        id="__ad_hoc__",
        path=Path("__ad_hoc__.md"),
        metadata={
            "id": "__ad_hoc__",
            "participants": ["coder"],
        },
        body="## Purpose\nSolo implementation.",
        sections={"Purpose": "Solo implementation."},
    )


def _continuation_state():
    from ergon_studio.proxy.continuation import ContinuationState

    return ContinuationState(
        mode="workroom",
        workroom_id="debate",
        workroom_participants=("reviewer",),
        member_index=0,
        agent_id="reviewer",
        round_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
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
    return AgentRunResult(
        text="",
        tool_calls=(
            ProxyToolCall(
                id=f"call_{call_index}",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
        ),
    )


def _internal_reply_tool_call(message: str) -> ProxyToolCall:
    return ProxyToolCall(
        id="internal_reply_lead_dev",
        name="reply_lead_dev",
        arguments_json='{"message":' + json.dumps(message) + "}",
    )


def _no_tool_calls(**_kwargs):
    return []
