from __future__ import annotations

import asyncio
import json
import time
import unittest

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workroom_executor import ProxyWorkroomExecutor
from ergon_studio.response_stream import ResponseStream


class WorkroomExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_one_full_round_and_keeps_room_open(self) -> None:
        streamed_agents: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            return _response_stream(
                {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]
            )

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()
        stream = executor.execute(
            request=request,
            workroom_name="debate",
            participants=("architect", "reviewer"),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(streamed_agents, ["architect", "reviewer"])
        self.assertEqual(
            result,
            ("architect: Idea", "reviewer: Refine"),
        )
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

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            return _response_stream(
                {
                    "architect": "Idea",
                    "reviewer": "Challenge",
                }[kwargs["agent_id"]]
            )

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Debate it"),),
        )
        state = ProxyTurnState()
        stream = executor.execute(
            request=request,
            workroom_name="debate",
            participants=("architect", "reviewer", "reviewer"),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(streamed_agents, ["architect", "reviewer", "reviewer"])
        self.assertIn("Current staffed instance: reviewer[1]", streamed_prompts[1])
        self.assertIn("instance 1 of 2 staffed reviewers", streamed_prompts[1])
        self.assertIn("Current staffed instance: reviewer[2]", streamed_prompts[2])
        self.assertEqual(
            result,
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

        def _stream_text_agent(**_kwargs):
            return _response_stream(next(coder_outputs), delay=0.05)

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Try a few approaches"),),
        )
        state = ProxyTurnState()
        stream = executor.execute(
            request=request,
            workroom_name="best-of-n",
            participants=("coder", "coder", "coder"),
            state=state,
        )

        started_at = time.perf_counter()
        [event async for event in stream]
        elapsed = time.perf_counter() - started_at
        result = await stream.get_final_response()

        self.assertLess(elapsed, 0.12)
        self.assertEqual(
            result,
            ("coder[1]: Idea A", "coder[2]: Idea B", "coder[3]: Chosen"),
        )

    async def test_execute_falls_back_to_sequential_when_parallel_round_uses_tools(
        self,
    ) -> None:
        call_count = 0

        def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _response_stream(
                    f"Draft {call_count}",
                    response=_fake_response_with_tool_call(
                        kwargs["agent_id"], call_count
                    ),
                )
            return _response_stream(f"Final {call_count}")

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
        stream = executor.execute(
            request=request,
            workroom_name="best-of-n",
            participants=("coder", "coder", "coder"),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("rerunning this staffed group sequentially", reasoning)
        self.assertEqual(call_count, 6)
        self.assertEqual(
            result,
            ("coder[1]: Final 4", "coder[2]: Final 5", "coder[3]: Final 6"),
        )

    async def test_execute_keeps_single_tool_worker_until_it_replies_to_lead_dev(
        self,
    ) -> None:
        call_count = 0

        def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response_stream(
                    "Checked README.md and found the current intro."
                )
            return _response_stream(
                "",
                response=AgentRunResult(
                    text="",
                    tool_calls=(
                        _internal_reply_tool_call(
                            "Updated the README intro and added setup notes."
                        ),
                    ),
                ),
            )

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
        stream = executor.execute(
            request=request,
            workroom_name="ad hoc",
            participants=("coder",),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(call_count, 2)
        self.assertEqual(
            result,
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

        def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            return _response_stream("Reviewer chose candidate 2")

        executor = ProxyWorkroomExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()
        stream = executor.execute(
            request=request,
            workroom_name="debate",
            workroom_message="Choose one clear direction.",
            participants=("reviewer",),
            state=state,
            continuation=_continuation_state(),
        )

        [event async for event in stream]

        reviewer_prompt = streamed_prompts[0]
        self.assertIn("Latest lead-dev message to this workroom:", reviewer_prompt)
        self.assertIn("Choose one clear direction.", reviewer_prompt)
        self.assertIn("Relevant team work so far:", reviewer_prompt)
        self.assertIn("coder[1]: Idea A", reviewer_prompt)
        self.assertIn("coder[2]: Idea B", reviewer_prompt)

def _continuation_state():
    from ergon_studio.proxy.continuation import ContinuationState

    return ContinuationState(
        workroom_name="debate",
        workroom_participants=("reviewer",),
        actor="reviewer",
        worklog=("coder[1]: Idea A", "coder[2]: Idea B"),
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


def _response_stream(
    text: str,
    *,
    response: AgentRunResult | None = None,
    delay: float = 0.0,
) -> ResponseStream[str, AgentRunResult]:
    async def _events():
        if delay:
            await asyncio.sleep(delay)
        if text:
            yield text

    final = response or AgentRunResult(text=text, tool_calls=())
    return ResponseStream(_events(), finalizer=lambda _updates: final)
