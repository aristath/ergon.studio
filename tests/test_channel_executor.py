from __future__ import annotations

import unittest

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.channel_executor import ProxyChannelExecutor
from ergon_studio.proxy.channels import ChannelMessage, ChannelSnapshot, OpenChannel
from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.response_stream import ResponseStream


class ChannelExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_collects_one_natural_reply_per_participant(self) -> None:
        streamed_agents: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            return _response_stream(
                {"architect": "Idea", "reviewer": "Refine"}[kwargs["agent_id"]]
            )

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
            channel_id="channel-1",
            name="debate",
            participants=("architect", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            channel=channel,
            channels={"channel-1": channel},
            recipients=("architect", "reviewer"),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(streamed_agents, ["architect", "reviewer"])
        self.assertEqual(
            result,
            (
                ChannelMessage("architect", "Idea"),
                ChannelMessage("reviewer", "Refine"),
            ),
        )
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Idea", reasoning)
        self.assertIn("reviewer: Refine", reasoning)

    async def test_execute_labels_repeated_staffed_instances(self) -> None:
        streamed_prompts: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_prompts.append(kwargs["prompt"])
            return _response_stream("Challenge")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Debate it"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
            channel_id="channel-1",
            name="debate",
            participants=("reviewer", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            channel=channel,
            channels={"channel-1": channel},
            recipients=("reviewer", "reviewer"),
            state=state,
        )

        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertIn("Current staffed instance: reviewer[1]", streamed_prompts[0])
        self.assertIn("Current staffed instance: reviewer[2]", streamed_prompts[1])
        self.assertEqual(
            result,
            (
                ChannelMessage("reviewer[1]", "Challenge"),
                ChannelMessage("reviewer[2]", "Challenge"),
            ),
        )

    async def test_execute_resumes_same_participant_after_tool_result(self) -> None:
        emitted_context: dict[str, object] = {}
        call_count = 0

        def _stream_text_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="call_1",
                                name="read_file",
                                arguments_json='{"path":"main.py"}',
                            ),
                        ),
                    ),
                )
            return _response_stream("Updated main.py")

        def _emit_tool_calls(**kwargs):
            emitted_context.update(kwargs)
            tool_calls = kwargs["tool_calls"]
            return [
                ProxyToolCallEvent(call=tool_calls[0], index=0),
            ]

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Update it"),),
            tools=(_host_tool("read_file"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
            channel_id="channel-1",
            name="ad hoc",
            participants=("coder",),
        )

        first_stream = executor.execute(
            request=request,
            channel=channel,
            channels={"channel-1": channel},
            channel_message="Update it.",
            recipients=("coder",),
            state=state,
        )
        [event async for event in first_stream]

        continuation = emitted_context["continuation"]
        assert isinstance(continuation, ContinuationState)
        self.assertEqual(continuation.actor, "coder")
        self.assertEqual(continuation.active_channel_id, "channel-1")

        resumed_stream = executor.execute(
            request=request,
            channel=OpenChannel(
                channel_id="channel-1",
                name="ad hoc",
                participants=("coder",),
                transcript=[
                    ChannelMessage("orchestrator", "Update it."),
                ],
            ),
            channels={
                "channel-1": OpenChannel(
                    channel_id="channel-1",
                    name="ad hoc",
                    participants=("coder",),
                    transcript=[ChannelMessage("orchestrator", "Update it.")],
                )
            },
            channel_message="Update it.",
            state=ProxyTurnState(),
            continuation=continuation,
            pending=_pending_tool_result("call_1"),
        )
        events = [event async for event in resumed_stream]
        result = await resumed_stream.get_final_response()

        self.assertEqual(call_count, 2)
        self.assertEqual(result, (ChannelMessage("coder", "Updated main.py"),))
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder: Updated", reasoning)

    async def test_execute_allows_multiple_participants_to_request_tools(
        self,
    ) -> None:
        captured_continuations: list[ContinuationState] = []

        def _stream_text_agent(**kwargs):
            agent_id = kwargs["agent_id"]
            return _response_stream(
                "",
                response=AgentRunResult(
                    text="",
                    tool_calls=(
                        ProxyToolCall(
                            id=f"call_{agent_id}",
                            name="read_file",
                            arguments_json='{"path":"main.py"}',
                        ),
                    ),
                ),
            )

        def _emit_tool_calls(**kwargs):
            continuation = kwargs["continuation"]
            assert isinstance(continuation, ContinuationState)
            captured_continuations.append(continuation)
            return [
                ProxyToolCallEvent(call=tool_call, index=index)
                for index, tool_call in enumerate(kwargs["tool_calls"])
            ]

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Inspect it"),),
            tools=(_host_tool("read_file"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
            channel_id="channel-2",
            name="debate",
            participants=("architect", "coder"),
        )

        stream = executor.execute(
            request=request,
            channel=channel,
            channels={"channel-2": channel},
            channel_message="Inspect the current state.",
            recipients=("architect", "coder"),
            state=state,
        )
        events = [event async for event in stream]
        await stream.get_final_response()

        tool_events = [
            event for event in events if isinstance(event, ProxyToolCallEvent)
        ]
        self.assertEqual(len(tool_events), 2)
        self.assertEqual([event.index for event in tool_events], [0, 1])
        self.assertEqual(
            {continuation.actor for continuation in captured_continuations},
            {"architect", "coder"},
        )
        self.assertTrue(
            all(
                continuation.active_channel_id == "channel-2"
                for continuation in captured_continuations
            )
        )

    async def test_channel_prompt_includes_message_transcript_and_team_notes(
        self,
    ) -> None:
        streamed_kwargs: list[dict[str, object]] = []

        def _stream_text_agent(**kwargs):
            streamed_kwargs.append(kwargs)
            return _response_stream("Reviewer chose candidate 2")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Pick the best one"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
            channel_id="channel-7",
            name="debate",
            participants=("reviewer",),
            transcript=[
                ChannelMessage("architect", "Option A"),
                ChannelMessage("coder", "Option B"),
            ],
        )
        stream = executor.execute(
            request=request,
            channel=channel,
            channels={"channel-7": channel},
            channel_message="Choose one clear direction.",
            recipients=("reviewer",),
            state=state,
            continuation=ContinuationState(
                actor="reviewer",
                active_channel_id="channel-7",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-7",
                        name="debate",
                        participants=("reviewer",),
                        transcript=(
                            ChannelMessage("architect", "Option A"),
                            ChannelMessage("coder", "Option B"),
                        ),
                    ),
                ),
                worklog=("researcher: constraints noted",),
            ),
        )

        [event async for event in stream]

        reviewer_prompt = streamed_kwargs[0]["prompt"]
        assert isinstance(reviewer_prompt, str)
        self.assertIn(
            "Recent channel messages are provided separately as conversation history.",
            reviewer_prompt,
        )
        self.assertIn("Recent team notes:", reviewer_prompt)
        self.assertIn("researcher: constraints noted", reviewer_prompt)
        self.assertNotIn("architect: Option A", reviewer_prompt)
        conversation_messages = streamed_kwargs[0]["conversation_messages"]
        assert isinstance(conversation_messages, tuple)
        self.assertEqual(
            conversation_messages,
            (
                ProxyInputMessage(
                    role="user",
                    name="architect",
                    content="Option A",
                ),
                ProxyInputMessage(
                    role="user",
                    name="coder",
                    content="Option B",
                ),
                ProxyInputMessage(
                    role="user",
                    name="orchestrator",
                    content="Choose one clear direction.",
                ),
            ),
        )


def _pending_tool_result(call_id: str):
    from ergon_studio.proxy.continuation import PendingContinuation

    return PendingContinuation(
        state=ContinuationState(actor="coder"),
        tool_states=(ContinuationState(actor="coder"),),
        assistant_message=None,
        tool_results=(
            ProxyInputMessage(
                role="tool",
                content="file contents",
                tool_call_id=call_id,
            ),
        ),
    )


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _no_tool_calls(**_kwargs):
    return []


def _response_stream(
    text: str,
    *,
    response: AgentRunResult | None = None,
) -> ResponseStream[str, AgentRunResult]:
    async def _events():
        if text:
            yield text

    final = response or AgentRunResult(text=text, tool_calls=())
    return ResponseStream(_events(), finalizer=lambda: final)
