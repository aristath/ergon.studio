from __future__ import annotations

import json
import unittest

from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.channel_executor import (
    MAX_CHANNEL_TRANSCRIPT_MESSAGES,
    ProxyChannelExecutor,
    _channel_conversation_messages,
)
from ergon_studio.proxy.channels import Channel, ChannelMessage
from ergon_studio.proxy.continuation import PendingToolContext
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
    async def test_execute_invokes_only_explicit_recipients(self) -> None:
        streamed_agents: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            return _response_stream("Refine")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("architect", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Architect, take a look.",
            recipients=("architect",),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(streamed_agents, ["architect"])
        self.assertEqual(result, (ChannelMessage("architect", "Refine"),))
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Refine", reasoning)

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
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("reviewer", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Both reviewers, challenge this.",
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

    async def test_multi_recipient_message_replies_in_explicit_order(self) -> None:
        captured_conversations: list[tuple[ProxyInputMessage, ...]] = []

        def _stream_text_agent(**kwargs):
            captured_conversations.append(kwargs["conversation_messages"])
            if kwargs["agent_id"] == "reviewer":
                return _response_stream("Reviewer critique")
            return _response_stream("Architect direction")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Discuss it"),),
        )
        state = ProxyTurnState()
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("architect", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Reviewer, then architect, weigh in.",
            recipients=("reviewer", "architect"),
            state=state,
        )

        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(
            result,
            (
                ChannelMessage("reviewer", "Reviewer critique"),
                ChannelMessage("architect", "Architect direction"),
            ),
        )
        self.assertEqual(
            [message.name for message in captured_conversations[0]],
            ["orchestrator"],
        )
        self.assertEqual(
            [message.name for message in captured_conversations[1]],
            ["orchestrator"],
        )

    async def test_execute_can_target_specific_staffed_label(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            return _response_stream("Second reviewer only")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Challenge it"),),
        )
        state = ProxyTurnState()
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("reviewer", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Reviewer[2], take this one.",
            recipients=("reviewer[2]",),
            state=state,
        )

        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(streamed_agents, ["reviewer"])
        self.assertEqual(len(streamed_prompts), 1)
        self.assertIn("Current staffed instance: reviewer[2]", streamed_prompts[0])
        self.assertEqual(
            result,
            (ChannelMessage("reviewer[2]", "Second reviewer only"),),
        )

    async def test_execute_rejects_ambiguous_duplicate_recipient(self) -> None:
        executor = ProxyChannelExecutor(
            stream_text_agent=lambda **kwargs: _response_stream("unused"),
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Challenge it"),),
        )
        state = ProxyTurnState()
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("reviewer", "reviewer"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Reviewer, take this one.",
            recipients=("reviewer",),
            state=state,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate staffed recipients must be addressed explicitly for reviewer",
        ):
            [event async for event in stream]

    async def test_execute_routes_participant_message_to_next_recipient(self) -> None:
        agent_order: list[str] = []

        def _stream_text_agent(**kwargs):
            agent_order.append(kwargs["agent_id"])
            if kwargs["agent_id"] == "architect":
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="internal_1",
                                name="message_channel",
                                arguments_json=json.dumps(
                                    {
                                        "message": "Coder, implement this plan.",
                                        "recipients": ["coder"],
                                    }
                                ),
                            ),
                        ),
                    ),
                )
            return _response_stream("Implemented.")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("architect", "coder"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Architect, decide the direction.",
            recipients=("architect",),
            state=state,
        )

        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(agent_order, ["architect", "coder"])
        self.assertEqual(
            result,
            (
                ChannelMessage("architect", "Coder, implement this plan."),
                ChannelMessage("coder", "Implemented."),
            ),
        )
        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Coder, implement this plan.", reasoning)
        self.assertIn("coder: Implemented.", reasoning)

    async def test_execute_rejects_participant_message_to_non_staffed_recipient(
        self,
    ) -> None:
        def _stream_text_agent(**kwargs):
            return _response_stream(
                "",
                response=AgentRunResult(
                    text="",
                    tool_calls=(
                        ProxyToolCall(
                            id="internal_1",
                            name="message_channel",
                            arguments_json=json.dumps(
                                {
                                    "message": "QA, check this.",
                                    "recipients": ["qa"],
                                }
                            ),
                        ),
                    ),
                ),
            )

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=Channel(
                channel_id="channel-1",
                name="debate",
                participants=("architect", "coder"),
            ),
            channel_message="Architect, decide the direction.",
            recipients=("architect",),
            state=ProxyTurnState(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "channel recipients are not staffed in this channel: qa",
        ):
            [event async for event in stream]

    async def test_execute_resumes_same_participant_after_tool_result(self) -> None:
        captured_pending_args: list[tuple[str, str, str | None]] = []
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
            captured_pending_args.append(
                (
                    kwargs["session_id"],
                    kwargs["actor"],
                    kwargs.get("active_channel_id"),
                )
            )
            tool_calls = kwargs["tool_calls"]
            return [ProxyToolCallEvent(call=tool_calls[0], index=0)]

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
        channel = Channel(
            channel_id="channel-1",
            name="ad hoc",
            participants=("coder",),
        )

        first_stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Update it.",
            recipients=("coder",),
            state=state,
        )
        [event async for event in first_stream]
        self.assertEqual(
            captured_pending_args[0],
            ("session_1", "coder", "channel-1"),
        )

        resumed_stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=Channel(
                channel_id="channel-1",
                name="ad hoc",
                participants=("coder",),
                transcript=[ChannelMessage("orchestrator", "Update it.")],
            ),
            state=ProxyTurnState(),
            pending=(
                PendingToolContext(
                    pending_id="pending_1",
                    session_id="session_1",
                    actor="coder",
                    active_channel_id="channel-1",
                    tool_calls=(
                        ProxyToolCall(
                            id="call_1",
                            name="read_file",
                            arguments_json='{"path":"main.py"}',
                        ),
                    ),
                    tool_results=(
                        ProxyInputMessage(
                            role="tool",
                            content="file contents",
                            tool_call_id="ergon:3:pending_1:call_1",
                        ),
                    ),
                ),
            ),
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
        self.assertIn("coder: Updated main.py", reasoning)

    async def test_pending_resumes_update_transcript_before_next_actor(self) -> None:
        captured_conversations: list[tuple[ProxyInputMessage, ...]] = []

        def _stream_text_agent(**kwargs):
            captured_conversations.append(kwargs["conversation_messages"])
            if kwargs["agent_id"] == "architect":
                return _response_stream("Architecture checked")
            return _response_stream("Implementation updated")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Continue"),),
            tools=(_host_tool("read_file"),),
        )
        channel = Channel(
            channel_id="channel-1",
            name="debate",
            participants=("architect", "coder"),
            transcript=[ChannelMessage("orchestrator", "Continue.")],
        )

        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            state=ProxyTurnState(),
            pending=(
                PendingToolContext(
                    pending_id="pending_a",
                    session_id="session_1",
                    actor="architect",
                    active_channel_id="channel-1",
                    tool_calls=(
                        ProxyToolCall(
                            id="call_a",
                            name="read_file",
                            arguments_json='{"path":"plan.md"}',
                        ),
                    ),
                    tool_results=(
                        ProxyInputMessage(
                            role="tool",
                            content="plan text",
                            tool_call_id="ergon:3:pending_a:call_a",
                        ),
                    ),
                ),
                PendingToolContext(
                    pending_id="pending_b",
                    session_id="session_1",
                    actor="coder",
                    active_channel_id="channel-1",
                    tool_calls=(
                        ProxyToolCall(
                            id="call_b",
                            name="read_file",
                            arguments_json='{"path":"main.py"}',
                        ),
                    ),
                    tool_results=(
                        ProxyInputMessage(
                            role="tool",
                            content="main text",
                            tool_call_id="ergon:3:pending_b:call_b",
                        ),
                    ),
                ),
            ),
        )

        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(
            result,
            (
                ChannelMessage("architect", "Architecture checked"),
                ChannelMessage("coder", "Implementation updated"),
            ),
        )
        self.assertEqual(
            [message.name for message in captured_conversations[0]],
            ["orchestrator"],
        )
        self.assertEqual(
            [message.name for message in captured_conversations[1]],
            ["orchestrator", "architect"],
        )

    async def test_participants_in_same_delivery_see_identical_transcript_snapshot(
        self,
    ) -> None:
        captured_conversations: list[tuple[ProxyInputMessage, ...]] = []

        def _stream_text_agent(**kwargs):
            captured_conversations.append(kwargs["conversation_messages"])
            return _response_stream(f"Solution from {kwargs['agent_id']}")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        channel = Channel(
            channel_id="channel-1",
            name="best-of-n",
            participants=("coder", "coder"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Both coders, solve independently.",
            recipients=("coder[1]", "coder[2]"),
            state=ProxyTurnState(),
        )

        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(len(captured_conversations), 2)
        # Both participants see the same pre-delivery snapshot
        self.assertEqual(captured_conversations[0], captured_conversations[1])
        # Neither sees the other's response
        for conv in captured_conversations:
            self.assertFalse(any(m.name and "coder" in m.name for m in conv))

    async def test_sequential_delivery_via_message_channel_sees_prior_results(
        self,
    ) -> None:
        captured_conversations: list[tuple[ProxyInputMessage, ...]] = []

        def _stream_text_agent(**kwargs):
            captured_conversations.append(kwargs["conversation_messages"])
            if kwargs["agent_id"] == "architect":
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="internal_1",
                                name="message_channel",
                                arguments_json='{"message":"Coder, build it.","recipients":["coder"]}',
                            ),
                        ),
                    ),
                )
            return _response_stream("Built.")

        executor = ProxyChannelExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        channel = Channel(
            channel_id="channel-1",
            name="build",
            participants=("architect", "coder"),
        )
        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            channel_message="Architect, plan it.",
            recipients=("architect",),
            state=ProxyTurnState(),
        )

        [event async for event in stream]
        await stream.get_final_response()

        # Coder was invoked in a second delivery and should see architect's message
        coder_conversation = captured_conversations[1]
        self.assertTrue(
            any(m.name == "architect" for m in coder_conversation),
            "coder should see architect's message_channel message",
        )

    async def test_pending_resume_fails_when_actor_is_not_staffed(self) -> None:
        executor = ProxyChannelExecutor(
            stream_text_agent=lambda **kwargs: _response_stream("unused"),
            emit_tool_calls=_no_tool_calls,

        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Continue"),),
            tools=(_host_tool("read_file"),),
        )
        channel = Channel(
            channel_id="channel-1",
            name="ad hoc",
            participants=("coder",),
            transcript=[ChannelMessage("orchestrator", "Continue.")],
        )

        stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=channel,
            state=ProxyTurnState(),
            pending=(
                PendingToolContext(
                    pending_id="pending_1",
                    session_id="session_1",
                    actor="architect",
                    active_channel_id="channel-1",
                    tool_calls=(
                        ProxyToolCall(
                            id="call_1",
                            name="read_file",
                            arguments_json='{"path":"plan.md"}',
                        ),
                    ),
                    tool_results=(
                        ProxyInputMessage(
                            role="tool",
                            content="plan text",
                            tool_call_id="ergon:3:pending_1:call_1",
                        ),
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "pending actor is not staffed in this channel: architect",
        ):
            [event async for event in stream]


class ChannelConversationMessagesTests(unittest.TestCase):
    def test_channel_conversation_messages_caps_transcript(self) -> None:
        transcript = tuple(
            ChannelMessage(author="orchestrator", content=f"msg {i}")
            for i in range(60)
        )
        result = _channel_conversation_messages(
            channel_transcript=transcript,
            participant_label="coder",
        )
        self.assertLessEqual(len(result), MAX_CHANNEL_TRANSCRIPT_MESSAGES)

    def test_channel_conversation_messages_keeps_latest(self) -> None:
        transcript = tuple(
            ChannelMessage(author="orchestrator", content=f"msg {i}")
            for i in range(60)
        )
        result = _channel_conversation_messages(
            channel_transcript=transcript,
            participant_label="coder",
        )
        # The last message in the transcript must appear in the result
        self.assertEqual(result[-1].content, "msg 59")
        # The very first message should not appear (it's beyond the window)
        self.assertFalse(any(m.content == "msg 0" for m in result))



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
