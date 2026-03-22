from __future__ import annotations

import json
import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.channel_executor import ProxyChannelExecutor
from ergon_studio.proxy.channels import ChannelMessage, OpenChannel
from ergon_studio.proxy.continuation import PendingContinuation, PendingToolContext
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.pending_store import PendingSeed
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream
from ergon_studio.upstream import UpstreamSettings


class ChannelExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_invokes_only_explicit_recipients(self) -> None:
        streamed_agents: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            return _response_stream("Refine")

        executor = ProxyChannelExecutor(
            registry=_registry(),
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
            registry=_registry(),
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

    async def test_execute_can_target_specific_staffed_label(self) -> None:
        streamed_agents: list[str] = []
        streamed_prompts: list[str] = []

        def _stream_text_agent(**kwargs):
            streamed_agents.append(kwargs["agent_id"])
            streamed_prompts.append(kwargs["prompt"])
            return _response_stream("Second reviewer only")

        executor = ProxyChannelExecutor(
            registry=_registry(),
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Challenge it"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
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
            registry=_registry(),
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_no_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()
        channel = OpenChannel(
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

    async def test_execute_resumes_same_participant_after_tool_result(self) -> None:
        captured_seeds: list[PendingSeed] = []
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
            continuation = kwargs["continuation"]
            assert isinstance(continuation, PendingSeed)
            captured_seeds.append(continuation)
            tool_calls = kwargs["tool_calls"]
            return [ProxyToolCallEvent(call=tool_calls[0], index=0)]

        executor = ProxyChannelExecutor(
            registry=_registry(),
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
            session_id="session_1",
            channel=channel,
            channel_message="Update it.",
            recipients=("coder",),
            state=state,
        )
        [event async for event in first_stream]
        self.assertEqual(captured_seeds[0].actor, "coder")
        self.assertEqual(captured_seeds[0].active_channel_id, "channel-1")

        resumed_stream = executor.execute(
            request=request,
            session_id="session_1",
            channel=OpenChannel(
                channel_id="channel-1",
                name="ad hoc",
                participants=("coder",),
                transcript=[ChannelMessage("orchestrator", "Update it.")],
            ),
            state=ProxyTurnState(),
            pending=PendingContinuation(
                session_id="session_1",
                items=(
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


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
        agent_definitions={
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
            "coder": DefinitionDocument(
                id="coder",
                path=Path("coder.md"),
                metadata={"id": "coder", "role": "coder"},
                body="## Identity\nCoder.",
                sections={"Identity": "Coder."},
            ),
            "reviewer": DefinitionDocument(
                id="reviewer",
                path=Path("reviewer.md"),
                metadata={"id": "reviewer", "role": "reviewer"},
                body="## Identity\nReviewer.",
                sections={"Identity": "Reviewer."},
            ),
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead dev.",
                sections={"Identity": "Lead dev."},
            ),
        },
        channel_presets={},
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
