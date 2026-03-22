from __future__ import annotations

import json
import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentInvocation, AgentRunResult
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream
from ergon_studio.upstream import UpstreamSettings


class ProxyCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_replies_directly_when_orchestrator_just_answers(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker({"orchestrator": ["Hello world"]}),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1, session_id="session_1")
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
            "Hello world",
        )
        self.assertEqual(result.content, "Hello world")

    async def test_stream_turn_opens_channel_and_returns_participant_reply(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Implement it",
                            recipients=["coder"],
                        ),
                        "Final summary",
                    ],
                    "coder": ["Patch applied"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )

        stream = core.stream_turn(request, created_at=1, session_id="session_1")
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("opening channel channel-1 with coder", reasoning.lower())
        self.assertIn("coder: Patch applied", reasoning)
        self.assertEqual(result.content, "Final summary")

    async def test_stream_turn_persists_channels_by_session_id(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start here",
                            recipients=["coder"],
                        ),
                        "Opened",
                        _internal_action(
                            "message_channel",
                            channel="channel-1",
                            message="Continue",
                            recipients=["coder"],
                        ),
                        "Done",
                    ],
                    "coder": ["First pass", "Second pass"],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Start"),),
        )
        second_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Continue"),),
        )

        first_stream = core.stream_turn(
            first_request,
            created_at=1,
            session_id="session_1",
        )
        [event async for event in first_stream]
        await first_stream.get_final_response()

        second_stream = core.stream_turn(
            second_request,
            created_at=2,
            session_id="session_1",
        )
        events = [event async for event in second_stream]
        result = await second_stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("continuing channel channel-1", reasoning.lower())
        self.assertIn("coder: Second pass", reasoning)
        self.assertEqual(result.content, "Done")

    async def test_close_channel_ends_channel_lifecycle(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start",
                            recipients=["coder"],
                        ),
                        "Opened",
                        _internal_action("close_channel", channel="channel-1"),
                        "Closed",
                        _internal_action(
                            "message_channel",
                            channel="channel-1",
                            message="Continue",
                            recipients=["coder"],
                        ),
                    ],
                    "coder": ["First pass"],
                }
            ),
        )
        open_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Start"),),
        )
        close_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Close it"),),
        )
        reopen_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Continue it"),),
        )

        first_stream = core.stream_turn(
            open_request,
            created_at=1,
            session_id="session_1",
        )
        [event async for event in first_stream]
        await first_stream.get_final_response()

        close_stream = core.stream_turn(
            close_request,
            created_at=2,
            session_id="session_1",
        )
        [event async for event in close_stream]
        await close_stream.get_final_response()

        stream = core.stream_turn(
            reopen_request,
            created_at=3,
            session_id="session_1",
        )
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn(
            "unknown channel: channel-1",
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
        )

    async def test_stream_turn_resumes_orchestrator_tool_call_from_pending_store(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            ],
                        },
                        "Done",
                    ],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect it"),),
            tools=(_host_tool("read_file"),),
        )

        first_stream = core.stream_turn(
            first_request,
            created_at=1,
            session_id="session_1",
        )
        first_events = [event async for event in first_stream]
        first_result = await first_stream.get_final_response()

        tool_event = next(
            event for event in first_events if isinstance(event, ProxyToolCallEvent)
        )
        second_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect it"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=first_result.tool_calls,
                ),
                ProxyInputMessage(
                    role="tool",
                    content="file contents",
                    tool_call_id=tool_event.call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )

        second_stream = core.stream_turn(
            second_request,
            created_at=2,
            session_id="session_1",
        )
        [event async for event in second_stream]
        second_result = await second_stream.get_final_response()

        self.assertEqual(second_result.content, "Done")


def _fake_registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
            "coder": DefinitionDocument(
                id="coder",
                path=Path("coder.md"),
                metadata={"id": "coder", "role": "coder"},
                body="## Identity\nCoder.",
                sections={"Identity": "Coder."},
            ),
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
        },
        channel_presets={"standard-build": ("architect", "coder")},
    )


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _internal_action(name: str, **payload):
    return {
        "text": "",
        "tool_calls": [
            {
                "id": f"internal_{name}",
                "name": name,
                "arguments": json.dumps(payload),
            }
        ],
    }


def _fake_agent_invoker(
    responses_by_agent: dict[str, list[str | dict[str, object]]],
):
    counters = {agent_id: 0 for agent_id in responses_by_agent}

    def _invoker(invocation: AgentInvocation):
        index = counters[invocation.agent_id]
        counters[invocation.agent_id] += 1
        response = responses_by_agent[invocation.agent_id][index]
        if isinstance(response, str):
            return _response_stream(response)
        return _response_stream(
            str(response.get("text", "")),
            response=AgentRunResult(
                text=str(response.get("text", "")),
                tool_calls=tuple(
                    ProxyToolCall(
                        id=str(tool_call["id"]),
                        name=str(tool_call["name"]),
                        arguments_json=str(tool_call["arguments"]),
                    )
                    for tool_call in response.get("tool_calls", [])
                ),
            ),
        )

    return _invoker


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
