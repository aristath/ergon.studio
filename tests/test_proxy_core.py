from __future__ import annotations

import unittest
from pathlib import Path

from agent_framework import ResponseStream

from ergon_studio.proxy.continuation import decode_continuation_from_tool_call_id
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyInputMessage, ProxyReasoningDeltaEvent, ProxyToolCallEvent, ProxyTurnRequest
from ergon_studio.registry import RuntimeRegistry


class ProxyCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_handles_direct_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": ["{\"mode\":\"act\"}", "Hello world"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertTrue(any(isinstance(event, ProxyReasoningDeltaEvent) for event in events))
        self.assertEqual("".join(event.delta for event in events if isinstance(event, ProxyContentDeltaEvent)), "Hello world")
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.mode, "act")

    async def test_stream_turn_handles_delegate_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"delegate\",\"agent_id\":\"coder\",\"request\":\"Implement it\"}",
                "Final summary",
            ],
            "coder": ["Patch", " applied"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("delegating", reasoning.lower())
        self.assertIn("coder: Patch", reasoning)
        self.assertEqual(result.content, "Final summary")
        self.assertEqual(result.mode, "delegate")

    async def test_stream_turn_handles_workflow_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"standard-build\",\"goal\":\"Build calculator\"}",
                "Workflow final summary",
            ],
            "architect": ["Plan"],
            "coder": ["Built"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("workflow standard-build", reasoning)
        self.assertIn("architect: Plan", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(result.content, "Workflow final summary")
        self.assertEqual(result.mode, "workflow")

    async def test_stream_turn_emits_tool_call_events_for_direct_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"act\"}",
                {
                    "text": "",
                    "tool_calls": [
                        {"id": "call_1", "name": "read_file", "arguments": "{\"path\":\"main.py\"}"},
                    ],
                },
            ],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        tool_events = [event for event in events if isinstance(event, ProxyToolCallEvent)]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call.name, "read_file")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        continuation = decode_continuation_from_tool_call_id(result.tool_calls[0].id)
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation.mode, "act")

    async def test_stream_turn_resumes_workflow_from_tool_result(self) -> None:
        first_core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"standard-build\",\"goal\":\"Build calculator\"}",
            ],
            "architect": [
                {
                    "text": "",
                    "tool_calls": [
                        {"id": "call_arch_1", "name": "read_file", "arguments": "{\"path\":\"main.py\"}"},
                    ],
                }
            ],
        }))
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
            tools=(_host_tool("read_file"),),
        )
        first_stream = first_core.stream_turn(first_request, created_at=1)
        first_events = [event async for event in first_stream]
        first_result = await first_stream.get_final_response()
        tool_call = next(event.call for event in first_events if isinstance(event, ProxyToolCallEvent))

        resumed_core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "architect": ["Architecture plan"],
            "coder": ["Built feature"],
            "orchestrator": ["Workflow final summary"],
        }))
        resumed_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=(tool_call,),
                ),
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=tool_call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )
        resumed_stream = resumed_core.stream_turn(resumed_request, created_at=2)
        resumed_events = [event async for event in resumed_stream]
        resumed_result = await resumed_stream.get_final_response()

        reasoning = "".join(event.delta for event in resumed_events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("continuing workflow standard-build with architect", reasoning)
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(resumed_result.content, "Workflow final summary")
        self.assertEqual(resumed_result.finish_reason, "stop")


class _FakeRegistry:
    def __init__(self) -> None:
        self.inner = RuntimeRegistry(
            config={},
            agent_definitions={
                "orchestrator": DefinitionDocument(
                    id="orchestrator",
                    path=Path("orchestrator.md"),
                    metadata={"id": "orchestrator", "role": "orchestrator"},
                    body="## Identity\nLead engineer.",
                    sections={"Identity": "Lead engineer."},
                ),
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
            },
            workflow_definitions={
                "standard-build": DefinitionDocument(
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
            },
        )

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class _FakeAgent:
    def __init__(self, responses) -> None:
        self._responses = list(responses)

    def create_session(self, *, session_id: str):
        return object()

    def run(self, _messages, *, session, stream: bool = False, tools=None):
        raw = self._responses.pop(0)
        if isinstance(raw, str):
            payload = {"text": raw, "tool_calls": []}
        else:
            payload = raw
        text = payload.get("text", "")
        tool_calls = payload.get("tool_calls", [])
        if not stream:
            return _immediate_response(text, tool_calls=tool_calls)
        parts = [piece for piece in text.split(" ") if piece]

        async def _events():
            for index, part in enumerate(parts):
                suffix = " " if index < len(parts) - 1 else ""
                yield type("Update", (), {"text": part + suffix})()

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: _response_object(text, tool_calls=tool_calls),
        )


def _fake_agent_builder(mapping: dict[str, list[str]]):
    remaining = {agent_id: list(responses) for agent_id, responses in mapping.items()}

    def _build(_registry, agent_id: str, **_kwargs):
        queue = remaining[agent_id]
        if not queue:
            raise AssertionError(f"no fake responses left for {agent_id}")
        return _FakeAgent([queue.pop(0)])

    return _build


def _fake_registry():
    return _FakeRegistry()


def _response_object(text: str, *, tool_calls: list[dict[str, str]]):
    contents = [type("Content", (), {"type": "text", "text": text})()] if text else []
    for tool_call in tool_calls:
        contents.append(
            type(
                "Content",
                (),
                {
                    "type": "function_call",
                    "call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"],
                },
            )()
        )
    message = type("Message", (), {"contents": contents})()
    return type("Response", (), {"text": text, "messages": [message]})()


async def _immediate_response(text: str, *, tool_calls: list[dict[str, str]]):
    return _response_object(text, tool_calls=tool_calls)


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )


if __name__ == "__main__":
    unittest.main()
