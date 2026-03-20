from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from agent_framework import ResponseStream

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyInputMessage, ProxyReasoningDeltaEvent, ProxyTurnRequest
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
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def create_session(self, *, session_id: str):
        return object()

    def run(self, _messages, *, session, stream: bool = False):
        text = self._responses.pop(0)
        if not stream:
            return _immediate_response(text)
        parts = [piece for piece in text.split(" ") if piece]

        async def _events():
            for index, part in enumerate(parts):
                suffix = " " if index < len(parts) - 1 else ""
                yield type("Update", (), {"text": part + suffix})()

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: type("Response", (), {"text": text})(),
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


async def _immediate_response(text: str):
    return type("Response", (), {"text": text})()


if __name__ == "__main__":
    unittest.main()
