from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.proxy.workroom_dispatcher import ProxyWorkroomDispatcher
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class WorkroomDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_workroom_reports_unknown_workroom(self) -> None:
        dispatcher = ProxyWorkroomDispatcher(
            _registry(),
            execute_staged_workroom=_empty_handler,
            execute_discussion_workroom=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workroom(
                request=request,
                workroom_id="missing",
                goal="Build it",
                state=state,
            )
        ]

        self.assertEqual(state.finish_reason, "error")
        self.assertEqual(state.content, "Unknown workroom: missing")
        self.assertEqual(
            [type(event) for event in events],
            [ProxyContentDeltaEvent],
        )

    async def test_execute_workroom_dispatches_staged_handler(self) -> None:
        calls: list[tuple[str, str]] = []

        async def _staged_handler(**kwargs):
            calls.append((kwargs["definition"].id, kwargs["goal"]))
            yield ProxyContentDeltaEvent("done")

        dispatcher = ProxyWorkroomDispatcher(
            _registry(),
            execute_staged_workroom=_staged_handler,
            execute_discussion_workroom=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workroom(
                request=request,
                workroom_id="standard-build",
                goal="Build it",
                state=state,
            )
        ]

        self.assertEqual(calls, [("standard-build", "Build it")])
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyContentDeltaEvent)
        self.assertIn(
            "opening workroom standard-build",
            state.reasoning.lower(),
        )

    async def test_execute_workroom_builds_ad_hoc_workroom_definition(self) -> None:
        calls: list[tuple[str, str, tuple[str, ...]]] = []

        async def _discussion_handler(**kwargs):
            calls.append(
                (
                    kwargs["definition"].id,
                    kwargs["goal"],
                    tuple(kwargs["definition"].metadata["turns"]),
                )
            )
            yield ProxyContentDeltaEvent("done")

        dispatcher = ProxyWorkroomDispatcher(
            _registry(),
            execute_staged_workroom=_empty_handler,
            execute_discussion_workroom=_discussion_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Brainstorm it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workroom(
                request=request,
                workroom_id="ad-hoc-workroom",
                participants=("architect", "coder", "critic"),
                goal="Brainstorm it",
                state=state,
            )
        ]

        self.assertEqual(
            calls,
            [("ad-hoc-workroom", "Brainstorm it", ("architect", "coder", "critic"))],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyContentDeltaEvent)
        self.assertIn("opening an ad hoc workroom", state.reasoning.lower())

    async def test_execute_workroom_builds_parallel_attempt_ad_hoc_workroom(
        self,
    ) -> None:
        calls: list[tuple[str, str, tuple[str, ...], str]] = []

        async def _staged_handler(**kwargs):
            calls.append(
                (
                    kwargs["definition"].id,
                    kwargs["goal"],
                    tuple(kwargs["definition"].metadata["stages"]),
                    str(kwargs["definition"].metadata["shape"]),
                )
            )
            yield ProxyContentDeltaEvent("done")

        dispatcher = ProxyWorkroomDispatcher(
            _registry(),
            execute_staged_workroom=_staged_handler,
            execute_discussion_workroom=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(
                ProxyInputMessage(
                    role="user",
                    content="Try three implementations",
                ),
            ),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workroom(
                request=request,
                workroom_id="ad-hoc-workroom",
                participants=("coder", "coder", "coder"),
                goal="Try three implementations",
                state=state,
            )
        ]

        self.assertEqual(
            calls,
            [
                (
                    "ad-hoc-workroom",
                    "Try three implementations",
                    ("coder", "coder", "coder"),
                    "staged",
                )
            ],
        )
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyContentDeltaEvent)


async def _empty_handler(**_kwargs):
    sentinel = _kwargs.get("__never__")
    if sentinel is not None:
        yield ProxyContentDeltaEvent("")


def _registry() -> RuntimeRegistry:
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
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
        },
        workroom_definitions={
            "standard-build": DefinitionDocument(
                id="standard-build",
                path=Path("standard-build.md"),
                metadata={
                    "id": "standard-build",
                    "shape": "staged",
                    "stages": ["architect"],
                },
                body="## Purpose\nBuild.",
                sections={"Purpose": "Build."},
            ),
        },
    )
