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
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class WorkflowDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_workflow_reports_unknown_workflow(self) -> None:
        dispatcher = ProxyWorkflowDispatcher(
            _registry(),
            execute_grouped_workflow=_empty_handler,
            execute_group_chat_workflow=_empty_handler,
            execute_magentic_workflow=_empty_handler,
            execute_handoff_workflow=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workflow(
                request=request,
                workflow_id="missing",
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

    async def test_execute_workflow_dispatches_grouped_handler(self) -> None:
        calls: list[tuple[str, str]] = []

        async def _grouped_handler(**kwargs):
            calls.append((kwargs["definition"].id, kwargs["goal"]))
            yield ProxyContentDeltaEvent("done")

        dispatcher = ProxyWorkflowDispatcher(
            _registry(),
            execute_grouped_workflow=_grouped_handler,
            execute_group_chat_workflow=_empty_handler,
            execute_magentic_workflow=_empty_handler,
            execute_handoff_workflow=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workflow(
                request=request,
                workflow_id="standard-build",
                goal="Build it",
                state=state,
            )
        ]

        self.assertEqual(calls, [("standard-build", "Build it")])
        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertIsInstance(events[1], ProxyContentDeltaEvent)
        self.assertIn(
            "opening workroom template standard-build",
            state.reasoning.lower(),
        )

    async def test_execute_workflow_builds_ad_hoc_workroom_definition(self) -> None:
        calls: list[tuple[str, str, tuple[str, ...]]] = []

        async def _magentic_handler(**kwargs):
            calls.append(
                (
                    kwargs["definition"].id,
                    kwargs["goal"],
                    tuple(kwargs["definition"].metadata["steps"]),
                )
            )
            yield ProxyContentDeltaEvent("done")

        dispatcher = ProxyWorkflowDispatcher(
            _registry(),
            execute_grouped_workflow=_empty_handler,
            execute_group_chat_workflow=_empty_handler,
            execute_magentic_workflow=_magentic_handler,
            execute_handoff_workflow=_empty_handler,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Brainstorm it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in dispatcher.execute_workflow(
                request=request,
                workflow_id="ad-hoc-workroom",
                specialists=("architect", "coder", "critic"),
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
        workflow_definitions={
            "standard-build": DefinitionDocument(
                id="standard-build",
                path=Path("standard-build.md"),
                metadata={
                    "id": "standard-build",
                    "orchestration": "sequential",
                    "steps": ["architect"],
                },
                body="## Purpose\nBuild.",
                sections={"Purpose": "Build."},
            ),
        },
    )
