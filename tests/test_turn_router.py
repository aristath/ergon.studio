from __future__ import annotations

import unittest

from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.turn_router import ProxyTurnRouter
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState, ProxyTurnState


class TurnRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_plan_routes_delegate_mode(self) -> None:
        calls: list[str] = []

        async def _direct(**_kwargs):
            calls.append("direct")
            yield ProxyContentDeltaEvent("direct")

        async def _delegate(**_kwargs):
            calls.append("delegate")
            yield ProxyContentDeltaEvent("delegate")

        async def _finish(**_kwargs):
            calls.append("finish")
            yield ProxyContentDeltaEvent("finish")

        async def _workflow(**_kwargs):
            calls.append("workflow")
            yield ProxyContentDeltaEvent("workflow")

        async def _playbook_continuation(**_kwargs):
            calls.append("playbook_continuation")
            yield ProxyContentDeltaEvent("playbook_continuation")

        async def _workflow_continuation(**_kwargs):
            calls.append("workflow_continuation")
            yield ProxyContentDeltaEvent("workflow_continuation")

        router = ProxyTurnRouter(
            execute_direct=_direct,
            execute_finish=_finish,
            execute_delegation=_delegate,
            execute_workflow=_workflow,
            execute_playbook_continuation=_playbook_continuation,
            execute_workflow_continuation=_workflow_continuation,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Do it"),),
        )

        events = [
            event
            async for event in router.execute_plan(
                request=request,
                plan=ProxyTurnPlan(mode="delegate", agent_id="coder"),
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(calls, ["delegate"])
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "delegate")

    async def test_execute_plan_routes_continue_playbook_mode(self) -> None:
        calls: list[dict[str, object]] = []

        async def _direct(**kwargs):
            calls.append({"kind": "direct", **kwargs})
            yield ProxyContentDeltaEvent("direct")

        async def _delegate(**kwargs):
            calls.append({"kind": "delegate", **kwargs})
            yield ProxyContentDeltaEvent("delegate")

        async def _finish(**kwargs):
            calls.append({"kind": "finish", **kwargs})
            yield ProxyContentDeltaEvent("finish")

        async def _workflow(**kwargs):
            calls.append({"kind": "workflow", **kwargs})
            yield ProxyContentDeltaEvent("workflow")

        async def _playbook_continuation(**kwargs):
            calls.append({"kind": "playbook_continuation", **kwargs})
            yield ProxyContentDeltaEvent("playbook_continuation")

        async def _workflow_continuation(**kwargs):
            calls.append({"kind": "workflow_continuation", **kwargs})
            yield ProxyContentDeltaEvent("workflow_continuation")

        router = ProxyTurnRouter(
            execute_direct=_direct,
            execute_finish=_finish,
            execute_delegation=_delegate,
            execute_workflow=_workflow,
            execute_playbook_continuation=_playbook_continuation,
            execute_workflow_continuation=_workflow_continuation,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Keep going"),),
        )
        loop_state = ProxyDecisionLoopState(
            goal="Build calculator",
            current_brief="Architecture ready",
            workflow_progress=ContinuationState(
                mode="workflow",
                agent_id="architect",
                workflow_id="standard-build",
            ),
        )

        events = [
            event
            async for event in router.execute_plan(
                request=request,
                plan=ProxyTurnPlan(
                    mode="continue_playbook",
                    workflow_id="standard-build",
                ),
                state=ProxyTurnState(),
                loop_state=loop_state,
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kind"], "playbook_continuation")
        self.assertIs(calls[0]["loop_state"], loop_state)
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "playbook_continuation")

    async def test_execute_continuation_routes_delegate_mode(self) -> None:
        calls: list[dict[str, object]] = []

        async def _direct(**kwargs):
            calls.append({"kind": "direct", **kwargs})
            yield ProxyContentDeltaEvent("direct")

        async def _delegate(**kwargs):
            calls.append({"kind": "delegate", **kwargs})
            yield ProxyContentDeltaEvent("delegate")

        async def _finish(**kwargs):
            calls.append({"kind": "finish", **kwargs})
            yield ProxyContentDeltaEvent("finish")

        async def _workflow(**kwargs):
            calls.append({"kind": "workflow", **kwargs})
            yield ProxyContentDeltaEvent("workflow")

        async def _playbook_continuation(**kwargs):
            calls.append({"kind": "playbook_continuation", **kwargs})
            yield ProxyContentDeltaEvent("playbook_continuation")

        async def _workflow_continuation(**kwargs):
            calls.append({"kind": "workflow_continuation", **kwargs})
            yield ProxyContentDeltaEvent("workflow_continuation")

        router = ProxyTurnRouter(
            execute_direct=_direct,
            execute_finish=_finish,
            execute_delegation=_delegate,
            execute_workflow=_workflow,
            execute_playbook_continuation=_playbook_continuation,
            execute_workflow_continuation=_workflow_continuation,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Do it"),),
        )
        pending = PendingContinuation(
            state=ContinuationState(
                mode="delegate",
                agent_id="coder",
                request_text="Do it",
                current_brief="brief",
            ),
            assistant_message=None,
            tool_results=(),
        )

        events = [
            event
            async for event in router.execute_continuation(
                request=request,
                pending=pending,
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kind"], "delegate")
        self.assertEqual(calls[0]["current_brief"], "brief")
        self.assertIs(calls[0]["pending"], pending)
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "delegate")
