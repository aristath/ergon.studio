from __future__ import annotations

import unittest
from collections.abc import AsyncIterator, Callable
from typing import cast

from ergon_studio.proxy.continuation import ContinuationState, PendingContinuation
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState, ProxyTurnState
from ergon_studio.proxy.workflow_dispatcher import ProxyWorkflowDispatcher
from ergon_studio.proxy.workflow_request_executor import (
    ProxyWorkflowRequestExecutor,
)


class WorkflowRequestExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_workflow_uses_plan_goal_or_latest_user_text(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workflow_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        events = [
            event
            async for event in executor.execute_workflow(
                request=request,
                plan=ProxyTurnPlan(mode="workflow", workflow_id="standard-build"),
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["workflow_id"], "standard-build")
        self.assertEqual(calls[0]["specialists"], ())
        self.assertEqual(calls[0]["goal"], "Build it")
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "workflow")

    async def test_execute_workflow_forwards_staffed_specialists(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workflow_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        [event async for event in executor.execute_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="workflow",
                workflow_id="standard-build",
                specialists=("coder", "reviewer"),
                specialist_counts=(("coder", 3),),
            ),
            state=ProxyTurnState(),
        )]

        self.assertEqual(calls[0]["specialists"], ("coder", "reviewer"))
        self.assertEqual(calls[0]["specialist_counts"], (("coder", 3),))

    async def test_execute_workflow_forwards_playbook_request(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workflow_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        [event async for event in executor.execute_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="workflow",
                workflow_id="best-of-n",
                playbook_request="Compare three implementations and pick one.",
            ),
            state=ProxyTurnState(),
        )]

        self.assertEqual(
            calls[0]["workflow_request"],
            "Compare three implementations and pick one.",
        )

    async def test_execute_workflow_continuation_forwards_pending_state(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        pending = PendingContinuation(
            state=ContinuationState(
                mode="workflow",
                agent_id="coder",
                workflow_id="standard-build",
            ),
            assistant_message=None,
            tool_results=(),
        )

        events = [
            event
            async for event in executor.execute_workflow_continuation(
                request=request,
                continuation=pending.state,
                pending=pending,
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["pending"], pending)
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "continued")

    async def test_execute_active_workflow_uses_loop_state_progress(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
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
            async for event in executor.execute_active_workflow(
                request=request,
                state=ProxyTurnState(),
                loop_state=loop_state,
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["continuation"], loop_state.workflow_progress)
        self.assertIsNone(calls[0]["pending"])
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "continued")

    async def test_execute_active_workflow_can_override_staffing_from_plan(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
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
                workflow_specialists=("architect", "coder"),
            ),
        )

        [event async for event in executor.execute_active_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_playbook",
                workflow_id="standard-build",
                specialists=("coder",),
                specialist_counts=(("coder", 3),),
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(continuation.workflow_specialists, ("coder",))
        self.assertEqual(continuation.workflow_specialist_counts, (("coder", 3),))

    async def test_execute_active_workflow_can_augment_staffing_from_plan(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
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
                workflow_specialists=("architect", "coder"),
                workflow_specialist_counts=(("coder", 2),),
            ),
        )

        [event async for event in executor.execute_active_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_playbook",
                workflow_id="standard-build",
                staffing_action="augment",
                specialists=("tester",),
                specialist_counts=(("coder", 3),),
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(
            continuation.workflow_specialists,
            ("architect", "coder", "tester"),
        )
        self.assertEqual(continuation.workflow_specialist_counts, (("coder", 3),))

    async def test_execute_active_workflow_can_trim_staffing_from_plan(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
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
                workflow_specialists=("architect", "coder", "tester"),
                workflow_specialist_counts=(("coder", 3),),
            ),
        )

        [event async for event in executor.execute_active_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_playbook",
                workflow_id="standard-build",
                staffing_action="trim",
                specialists=("tester",),
                specialist_counts=(("coder", 2),),
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(continuation.workflow_specialists, ("architect", "coder"))
        self.assertEqual(continuation.workflow_specialist_counts, (("coder", 2),))

    async def test_execute_active_workflow_can_override_playbook_request(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workflow(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workflow_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_execute_workflow,
                    execute_workflow_continuation=_execute_workflow_continuation,
                ),
            )
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
                workflow_request="Old assignment",
            ),
        )

        [event async for event in executor.execute_active_workflow(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_playbook",
                workflow_id="standard-build",
                playbook_request="Polish the selected candidate into a final draft.",
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(
            continuation.workflow_request,
            "Polish the selected candidate into a final draft.",
        )

    async def test_execute_active_workflow_errors_without_progress(self) -> None:
        executor = ProxyWorkflowRequestExecutor(
            cast(
                ProxyWorkflowDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workflow=_unexpected_workflow,
                    execute_workflow_continuation=_unexpected_workflow,
                ),
            )
        )
        state = ProxyTurnState()
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Keep going"),),
        )

        events = [
            event
            async for event in executor.execute_active_workflow(
                request=request,
                state=state,
            )
        ]

        self.assertEqual(state.finish_reason, "error")
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertIn("No active playbook", first_event.delta)


class _FakeWorkflowDispatcher:
    def __init__(
        self,
        *,
        execute_workflow: Callable[..., AsyncIterator[ProxyContentDeltaEvent]],
        execute_workflow_continuation: Callable[
            ..., AsyncIterator[ProxyContentDeltaEvent]
        ],
    ) -> None:
        self._execute_workflow = execute_workflow
        self._execute_workflow_continuation = execute_workflow_continuation

    async def execute_workflow(self, **kwargs):
        async for event in self._execute_workflow(**kwargs):
            yield event

    async def execute_workflow_continuation(self, **kwargs):
        async for event in self._execute_workflow_continuation(**kwargs):
            yield event


async def _unexpected_workflow(**_kwargs):
    raise AssertionError("not expected")
