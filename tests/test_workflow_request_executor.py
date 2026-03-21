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
from ergon_studio.proxy.workroom_dispatcher import ProxyWorkroomDispatcher
from ergon_studio.proxy.workroom_request_executor import (
    ProxyWorkroomRequestExecutor,
)


class WorkflowRequestExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_workroom_uses_plan_goal_or_latest_user_text(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workroom_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        events = [
            event
            async for event in executor.execute_workroom(
                request=request,
                plan=ProxyTurnPlan(mode="workroom", workroom_id="standard-build"),
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["workroom_id"], "standard-build")
        self.assertEqual(calls[0]["specialists"], ())
        self.assertEqual(calls[0]["goal"], "Build it")
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "workflow")

    async def test_execute_workroom_forwards_staffed_specialists(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workroom_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        [event async for event in executor.execute_workroom(
            request=request,
            plan=ProxyTurnPlan(
                mode="workroom",
                workroom_id="standard-build",
                specialists=("coder", "reviewer"),
                specialist_counts=(("coder", 3),),
            ),
            state=ProxyTurnState(),
        )]

        self.assertEqual(calls[0]["specialists"], ("coder", "reviewer"))
        self.assertEqual(calls[0]["specialist_counts"], (("coder", 3),))

    async def test_execute_workroom_forwards_workroom_request(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workflow")

        async def _execute_workroom_continuation(**kwargs):
            raise AssertionError("not expected")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        [event async for event in executor.execute_workroom(
            request=request,
            plan=ProxyTurnPlan(
                mode="workroom",
                workroom_id="best-of-n",
                workroom_request="Compare three implementations and pick one.",
            ),
            state=ProxyTurnState(),
        )]

        self.assertEqual(
            calls[0]["workroom_request"],
            "Compare three implementations and pick one.",
        )

    async def test_execute_workroom_continuation_forwards_pending_state(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
                ),
            )
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        pending = PendingContinuation(
            state=ContinuationState(
                mode="workroom",
                agent_id="coder",
                workroom_id="standard-build",
            ),
            assistant_message=None,
            tool_results=(),
        )

        events = [
            event
            async for event in executor.execute_workroom_continuation(
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

    async def test_execute_active_workroom_uses_loop_state_progress(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
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
            workroom_progress=ContinuationState(
                mode="workroom",
                agent_id="architect",
                workroom_id="standard-build",
            ),
        )

        events = [
            event
            async for event in executor.execute_active_workroom(
                request=request,
                state=ProxyTurnState(),
                loop_state=loop_state,
            )
        ]

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["continuation"], loop_state.workroom_progress)
        self.assertIsNone(calls[0]["pending"])
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertEqual(first_event.delta, "continued")

    async def test_execute_active_workroom_can_override_staffing_from_plan(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
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
            workroom_progress=ContinuationState(
                mode="workroom",
                agent_id="architect",
                workroom_id="standard-build",
                workroom_specialists=("architect", "coder"),
            ),
        )

        [event async for event in executor.execute_active_workroom(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_workroom",
                workroom_id="standard-build",
                specialists=("coder",),
                specialist_counts=(("coder", 3),),
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(continuation.workroom_specialists, ("coder",))
        self.assertEqual(continuation.workroom_specialist_counts, (("coder", 3),))

    async def test_execute_active_workroom_can_override_workroom_request(
        self,
    ) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            raise AssertionError("not expected")

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_execute_workroom_continuation,
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
            workroom_progress=ContinuationState(
                mode="workroom",
                agent_id="architect",
                workroom_id="standard-build",
                workroom_request="Old assignment",
            ),
        )

        [event async for event in executor.execute_active_workroom(
            request=request,
            plan=ProxyTurnPlan(
                mode="continue_workroom",
                workroom_id="standard-build",
                workroom_request="Polish the selected candidate into a final draft.",
            ),
            state=ProxyTurnState(),
            loop_state=loop_state,
        )]

        continuation = calls[0]["continuation"]
        self.assertEqual(
            continuation.workroom_request,
            "Polish the selected candidate into a final draft.",
        )

    async def test_execute_active_workroom_errors_without_progress(self) -> None:
        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkflowDispatcher(
                    execute_workroom=_unexpected_workroom,
                    execute_workroom_continuation=_unexpected_workroom,
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
            async for event in executor.execute_active_workroom(
                request=request,
                state=state,
            )
        ]

        self.assertEqual(state.finish_reason, "error")
        first_event = events[0]
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        if not isinstance(first_event, ProxyContentDeltaEvent):
            raise AssertionError("expected ProxyContentDeltaEvent")
        self.assertIn("No active workroom", first_event.delta)


class _FakeWorkflowDispatcher:
    def __init__(
        self,
        *,
        execute_workroom: Callable[..., AsyncIterator[ProxyContentDeltaEvent]],
        execute_workroom_continuation: Callable[
            ..., AsyncIterator[ProxyContentDeltaEvent]
        ],
    ) -> None:
        self._execute_workroom = execute_workroom
        self._execute_workroom_continuation = execute_workroom_continuation

    async def execute_workroom(self, **kwargs):
        async for event in self._execute_workroom(**kwargs):
            yield event

    async def execute_workroom_continuation(self, **kwargs):
        async for event in self._execute_workroom_continuation(**kwargs):
            yield event


async def _unexpected_workroom(**_kwargs):
    raise AssertionError("not expected")
