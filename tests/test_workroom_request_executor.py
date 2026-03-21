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
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState, ProxyTurnState
from ergon_studio.proxy.workroom_dispatcher import ProxyWorkroomDispatcher
from ergon_studio.proxy.workroom_request_executor import (
    ProxyWorkroomRequestExecutor,
)


class WorkroomRequestExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_workroom_forwards_goal_and_staffing(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("workroom")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkroomDispatcher(
                    execute_workroom=_execute_workroom,
                    execute_workroom_continuation=_unexpected_workroom,
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
                workroom_id="standard-build",
                participants=("coder", "coder", "coder", "reviewer"),
                workroom_request="Build it safely",
                state=ProxyTurnState(),
            )
        ]

        self.assertEqual(calls[0]["workroom_id"], "standard-build")
        self.assertEqual(
            calls[0]["participants"],
            ("coder", "coder", "coder", "reviewer"),
        )
        self.assertEqual(calls[0]["workroom_request"], "Build it safely")
        self.assertEqual(calls[0]["goal"], "Build it")
        self.assertEqual(events[0].delta, "workroom")

    async def test_execute_active_workroom_uses_loop_state_progress(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkroomDispatcher(
                    execute_workroom=_unexpected_workroom,
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
                workroom_participants=("architect", "coder"),
            ),
        )

        events = [
            event
            async for event in executor.execute_active_workroom(
                request=request,
                message="Polish it",
                participants=("coder", "coder"),
                state=ProxyTurnState(),
                loop_state=loop_state,
            )
        ]

        continuation = calls[0]["continuation"]
        self.assertEqual(continuation.workroom_request, "Polish it")
        self.assertEqual(continuation.workroom_participants, ("coder", "coder"))
        self.assertEqual(events[0].delta, "continued")

    async def test_execute_workroom_continuation_forwards_pending_state(self) -> None:
        calls: list[dict[str, object]] = []

        async def _execute_workroom_continuation(**kwargs):
            calls.append(kwargs)
            yield ProxyContentDeltaEvent("continued")

        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkroomDispatcher(
                    execute_workroom=_unexpected_workroom,
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

        self.assertIs(calls[0]["pending"], pending)
        self.assertEqual(events[0].delta, "continued")

    async def test_execute_active_workroom_errors_without_progress(self) -> None:
        executor = ProxyWorkroomRequestExecutor(
            cast(
                ProxyWorkroomDispatcher,
                _FakeWorkroomDispatcher(
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
                message="Keep going",
                participants=(),
                state=state,
            )
        ]

        self.assertEqual(state.finish_reason, "error")
        self.assertIn("No active workroom", events[0].delta)


class _FakeWorkroomDispatcher:
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
