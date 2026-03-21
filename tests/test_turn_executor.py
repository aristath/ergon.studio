from __future__ import annotations

import unittest

from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.planner import ProxyTurnPlan
from ergon_studio.proxy.turn_executor import ProxyTurnExecutor
from ergon_studio.proxy.turn_state import ProxyDecisionLoopState, ProxyTurnState


class TurnExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_direct_streams_content(self) -> None:
        async def _stream_text_agent(**_kwargs):
            yield "Hello"
            yield " world"

        async def _run_text_agent(**_kwargs):
            return None

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
            run_text_agent=_run_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute_direct(request=request, state=state)
        ]

        self.assertIsInstance(events[0], ProxyReasoningDeltaEvent)
        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
            "Hello world",
        )
        self.assertEqual(state.content, "Hello world")

    async def test_execute_delegation_summarizes_specialist_result(self) -> None:
        async def _stream_text_agent(**_kwargs):
            yield "Patch"
            yield " applied"

        async def _run_text_agent(**_kwargs):
            return "Final summary"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
            run_text_agent=_run_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute_delegation(
                request=request,
                plan=ProxyTurnPlan(
                    mode="delegate",
                    agent_id="coder",
                    request="Implement it",
                ),
                state=state,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("delegating", reasoning.lower())
        self.assertIn("coder: Patch", reasoning)
        self.assertEqual(state.content, "Final summary")

    async def test_execute_delegation_passes_decision_context_to_specialist(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        async def _stream_text_agent(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            yield "Patch applied"

        async def _run_text_agent(**_kwargs):
            return "Final summary"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
            run_text_agent=_run_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )
        state = ProxyTurnState()
        loop_state = ProxyDecisionLoopState(
            goal="Ship calculator",
            current_brief="Plan is approved",
            current_move_rationale="The coder can implement the agreed plan quickly.",
        )

        [
            event
            async for event in executor.execute_delegation(
                request=request,
                plan=ProxyTurnPlan(
                    mode="delegate",
                    agent_id="coder",
                    request="Implement it",
                    rationale="The coder can implement the agreed plan quickly.",
                ),
                state=state,
                loop_state=loop_state,
            )
        ]

        prompt = captured["prompt"]
        self.assertIsInstance(prompt, str)
        if not isinstance(prompt, str):
            raise AssertionError("expected prompt string")
        self.assertIn("Why the lead developer assigned you this slice", prompt)

    async def test_execute_finish_streams_final_delivery(self) -> None:
        async def _stream_text_agent(**_kwargs):
            yield "Ready"
            yield " to ship"

        async def _run_text_agent(**_kwargs):
            return None

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
            run_text_agent=_run_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Ship it"),),
        )
        state = ProxyTurnState()

        events = [
            event
            async for event in executor.execute_finish(request=request, state=state)
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("delivering", reasoning.lower())
        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
            "Ready to ship",
        )
        self.assertEqual(state.content, "Ready to ship")
