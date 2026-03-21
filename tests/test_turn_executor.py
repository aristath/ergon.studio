from __future__ import annotations

import unittest

from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_executor import ProxyTurnExecutor
from ergon_studio.proxy.turn_state import (
    ProxyDecisionLoopState,
    ProxyMoveResult,
    ProxyTurnState,
)


class TurnExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_delegation_streams_specialist_reasoning_and_returns_result(
        self,
    ) -> None:
        async def _stream_text_agent(**_kwargs):
            yield "Patch"
            yield " applied"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
            emit_tool_calls=_emit_tool_calls,
        )
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )
        state = ProxyTurnState()
        captured: list[ProxyMoveResult] = []

        events = [
            event
            async for event in executor.execute_delegation(
                request=request,
                agent_id="coder",
                message="Implement it",
                state=state,
                result_sink=captured.append,
            )
        ]

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("messaging specialist coder", reasoning.lower())
        self.assertIn("coder: Patch", reasoning)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].worklog_lines, ("coder: Patch applied",))
        self.assertEqual(captured[0].current_brief, "Patch applied")

    async def test_execute_delegation_passes_context_to_specialist(self) -> None:
        captured_prompt: dict[str, object] = {}

        async def _stream_text_agent(**kwargs):
            captured_prompt["prompt"] = kwargs["prompt"]
            yield "Patch applied"

        def _emit_tool_calls(**_kwargs):
            return []

        executor = ProxyTurnExecutor(
            stream_text_agent=_stream_text_agent,
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
        )

        [
            event
            async for event in executor.execute_delegation(
                request=request,
                agent_id="coder",
                message="Implement it",
                state=state,
                result_sink=lambda _result: None,
                loop_state=loop_state,
            )
        ]

        prompt = captured_prompt["prompt"]
        self.assertIsInstance(prompt, str)
        if not isinstance(prompt, str):
            raise AssertionError("expected prompt string")
        self.assertIn("Current progress:", prompt)
        self.assertIn("Plan is approved", prompt)
