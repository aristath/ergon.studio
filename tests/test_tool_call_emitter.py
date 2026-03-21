from __future__ import annotations

import unittest
from typing import cast

from ergon_studio.proxy.agent_runner import ProxyAgentRunner
from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.tool_call_emitter import ProxyToolCallEmitter
from ergon_studio.proxy.turn_state import ProxyTurnState


class ToolCallEmitterTests(unittest.TestCase):
    def test_emit_tool_calls_updates_state_and_returns_events(self) -> None:
        call = ProxyToolCall(
            id="call_1",
            name="read_file",
            arguments_json='{"path":"a"}',
        )
        event = ProxyToolCallEvent(call)
        runner = _FakeAgentRunner((call,), [event])
        emitter = ProxyToolCallEmitter(cast(ProxyAgentRunner, runner))
        state = ProxyTurnState()
        request = ProxyTurnRequest(
            model="qwen",
            messages=(ProxyInputMessage(role="user", content="Inspect"),),
        )

        events = emitter.emit_tool_calls(
            response={"tool_calls": []},
            request=request,
            continuation=ContinuationState(mode="act", agent_id="orchestrator"),
            state=state,
        )

        self.assertEqual(events, [event])
        self.assertEqual(state.finish_reason, "tool_calls")
        self.assertEqual(state.tool_calls, (call,))
        self.assertEqual(state.output_items[0].kind, "tool_call")
        self.assertEqual(state.output_items[0].call_id, "call_1")


class _FakeAgentRunner:
    def __init__(
        self,
        encoded_calls: tuple[ProxyToolCall, ...],
        events: list[ProxyToolCallEvent],
    ) -> None:
        self._encoded_calls = encoded_calls
        self._events = events

    def emit_tool_calls(self, **_kwargs):
        return self._encoded_calls, self._events
