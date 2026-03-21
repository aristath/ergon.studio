from __future__ import annotations

from typing import Any

from ergon_studio.proxy.agent_runner import ProxyAgentRunner
from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import ProxyToolCallEvent, ProxyTurnRequest
from ergon_studio.proxy.turn_state import ProxyTurnState


class ProxyToolCallEmitter:
    def __init__(self, agent_runner: ProxyAgentRunner) -> None:
        self._agent_runner = agent_runner

    def emit_tool_calls(
        self,
        *,
        response: Any,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        state: ProxyTurnState,
    ) -> list[ProxyToolCallEvent]:
        encoded_calls, events = self._agent_runner.emit_tool_calls(
            response=response,
            request=request,
            continuation=continuation,
        )
        if not events:
            return []
        state.tool_calls = encoded_calls
        state.finish_reason = "tool_calls"
        for call in encoded_calls:
            state.record_output_item("tool_call", call_id=call.id)
        return events
