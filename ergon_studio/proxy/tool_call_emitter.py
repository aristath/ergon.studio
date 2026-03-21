from __future__ import annotations

from typing import Any

from ergon_studio.proxy.agent_runner import ProxyAgentRunner
from ergon_studio.proxy.continuation import (
    ContinuationState,
    encode_continuation_tool_call,
)
from ergon_studio.proxy.models import (
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.turn_state import ProxyTurnState


class ProxyToolCallEmitter:
    def __init__(self, agent_runner: ProxyAgentRunner) -> None:
        self._agent_runner = agent_runner

    def emit_tool_calls(
        self,
        *,
        response: Any | None = None,
        tool_calls: tuple[ProxyToolCall, ...] | None = None,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        state: ProxyTurnState,
    ) -> list[ProxyToolCallEvent]:
        if tool_calls is None:
            if response is None:
                raise ValueError(
                    "response is required when tool_calls are not provided"
                )
            encoded_calls, events = self._agent_runner.emit_tool_calls(
                response=response,
                request=request,
                continuation=continuation,
            )
        else:
            validated_tool_calls = self._agent_runner.validate_host_tool_calls(
                tuple(tool_calls),
                request=request,
            )
            encoded_calls = tuple(
                encode_continuation_tool_call(tool_call, state=continuation)
                for tool_call in validated_tool_calls
            )
            events = [
                ProxyToolCallEvent(call=call, index=index)
                for index, call in enumerate(encoded_calls)
            ]
        if not events:
            return []
        state.tool_calls = encoded_calls
        state.finish_reason = "tool_calls"
        for call in encoded_calls:
            state.record_output_item("tool_call", call_id=call.id)
        return events
