from __future__ import annotations

from dataclasses import dataclass, field

from ergon_studio.proxy.models import ProxyOutputItemRef, ProxyToolCall


@dataclass
class ProxyTurnState:
    content: str = ""
    reasoning: str = ""
    finish_reason: str = "stop"
    tool_calls: tuple[ProxyToolCall, ...] = ()
    output_items: tuple[ProxyOutputItemRef, ...] = field(default_factory=tuple)
    _output_item_set: set[ProxyOutputItemRef] = field(default_factory=set)

    def append_content(self, delta: str) -> None:
        self.content += delta
        self.record_output_item("content")

    def append_reasoning(self, delta: str) -> None:
        self.reasoning += delta
        self.record_output_item("reasoning")

    def record_output_item(self, kind: str, *, call_id: str | None = None) -> None:
        item = ProxyOutputItemRef(kind, call_id)
        if item in self._output_item_set:
            return
        self._output_item_set.add(item)
        self.output_items = (*self.output_items, item)
