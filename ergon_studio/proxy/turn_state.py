from __future__ import annotations

from dataclasses import dataclass, field

from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import ProxyOutputItemRef, ProxyToolCall


@dataclass(frozen=True)
class ProxyMoveResult:
    worklog_lines: tuple[str, ...]
    active_workroom: ContinuationState | None = None


@dataclass
class ProxyDecisionLoopState:
    worklog: tuple[str, ...] = field(default_factory=tuple)
    active_workroom: ContinuationState | None = None

    def absorb_result(
        self,
        *,
        result: ProxyMoveResult,
    ) -> None:
        if result.worklog_lines:
            self.worklog = (*self.worklog, *result.worklog_lines)
        self.active_workroom = result.active_workroom


@dataclass
class ProxyTurnState:
    content: str = ""
    reasoning: str = ""
    mode: str = "orchestrator"
    finish_reason: str = "stop"
    tool_calls: tuple[ProxyToolCall, ...] = ()
    output_items: tuple[ProxyOutputItemRef, ...] = field(default_factory=tuple)

    def append_content(self, delta: str) -> None:
        self.content += delta
        self.record_output_item("content")

    def set_content(self, content: str) -> None:
        self.content = content
        if content:
            self.record_output_item("content")

    def append_reasoning(self, delta: str) -> None:
        self.reasoning += delta
        self.record_output_item("reasoning")

    def record_output_item(self, kind: str, *, call_id: str | None = None) -> None:
        item = ProxyOutputItemRef(kind, call_id)
        if item in self.output_items:
            return
        self.output_items = (*self.output_items, item)
