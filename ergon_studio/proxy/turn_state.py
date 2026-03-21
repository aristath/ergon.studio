from __future__ import annotations

from dataclasses import dataclass, field

from ergon_studio.proxy.continuation import ContinuationState
from ergon_studio.proxy.models import ProxyOutputItemRef, ProxyToolCall
from ergon_studio.proxy.selection_outcome import ProxySelectionOutcome


@dataclass(frozen=True)
class ProxyMoveResult:
    worklog_lines: tuple[str, ...]
    current_brief: str
    workflow_progress: ContinuationState | None = None
    selection_outcome: ProxySelectionOutcome | None = None
    selection_outcome_changed: bool = False


@dataclass
class ProxyDecisionLoopState:
    goal: str
    current_brief: str
    worklog: tuple[str, ...] = field(default_factory=tuple)
    workflow_progress: ContinuationState | None = None
    latest_selection_outcome: ProxySelectionOutcome | None = None
    current_playbook_request: str | None = None
    current_move_rationale: str | None = None
    current_move_success_criteria: str | None = None
    current_comparison_mode: str | None = None
    current_comparison_criteria: str | None = None

    def absorb_result(
        self,
        *,
        result: ProxyMoveResult,
    ) -> None:
        if result.worklog_lines:
            self.worklog = (*self.worklog, *result.worklog_lines)
        if result.current_brief:
            self.current_brief = result.current_brief
        self.workflow_progress = result.workflow_progress
        if result.selection_outcome_changed:
            self.latest_selection_outcome = result.selection_outcome
        self.current_playbook_request = None
        self.current_move_rationale = None
        self.current_move_success_criteria = None
        self.current_comparison_mode = None
        self.current_comparison_criteria = None


@dataclass
class ProxyTurnState:
    content: str = ""
    reasoning: str = ""
    mode: str = "act"
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
