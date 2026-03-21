from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProxySelectionOutcome:
    mode: str
    selected_candidate_index: int | None = None
    selected_candidate_text: str | None = None
    summary: str | None = None
    next_refinement: str | None = None


def selection_outcome_lines(
    outcome: ProxySelectionOutcome | None,
) -> tuple[str, ...]:
    if outcome is None:
        return ()
    lines = ["Latest structured comparison outcome:"]
    if outcome.selected_candidate_text:
        lines.extend(
            [
                "Selected candidate:",
                outcome.selected_candidate_text,
            ]
        )
    if outcome.summary:
        lines.extend(
            [
                "Decision summary:",
                outcome.summary,
            ]
        )
    if outcome.next_refinement:
        lines.extend(
            [
                "Suggested refinement:",
                outcome.next_refinement,
            ]
        )
    return tuple(lines)
