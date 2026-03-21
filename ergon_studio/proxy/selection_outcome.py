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


def selection_outcome_brief(
    outcome: ProxySelectionOutcome | None,
    *,
    fallback: str,
) -> str:
    if outcome is None:
        return fallback
    if outcome.summary:
        return outcome.summary
    if outcome.selected_candidate_text:
        return outcome.selected_candidate_text
    if outcome.next_refinement:
        return outcome.next_refinement
    return fallback


def selection_outcome_worklog_line(
    outcome: ProxySelectionOutcome | None,
) -> str | None:
    if outcome is None:
        return None
    parts: list[str] = [f"Orchestrator comparison result ({outcome.mode})"]
    if outcome.selected_candidate_text:
        parts.append(f"selected {outcome.selected_candidate_text}")
    if outcome.summary:
        parts.append(outcome.summary)
    if outcome.next_refinement:
        parts.append(f"Next: {outcome.next_refinement}")
    if len(parts) == 1:
        return None
    head, *tail = parts
    if not tail:
        return head
    return f"{head}: " + " | ".join(tail)
