from __future__ import annotations

from collections.abc import Mapping


def workroom_selection_hints_for_metadata(
    metadata: Mapping[str, object],
) -> tuple[str, ...]:
    configured = metadata.get("selection_hints")
    if configured is None:
        return ()
    if not isinstance(configured, list):
        return ()
    hints: list[str] = []
    for item in configured:
        if not isinstance(item, str):
            continue
        hint = item.strip()
        if not hint or hint in hints:
            continue
        hints.append(hint)
    return tuple(hints)


def workroom_acceptance_mode_for_metadata(metadata: Mapping[str, object]) -> str:
    value = metadata.get("acceptance_mode", "delivery")
    return str(value)


def workroom_delivery_candidate_for_metadata(metadata: Mapping[str, object]) -> bool:
    configured = metadata.get("delivery_candidate")
    if isinstance(configured, bool):
        return configured
    return workroom_acceptance_mode_for_metadata(metadata) == "delivery"
