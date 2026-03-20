from __future__ import annotations

from collections.abc import Mapping


def selection_hints_for_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
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


def acceptance_mode_for_metadata(metadata: Mapping[str, object]) -> str:
    value = metadata.get("acceptance_mode", "delivery")
    return str(value)


def is_non_delivery_acceptance_mode(acceptance_mode: str) -> bool:
    return acceptance_mode != "delivery"


def delivery_candidate_for_metadata(metadata: Mapping[str, object]) -> bool:
    configured = metadata.get("delivery_candidate")
    if isinstance(configured, bool):
        return configured
    return not is_non_delivery_acceptance_mode(acceptance_mode_for_metadata(metadata))


def is_decision_ready_acceptance_mode(acceptance_mode: str) -> bool:
    return acceptance_mode == "decision_ready"


def is_planning_acceptance_mode(acceptance_mode: str) -> bool:
    return acceptance_mode in {"research_brief", "design_brief", "revised_plan"}


def acceptance_rule_for_mode(acceptance_mode: str) -> str:
    if acceptance_mode == "decision_ready":
        return "Decide whether the work produced a concrete decision-ready recommendation that addresses the goal."
    if acceptance_mode == "research_brief":
        return "Decide whether the work produced a concrete research brief with enough evidence for the orchestrator to choose the next step."
    if acceptance_mode == "design_brief":
        return "Decide whether the work produced a concrete design brief that is implementation-ready and aligned with the goal."
    if acceptance_mode == "revised_plan":
        return "Decide whether the work produced an explicit revised plan that realigns the project and is actionable."
    return (
        "Decide whether the work satisfies the goal and represents a minimal working delivery. "
        "For runnable deliverables, require concrete command evidence that at least one direct invocation worked."
    )


def acceptance_criteria_for_mode(acceptance_mode: str) -> str:
    if acceptance_mode == "decision_ready":
        return "Produce a clear decision-ready recommendation that addresses the goal and passes orchestrator review."
    if acceptance_mode == "research_brief":
        return "Produce a concrete research brief with enough evidence for the orchestrator to choose the next step."
    if acceptance_mode == "design_brief":
        return "Produce a concrete design brief that is implementation-ready and passes orchestrator review."
    if acceptance_mode == "revised_plan":
        return "Produce an explicit revised plan that realigns the work and passes orchestrator review."
    return "Deliver a minimal working result that satisfies the goal and passes orchestrator review."


def step_groups_for_metadata(
    *,
    workflow_id: str,
    metadata: Mapping[str, object],
    metadata_key: str,
) -> tuple[tuple[str, ...], ...]:
    configured = metadata.get(metadata_key)
    if configured is None:
        return ()
    if not isinstance(configured, list):
        raise ValueError(f"workflow '{workflow_id}' metadata '{metadata_key}' must be a list")
    groups: list[tuple[str, ...]] = []
    for group in configured:
        if isinstance(group, str):
            if not group:
                raise ValueError(f"workflow '{workflow_id}' metadata '{metadata_key}' contains an empty step")
            groups.append((group,))
            continue
        if not isinstance(group, list) or not group:
            raise ValueError(f"workflow '{workflow_id}' metadata '{metadata_key}' must contain non-empty lists")
        validated: list[str] = []
        for item in group:
            if not isinstance(item, str) or not item:
                raise ValueError(f"workflow '{workflow_id}' metadata '{metadata_key}' contains an invalid step")
            validated.append(item)
        groups.append(tuple(validated))
    return tuple(groups)
