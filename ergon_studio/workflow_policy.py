from __future__ import annotations

from collections.abc import Mapping


def acceptance_mode_for_metadata(metadata: Mapping[str, object]) -> str:
    value = metadata.get("acceptance_mode", "delivery")
    return str(value)


def is_non_delivery_acceptance_mode(acceptance_mode: str) -> bool:
    return acceptance_mode != "delivery"


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
