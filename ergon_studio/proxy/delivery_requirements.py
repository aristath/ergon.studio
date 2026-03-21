from __future__ import annotations

DELIVERY_REQUIREMENT_VALUES = ("review", "verify", "critique")

_ALIASES = {
    "reviewed": "review",
    "verification": "verify",
    "verified": "verify",
    "test": "verify",
    "testing": "verify",
    "criticism": "critique",
    "critic": "critique",
}

_AGENT_EVIDENCE = {
    "reviewer": ("review",),
    "tester": ("verify",),
    "critic": ("critique",),
}


def normalize_delivery_requirements(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items: list[object] = [value]
    elif isinstance(value, list):
        items = value
    else:
        return None
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        candidate = item.strip().lower()
        if not candidate:
            continue
        candidate = _ALIASES.get(candidate, candidate)
        if candidate not in DELIVERY_REQUIREMENT_VALUES:
            continue
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def delivery_evidence_for_agent(agent_id: str) -> tuple[str, ...]:
    return _AGENT_EVIDENCE.get(agent_id, ())


def delivery_evidence_for_agents(agent_ids: tuple[str, ...]) -> tuple[str, ...]:
    evidence: list[str] = []
    for agent_id in agent_ids:
        for requirement in delivery_evidence_for_agent(agent_id):
            if requirement not in evidence:
                evidence.append(requirement)
    return tuple(evidence)


def merge_delivery_evidence(
    existing: tuple[str, ...],
    new: tuple[str, ...],
) -> tuple[str, ...]:
    merged = list(existing)
    for requirement in new:
        if requirement not in merged:
            merged.append(requirement)
    return tuple(merged)


def unmet_delivery_requirements(
    requirements: tuple[str, ...],
    evidence: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        requirement for requirement in requirements if requirement not in evidence
    )
