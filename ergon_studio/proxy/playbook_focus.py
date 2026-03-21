from __future__ import annotations

PLAYBOOK_FOCUS_VALUES = (
    "research",
    "plan",
    "implement",
    "compare",
    "critique",
    "verify",
    "polish",
    "deliver",
)

_PLAYBOOK_FOCUS_ALIASES = {
    "researching": "research",
    "investigate": "research",
    "investigation": "research",
    "planning": "plan",
    "design": "plan",
    "implementation": "implement",
    "build": "implement",
    "coding": "implement",
    "comparison": "compare",
    "select": "compare",
    "selection": "compare",
    "review": "critique",
    "challenge": "critique",
    "verification": "verify",
    "test": "verify",
    "testing": "verify",
    "polishing": "polish",
    "refine": "polish",
    "refinement": "polish",
    "delivery": "deliver",
    "finalize": "deliver",
    "ship": "deliver",
}

_PLAYBOOK_FOCUS_INSTRUCTIONS = {
    "research": "Gather missing facts, options, constraints, or external context.",
    "plan": "Shape the design, approach, breakdown, or implementation strategy.",
    "implement": "Build or change the actual artifact instead of only discussing it.",
    "compare": "Judge alternatives, select a winner, or synthesize the best parts.",
    "critique": (
        "Challenge weaknesses, risks, blind spots, or questionable assumptions."
    ),
    "verify": "Check behavior, run validation, or gather concrete evidence.",
    "polish": "Refine the chosen result so it is cleaner, clearer, or more complete.",
    "deliver": "Prepare the result for handoff back to the product manager.",
}


def normalize_playbook_focus(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = _PLAYBOOK_FOCUS_ALIASES.get(normalized, normalized)
    if normalized not in PLAYBOOK_FOCUS_VALUES:
        return None
    return normalized


def playbook_focus_instruction(focus: str) -> str:
    return _PLAYBOOK_FOCUS_INSTRUCTIONS.get(focus, focus)
