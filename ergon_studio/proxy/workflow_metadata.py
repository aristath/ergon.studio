from __future__ import annotations

from collections.abc import Mapping

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.workflow_compiler import workflow_step_groups_for_definition


def workflow_orchestration_for_definition(definition: DefinitionDocument) -> str:
    value = definition.metadata.get("orchestration", "sequential")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "sequential"


def workflow_participants_for_definition(definition: DefinitionDocument) -> tuple[str, ...]:
    participants: list[str] = []
    for group in workflow_step_groups_for_definition(definition):
        for agent_id in group:
            if agent_id not in participants:
                participants.append(agent_id)
    return tuple(participants)


def workflow_max_rounds_for_definition(definition: DefinitionDocument, *, default: int = 1) -> int:
    value = definition.metadata.get("max_rounds", default)
    if isinstance(value, int) and value > 0:
        return value
    return default


def workflow_selection_sequence_for_definition(definition: DefinitionDocument) -> tuple[str, ...]:
    configured = definition.metadata.get("selection_sequence")
    if not isinstance(configured, list):
        return ()
    sequence: list[str] = []
    for item in configured:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            sequence.append(stripped)
    return tuple(sequence)


def workflow_start_agent_for_definition(definition: DefinitionDocument) -> str | None:
    value = definition.metadata.get("start_agent")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def workflow_finalizers_for_definition(definition: DefinitionDocument) -> tuple[str, ...]:
    configured = definition.metadata.get("finalizers")
    if not isinstance(configured, list):
        return ()
    finalizers: list[str] = []
    for item in configured:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped and stripped not in finalizers:
            finalizers.append(stripped)
    return tuple(finalizers)


def workflow_handoffs_for_definition(definition: DefinitionDocument) -> dict[str, tuple[str, ...]]:
    configured = definition.metadata.get("handoffs")
    if not isinstance(configured, Mapping):
        return {}
    handoffs: dict[str, tuple[str, ...]] = {}
    for agent_id, allowed in configured.items():
        if not isinstance(agent_id, str) or not isinstance(allowed, list):
            continue
        normalized: list[str] = []
        for item in allowed:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped and stripped not in normalized:
                normalized.append(stripped)
        if normalized:
            handoffs[agent_id.strip()] = tuple(normalized)
    return handoffs
