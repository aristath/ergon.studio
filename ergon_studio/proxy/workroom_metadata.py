from __future__ import annotations

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.workroom_compiler import workroom_step_groups_for_definition


def workroom_shape_for_definition(definition: DefinitionDocument) -> str:
    value = definition.metadata.get("shape", "staged")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "staged"


def workroom_participants_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    participants: list[str] = []
    for group in workroom_step_groups_for_definition(definition):
        for agent_id in group:
            if agent_id not in participants:
                participants.append(agent_id)
    return tuple(participants)


def workroom_turn_sequence_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    return tuple(
        agent_id
        for group in workroom_step_groups_for_definition(definition)
        for agent_id in group
    )


def workroom_max_rounds_for_definition(
    definition: DefinitionDocument, *, default: int = 1
) -> int:
    value = definition.metadata.get("max_rounds", default)
    if isinstance(value, int) and value > 0:
        return value
    return default
