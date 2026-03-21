from __future__ import annotations

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.workroom_layout import (
    discussion_turns_for_definition,
    referenced_agents_for_definition,
)


def workroom_shape_for_definition(definition: DefinitionDocument) -> str:
    value = definition.metadata.get("shape", "staged")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "staged"


def workroom_participants_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    participants: list[str] = []
    for agent_id in referenced_agents_for_definition(definition):
        if agent_id not in participants:
            participants.append(agent_id)
    return tuple(participants)


def workroom_turn_sequence_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    if workroom_shape_for_definition(definition) != "discussion":
        return ()
    return discussion_turns_for_definition(definition)
