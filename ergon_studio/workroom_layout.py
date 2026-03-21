from __future__ import annotations

from ergon_studio.definitions import DefinitionDocument


def workroom_participants_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    configured_participants = definition.metadata.get("participants")
    if configured_participants is None:
        raise ValueError(
            f"workroom preset '{definition.id}' must declare `participants`"
        )
    if not isinstance(configured_participants, list):
        raise ValueError(
            f"workroom preset '{definition.id}' participants must be a list"
        )
    return _validate_member_sequence(
        definition.id,
        configured_participants,
        field_name="participants",
    )


def _validate_member_sequence(
    workroom_id: str,
    values: list[object],
    *,
    field_name: str,
) -> tuple[str, ...]:
    validated: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(
                "workroom preset "
                f"'{workroom_id}' {field_name} must be non-empty strings"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                "workroom preset "
                f"'{workroom_id}' {field_name} must be non-empty strings"
            )
        validated.append(stripped)
    return tuple(validated)
