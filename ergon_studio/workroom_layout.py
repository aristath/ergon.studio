from __future__ import annotations

from ergon_studio.definitions import DefinitionDocument


def workroom_kind_for_definition(definition: DefinitionDocument) -> str:
    has_stages = "stages" in definition.metadata
    has_turns = "turns" in definition.metadata
    if has_stages and has_turns:
        raise ValueError(
            "workroom template "
            f"'{definition.id}' cannot declare both `stages` and `turns`"
        )
    if has_turns:
        return "discussion"
    if has_stages:
        return "staged"
    raise ValueError(
        f"workroom template '{definition.id}' must declare either `stages` or `turns`"
    )


def staged_groups_for_definition(
    definition: DefinitionDocument,
) -> tuple[tuple[str, ...], ...]:
    configured_stages = definition.metadata.get("stages")
    if configured_stages is None:
        raise ValueError(
            f"workroom template '{definition.id}' must declare `stages`"
        )
    if not isinstance(configured_stages, list):
        raise ValueError(f"workroom template '{definition.id}' stages must be a list")
    return tuple(
        _validate_staged_group(definition.id, group) for group in configured_stages
    )


def discussion_turns_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    configured_turns = definition.metadata.get("turns")
    if configured_turns is None:
        raise ValueError(f"workroom template '{definition.id}' must declare `turns`")
    if not isinstance(configured_turns, list):
        raise ValueError(f"workroom template '{definition.id}' turns must be a list")
    return _validate_member_sequence(
        definition.id,
        configured_turns,
        field_name="turns",
    )


def referenced_agents_for_definition(
    definition: DefinitionDocument,
) -> tuple[str, ...]:
    if workroom_kind_for_definition(definition) == "discussion":
        return discussion_turns_for_definition(definition)
    return tuple(
        agent_id
        for group in staged_groups_for_definition(definition)
        for agent_id in group
    )


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
    if workroom_kind_for_definition(definition) != "discussion":
        return ()
    return discussion_turns_for_definition(definition)


def _validate_staged_group(
    workroom_id: str,
    group: object,
) -> tuple[str, ...]:
    if isinstance(group, str):
        stripped = group.strip()
        if not stripped:
            raise ValueError(
                "workroom template "
                f"'{workroom_id}' stage entries must be non-empty strings"
            )
        return (stripped,)
    if not isinstance(group, list) or not group:
        raise ValueError(
            f"workroom template '{workroom_id}' stages must contain non-empty groups"
        )
    return _validate_member_sequence(
        workroom_id,
        group,
        field_name="stage entries",
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
                "workroom template "
                f"'{workroom_id}' {field_name} must be non-empty strings"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                "workroom template "
                f"'{workroom_id}' {field_name} must be non-empty strings"
            )
        validated.append(stripped)
    return tuple(validated)
