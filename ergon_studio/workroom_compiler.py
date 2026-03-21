from __future__ import annotations

from ergon_studio.definitions import DefinitionDocument


def workroom_step_groups_for_definition(
    definition: DefinitionDocument,
) -> tuple[tuple[str, ...], ...]:
    configured_step_groups = definition.metadata.get("step_groups")
    if configured_step_groups is not None:
        if not isinstance(configured_step_groups, list):
            raise ValueError(
                f"workroom template '{definition.id}' step_groups must be a list"
            )
        return tuple(
            validate_workroom_group(definition.id, group)
            for group in configured_step_groups
        )

    configured_steps = definition.metadata.get("steps")
    if configured_steps is None:
        raise ValueError(
            f"workroom template '{definition.id}' must declare `steps` or `step_groups`"
        )
    if not isinstance(configured_steps, list):
        raise ValueError(f"workroom template '{definition.id}' steps must be a list")
    return (
        tuple(
            (step,) for step in validate_workroom_group(definition.id, configured_steps)
        )
        if configured_steps
        else ()
    )


def validate_workroom_group(workroom_id: str, group: object) -> tuple[str, ...]:
    if isinstance(group, str):
        stripped = group.strip()
        if not stripped:
            raise ValueError(
                "workroom template "
                f"'{workroom_id}' step entries must be non-empty strings"
            )
        return (stripped,)
    if not isinstance(group, list) or not group:
        raise ValueError(
            f"workroom template '{workroom_id}' step groups must be non-empty lists"
        )
    validated: list[str] = []
    for item in group:
        if not isinstance(item, str):
            raise ValueError(
                "workroom template "
                f"'{workroom_id}' step entries must be non-empty strings"
            )
        stripped = item.strip()
        if not stripped:
            raise ValueError(
                "workroom template "
                f"'{workroom_id}' step entries must be non-empty strings"
            )
        validated.append(stripped)
    return tuple(validated)
