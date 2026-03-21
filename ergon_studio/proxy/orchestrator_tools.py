from __future__ import annotations

import json
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall
from ergon_studio.registry import RuntimeRegistry

INTERNAL_TOOL_NAMES = frozenset({"message_workroom"})


@dataclass(frozen=True)
class MessageWorkroomAction:
    workroom_id: str | None
    participants: tuple[str, ...]
    message: str


InternalAction = MessageWorkroomAction


def build_orchestrator_internal_tools(
    registry: RuntimeRegistry,
) -> tuple[ProxyFunctionTool, ...]:
    specialist_ids = tuple(
        agent_id
        for agent_id in sorted(registry.agent_definitions)
        if agent_id != "orchestrator"
    )
    workroom_ids = tuple(sorted(registry.workroom_definitions))
    return (
        ProxyFunctionTool(
            name="message_workroom",
            description=(
                "Message a workroom. Provide a preset workroom_id or participants "
                "to open a room. If a room is already active, omitting both means "
                "continue it. Repeating a participant means multiple staffed "
                "instances of that role."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "workroom_id": {
                        "type": "string",
                        "enum": list(workroom_ids),
                    },
                    "participants": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(specialist_ids),
                        },
                    },
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
        ),
    )


def parse_internal_action(
    tool_call: ProxyToolCall,
    *,
    registry: RuntimeRegistry,
) -> InternalAction:
    try:
        payload = json.loads(tool_call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid arguments for {tool_call.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{tool_call.name} arguments must be an object")

    if tool_call.name != "message_workroom":
        raise ValueError(f"unsupported internal tool: {tool_call.name}")

    return MessageWorkroomAction(
        workroom_id=_optional_workroom_id(payload.get("workroom_id"), registry),
        participants=_normalize_staffing_list(
            payload.get("participants"),
            registry=registry,
        ),
        message=_required_text(payload.get("message"), field="message"),
    )


def is_internal_tool_name(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def _optional_workroom_id(
    value: object,
    registry: RuntimeRegistry,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("workroom_id must be a string when provided")
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in registry.workroom_definitions:
        raise ValueError(f"unknown workroom preset: {stripped}")
    return stripped


def _normalize_staffing_list(
    value: object,
    *,
    registry: RuntimeRegistry,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    participants: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if (
            not candidate
            or candidate == "orchestrator"
            or candidate not in registry.agent_definitions
        ):
            continue
        participants.append(candidate)
    return tuple(participants)
