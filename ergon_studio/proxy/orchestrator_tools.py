from __future__ import annotations

import json
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall
from ergon_studio.registry import RuntimeRegistry

INTERNAL_TOOL_NAMES = frozenset(
    {"message_specialist", "open_workroom", "continue_workroom"}
)


@dataclass(frozen=True)
class MessageSpecialistAction:
    agent_id: str
    message: str


@dataclass(frozen=True)
class OpenWorkroomAction:
    workroom_id: str | None
    participants: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class ContinueWorkroomAction:
    message: str


InternalAction = MessageSpecialistAction | OpenWorkroomAction | ContinueWorkroomAction


def build_orchestrator_internal_tools(
    registry: RuntimeRegistry,
    *,
    has_active_workroom: bool,
) -> tuple[ProxyFunctionTool, ...]:
    specialist_ids = tuple(
        agent_id
        for agent_id in sorted(registry.agent_definitions)
        if agent_id != "orchestrator"
    )
    workroom_ids = tuple(sorted(registry.workroom_definitions))
    tools = [
        ProxyFunctionTool(
            name="message_specialist",
            description=(
                "Open or continue a direct channel with one specialist and send "
                "them a focused natural-language brief."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "enum": list(specialist_ids),
                    },
                    "message": {"type": "string"},
                },
                "required": ["agent_id", "message"],
            },
        ),
        ProxyFunctionTool(
            name="open_workroom",
            description=(
                "Open a collaborative workroom. Use a preset by id, or provide "
                "participants for an ad hoc room. Repeating a participant means "
                "multiple staffed instances of that role."
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
    ]
    if has_active_workroom:
        tools.append(
            ProxyFunctionTool(
                name="continue_workroom",
                description=(
                    "Continue the workroom already in progress with a new "
                    "natural-language assignment from the lead developer."
                ),
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            )
        )
    return tuple(tools)


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

    if tool_call.name == "message_specialist":
        agent_id = _required_specialist(payload.get("agent_id"), registry=registry)
        message = _required_text(payload.get("message"), field="message")
        return MessageSpecialistAction(agent_id=agent_id, message=message)

    if tool_call.name == "open_workroom":
        workroom_id = _optional_workroom_id(payload.get("workroom_id"), registry)
        participants = _normalize_staffing_list(
            payload.get("participants"),
            registry=registry,
        )
        if workroom_id is None and not participants:
            raise ValueError(
                "open_workroom requires either a preset workroom_id or participants"
            )
        message = _required_text(payload.get("message"), field="message")
        return OpenWorkroomAction(
            workroom_id=workroom_id,
            participants=participants,
            message=message,
        )

    if tool_call.name == "continue_workroom":
        message = _required_text(payload.get("message"), field="message")
        return ContinueWorkroomAction(message=message)

    raise ValueError(f"unsupported internal tool: {tool_call.name}")


def is_internal_tool_name(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def _required_specialist(value: object, *, registry: RuntimeRegistry) -> str:
    agent_id = _required_text(value, field="agent_id")
    if agent_id == "orchestrator" or agent_id not in registry.agent_definitions:
        raise ValueError(f"unknown specialist: {agent_id}")
    return agent_id


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
