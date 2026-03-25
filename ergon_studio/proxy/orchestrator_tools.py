from __future__ import annotations

import json
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall
from ergon_studio.registry import RuntimeRegistry

INTERNAL_TOOL_NAMES = frozenset(
    {"open_channel", "message_channel", "close_channel", "run_parallel"}
)


class MalformedToolCallError(ValueError):
    """Raised when a tool call's arguments_json cannot be decoded as JSON."""


@dataclass(frozen=True)
class OpenChannelAction:
    preset: str | None
    participants: tuple[str, ...]
    message: str
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class MessageChannelAction:
    channel: str
    message: str
    recipients: tuple[str, ...]


@dataclass(frozen=True)
class CloseChannelAction:
    channel: str


@dataclass(frozen=True)
class RunParallelAction:
    agent: str
    count: int
    task: str


def build_orchestrator_internal_tools(
    registry: RuntimeRegistry,
) -> tuple[ProxyFunctionTool, ...]:
    specialist_ids = tuple(
        agent_id
        for agent_id in sorted(registry.agent_definitions)
        if agent_id != "orchestrator"
    )
    preset_ids = tuple(sorted(registry.channel_presets))
    return (
        ProxyFunctionTool(
            name="open_channel",
            description=(
                "Open a new channel with teammates. Provide a preset or participants "
                "to choose who is on the call. Repeating a participant means "
                "multiple staffed instances of that role."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": list(preset_ids),
                    },
                    "participants": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(specialist_ids),
                        },
                    },
                    "message": {"type": "string"},
                    "recipients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Explicit teammates to address. Use role ids like "
                            "'coder' or staffed labels like 'coder[2]' when a "
                            "channel has repeated instances."
                        ),
                    },
                },
                "required": ["message", "recipients"],
            },
        ),
        ProxyFunctionTool(
            name="message_channel",
            description=(
                "Send another message into an already-open channel by channel id. "
                "Recipients decides who you are actively addressing."
            ),
            parameters=_message_channel_parameters(
                include_channel=True,
            ),
        ),
        ProxyFunctionTool(
            name="close_channel",
            description="Close an open channel when the conversation is done.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                },
                "required": ["channel"],
            },
        ),
        ProxyFunctionTool(
            name="run_parallel",
            description=(
                "Run N isolated sub-sessions of a specialist agent in parallel on a "
                "task. Each sub-session gets its own workspace overlay and cannot use "
                "host tools. All results are automatically returned to you in the next "
                "turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": list(specialist_ids),
                        "description": "Specialist agent id to run.",
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Number of parallel sub-sessions (default 1).",
                    },
                    "task": {
                        "type": "string",
                        "description": "Prompt given to each sub-session.",
                    },
                },
                "required": ["agent", "task"],
            },
        ),
    )

def parse_open_channel_action(
    tool_call: ProxyToolCall,
    *,
    registry: RuntimeRegistry,
) -> OpenChannelAction:
    if tool_call.name != "open_channel":
        raise ValueError(f"unsupported orchestrator tool: {tool_call.name}")
    payload = _parse_tool_payload(tool_call)
    preset = _optional_preset(payload.get("preset"), registry)
    participants = _normalize_staffing_list(
        payload.get("participants"),
        registry=registry,
    )
    if preset is not None and participants:
        raise ValueError(
            "open_channel requires either preset or participants, not both"
        )
    return OpenChannelAction(
        preset=preset,
        participants=participants,
        message=_required_text(payload.get("message"), field="message"),
        recipients=_required_recipient_list(
            payload.get("recipients"),
            field="recipients",
        ),
    )


def parse_message_channel_action(
    tool_call: ProxyToolCall,
    *,
    require_channel: bool = True,
) -> MessageChannelAction:
    if tool_call.name != "message_channel":
        raise ValueError(f"unsupported orchestrator tool: {tool_call.name}")
    payload = _parse_tool_payload(tool_call)
    return MessageChannelAction(
        channel=(
            _required_text(payload.get("channel"), field="channel")
            if require_channel
            else ""
        ),
        message=_required_text(payload.get("message"), field="message"),
        recipients=_required_recipient_list(
            payload.get("recipients"),
            field="recipients",
        ),
    )


def parse_close_channel_action(tool_call: ProxyToolCall) -> CloseChannelAction:
    if tool_call.name != "close_channel":
        raise ValueError(f"unsupported orchestrator tool: {tool_call.name}")
    payload = _parse_tool_payload(tool_call)
    return CloseChannelAction(
        channel=_required_text(payload.get("channel"), field="channel"),
    )


def parse_run_parallel_action(
    tool_call: ProxyToolCall,
    *,
    registry: RuntimeRegistry,
) -> RunParallelAction:
    if tool_call.name != "run_parallel":
        raise ValueError(f"unsupported orchestrator tool: {tool_call.name}")
    payload = _parse_tool_payload(tool_call)
    agent = _required_text(payload.get("agent"), field="agent")
    if agent not in registry.agent_definitions or agent == "orchestrator":
        raise ValueError(f"unknown agent for run_parallel: {agent}")
    raw_count = payload.get("count", 1)
    if not isinstance(raw_count, int) or raw_count < 1 or raw_count > 8:
        raise ValueError("count must be an integer between 1 and 8")
    task = _required_text(payload.get("task"), field="task")
    return RunParallelAction(agent=agent, count=raw_count, task=task)


def is_internal_tool_name(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def _parse_tool_payload(tool_call: ProxyToolCall) -> dict[str, object]:
    try:
        payload = json.loads(tool_call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise MalformedToolCallError(
            f"invalid arguments for {tool_call.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{tool_call.name} arguments must be an object")
    return payload


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty string")
    return stripped


def _optional_preset(
    value: object,
    registry: RuntimeRegistry,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("preset must be a string when provided")
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in registry.channel_presets:
        raise ValueError(f"unknown channel preset: {stripped}")
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


def _message_channel_parameters(
    *,
    include_channel: bool,
) -> dict[str, object]:
    properties: dict[str, object] = {
        "message": {"type": "string"},
        "recipients": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Explicit teammates to address. Use role ids like 'coder' or "
                "staffed labels like 'coder[2]' when repeated instances are on "
                "the channel."
            ),
        },
    }
    required = ["message", "recipients"]
    if include_channel:
        properties["channel"] = {"type": "string"}
        required.insert(0, "channel")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _required_recipient_list(
    value: object,
    *,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must name at least one teammate")
    recipients: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} entries must be strings")
        candidate = item.strip()
        if not candidate or candidate == "orchestrator":
            raise ValueError(
                f"{field} entries must be non-empty teammate identifiers"
            )
        recipients.append(candidate)
    return tuple(recipients)


PARTICIPANT_INTERNAL_TOOLS = (
    ProxyFunctionTool(
        name="message_channel",
        description=(
            "Send another message into the current channel and explicitly "
            "target the teammates you want to answer next."
        ),
        parameters=_message_channel_parameters(
            include_channel=False,
        ),
    ),
)
