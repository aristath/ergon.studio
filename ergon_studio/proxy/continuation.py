from __future__ import annotations

import json
import zlib
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall

_TOKEN_PREFIX = "ergon:"
_TOKEN_VERSION = 1


@dataclass(frozen=True)
class ContinuationState:
    mode: str
    agent_id: str
    participant_label: str | None = None
    workroom_id: str | None = None
    workroom_participants: tuple[str, ...] = ()
    workroom_message: str | None = None
    member_index: int | None = None
    goal: str | None = None
    worklog: tuple[str, ...] = ()
    workroom_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingContinuation:
    state: ContinuationState
    assistant_message: ProxyInputMessage | None
    tool_results: tuple[ProxyInputMessage, ...]


def encode_continuation_tool_call(
    tool_call: ProxyToolCall, *, state: ContinuationState
) -> ProxyToolCall:
    payload = {
        "v": _TOKEN_VERSION,
        "m": state.mode,
        "a": state.agent_id,
        "tn": tool_call.name,
        "ta": tool_call.arguments_json,
    }
    if state.participant_label is not None:
        payload["al"] = state.participant_label
    if state.workroom_id is not None:
        payload["w"] = state.workroom_id
    if state.workroom_participants:
        payload["p"] = list(state.workroom_participants)
    if state.workroom_message is not None:
        payload["pr"] = state.workroom_message
    if state.member_index is not None:
        payload["i"] = state.member_index
    if state.goal is not None:
        payload["g"] = state.goal
    if state.worklog:
        payload["h"] = list(state.worklog)
    if state.workroom_outputs:
        payload["o"] = list(state.workroom_outputs)
    encoded = (
        urlsafe_b64encode(
            zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        )
        .decode("ascii")
        .rstrip("=")
    )
    return ProxyToolCall(
        id=f"{_TOKEN_PREFIX}{encoded}:{tool_call.id}",
        name=tool_call.name,
        arguments_json=tool_call.arguments_json,
    )


def decode_continuation_from_tool_call_id(
    tool_call_id: str,
) -> ContinuationState | None:
    payload = _decode_payload(tool_call_id)
    if payload is None:
        return None
    if payload.get("v") != _TOKEN_VERSION:
        return None
    mode = payload.get("m")
    agent_id = payload.get("a")
    participant_label = payload.get("al")
    workroom_id = payload.get("w")
    workroom_participants = payload.get("p", [])
    workroom_message = payload.get("pr")
    member_index = payload.get("i")
    goal = payload.get("g")
    worklog = payload.get("h", [])
    workroom_outputs = payload.get("o", [])
    if not isinstance(mode, str) or not isinstance(agent_id, str):
        return None
    if participant_label is not None and not isinstance(participant_label, str):
        return None
    if workroom_id is not None and not isinstance(workroom_id, str):
        return None
    if not isinstance(workroom_participants, list) or not all(
        isinstance(item, str) for item in workroom_participants
    ):
        return None
    if workroom_message is not None and not isinstance(workroom_message, str):
        return None
    if member_index is not None and not isinstance(member_index, int):
        return None
    if goal is not None and not isinstance(goal, str):
        return None
    if not isinstance(worklog, list) or not all(
        isinstance(item, str) for item in worklog
    ):
        return None
    if not isinstance(workroom_outputs, list) or not all(
        isinstance(item, str) for item in workroom_outputs
    ):
        return None
    return ContinuationState(
        mode=mode,
        agent_id=agent_id,
        participant_label=participant_label,
        workroom_id=workroom_id,
        workroom_participants=tuple(workroom_participants),
        workroom_message=workroom_message,
        member_index=member_index,
        goal=goal,
        worklog=tuple(worklog),
        workroom_outputs=tuple(workroom_outputs),
    )


def decode_original_tool_call(tool_call_id: str) -> ProxyToolCall | None:
    payload = _decode_payload(tool_call_id)
    original = original_tool_call_id(tool_call_id)
    if payload is None or original is None:
        return None
    name = payload.get("tn")
    arguments = payload.get("ta", "")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        return None
    return ProxyToolCall(
        id=tool_call_id,
        name=name,
        arguments_json=arguments,
    )


def _decode_payload(tool_call_id: str) -> dict[str, object] | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    try:
        encoded_payload, _original = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 1)
    except ValueError:
        return None
    padding = "=" * (-len(encoded_payload) % 4)
    try:
        payload = json.loads(
            zlib.decompress(
                urlsafe_b64decode((encoded_payload + padding).encode("ascii"))
            ).decode("utf-8")
        )
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def original_tool_call_id(tool_call_id: str) -> str | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    try:
        _encoded_payload, original = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 1)
    except ValueError:
        return None
    return original or None


def latest_continuation(
    messages: tuple[ProxyInputMessage, ...],
) -> ContinuationState | None:
    pending = latest_pending_continuation(messages)
    if pending is None:
        return None
    return pending.state


def latest_pending_continuation(
    messages: tuple[ProxyInputMessage, ...],
) -> PendingContinuation | None:
    if not messages or messages[-1].role != "tool":
        return None

    tool_results_reversed: list[ProxyInputMessage] = []
    index = len(messages) - 1
    while index >= 0 and messages[index].role == "tool":
        tool_results_reversed.append(messages[index])
        index -= 1
    tool_results = tuple(reversed(tool_results_reversed))
    assistant_message: ProxyInputMessage | None = None
    if index >= 0:
        candidate = messages[index]
        if candidate.role == "assistant" and candidate.tool_calls:
            assistant_message = candidate
            assistant_call_ids = {tool_call.id for tool_call in candidate.tool_calls}
            if any(
                (message.tool_call_id or "") not in assistant_call_ids
                for message in tool_results
            ):
                return None

    state: ContinuationState | None = None
    for message in tool_results:
        decoded = decode_continuation_from_tool_call_id(message.tool_call_id or "")
        if decoded is None:
            return None
        if state is None:
            state = decoded
        elif decoded != state:
            return None
    if state is None:
        return None

    return PendingContinuation(
        state=state,
        assistant_message=assistant_message,
        tool_results=tool_results,
    )


def continuation_tool_calls(pending: PendingContinuation) -> tuple[ProxyToolCall, ...]:
    if pending.assistant_message is None:
        return ()
    result_ids = {message.tool_call_id for message in pending.tool_results}
    return tuple(
        tool_call
        for tool_call in pending.assistant_message.tool_calls
        if tool_call.id in result_ids
    )


def continuation_result_map(pending: PendingContinuation) -> dict[str, str]:
    return {
        message.tool_call_id or "": message.content
        for message in pending.tool_results
        if message.tool_call_id
    }
