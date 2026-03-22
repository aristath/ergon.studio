from __future__ import annotations

import json
import zlib
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass

from ergon_studio.proxy.channels import ChannelMessage, ChannelSnapshot
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall

_TOKEN_PREFIX = "ergon:"
_TOKEN_VERSION = 2
_WORKLOG_TAIL = 12
_CHANNEL_TRANSCRIPT_TAIL = 12


@dataclass(frozen=True)
class ContinuationState:
    actor: str
    active_channel_id: str | None = None
    channels: tuple[ChannelSnapshot, ...] = ()
    worklog: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingContinuation:
    state: ContinuationState
    tool_states: tuple[ContinuationState, ...]
    assistant_message: ProxyInputMessage | None
    tool_results: tuple[ProxyInputMessage, ...]


def encode_continuation_tool_call(
    tool_call: ProxyToolCall, *, state: ContinuationState
) -> ProxyToolCall:
    payload = {
        "v": _TOKEN_VERSION,
        "a": state.actor,
        "tn": tool_call.name,
        "ta": tool_call.arguments_json,
    }
    if state.active_channel_id is not None:
        payload["c"] = state.active_channel_id
    if state.channels:
        payload["cs"] = [
            {
                "i": channel.channel_id,
                "n": channel.name,
                "p": list(channel.participants),
                "t": [
                    {
                        "a": message.author,
                        "c": message.content,
                    }
                    for message in channel.transcript[-_CHANNEL_TRANSCRIPT_TAIL:]
                ],
            }
            for channel in state.channels
        ]
    if state.worklog:
        payload["h"] = list(state.worklog[-_WORKLOG_TAIL:])
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
    actor = payload.get("a")
    active_channel_id = payload.get("c")
    raw_channels = payload.get("cs", [])
    worklog = payload.get("h", [])
    if not isinstance(actor, str):
        return None
    if active_channel_id is not None and not isinstance(active_channel_id, str):
        return None
    if not isinstance(raw_channels, list):
        return None
    if not isinstance(worklog, list) or not all(
        isinstance(item, str) for item in worklog
    ):
        return None
    channels: list[ChannelSnapshot] = []
    for item in raw_channels:
        if not isinstance(item, dict):
            return None
        channel_id = item.get("i")
        name = item.get("n")
        participants = item.get("p", [])
        transcript = item.get("t", [])
        if not isinstance(channel_id, str) or not channel_id:
            return None
        if not isinstance(name, str) or not name:
            return None
        if not isinstance(participants, list) or not all(
            isinstance(participant, str) for participant in participants
        ):
            return None
        if not isinstance(transcript, list):
            return None
        messages: list[ChannelMessage] = []
        for transcript_item in transcript:
            if not isinstance(transcript_item, dict):
                return None
            author = transcript_item.get("a")
            content = transcript_item.get("c")
            if not isinstance(author, str) or not author:
                return None
            if not isinstance(content, str):
                return None
            messages.append(ChannelMessage(author=author, content=content))
        channels.append(
            ChannelSnapshot(
                channel_id=channel_id,
                name=name,
                participants=tuple(participants),
                transcript=tuple(messages),
            )
        )
    return ContinuationState(
        actor=actor,
        active_channel_id=active_channel_id,
        channels=tuple(channels),
        worklog=tuple(worklog),
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
    tool_states: list[ContinuationState] = []
    for message in tool_results:
        decoded = decode_continuation_from_tool_call_id(message.tool_call_id or "")
        if decoded is None:
            return None
        if state is None:
            state = decoded
        elif (
            decoded.active_channel_id != state.active_channel_id
            or decoded.channels != state.channels
            or decoded.worklog != state.worklog
        ):
            return None
        tool_states.append(decoded)
    if state is None:
        return None

    return PendingContinuation(
        state=state,
        tool_states=tuple(tool_states),
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


def pending_actors(pending: PendingContinuation) -> tuple[str, ...]:
    seen: set[str] = set()
    actors: list[str] = []
    for state in pending.tool_states:
        if state.actor in seen:
            continue
        seen.add(state.actor)
        actors.append(state.actor)
    return tuple(actors)


def pending_for_actor(
    pending: PendingContinuation,
    actor: str,
) -> PendingContinuation | None:
    matched_pairs = [
        (state, tool_result)
        for state, tool_result in zip(
            pending.tool_states,
            pending.tool_results,
            strict=True,
        )
        if state.actor == actor
    ]
    if not matched_pairs:
        return None
    actor_states = tuple(state for state, _tool_result in matched_pairs)
    actor_results = tuple(tool_result for _state, tool_result in matched_pairs)
    return PendingContinuation(
        state=actor_states[0],
        tool_states=actor_states,
        assistant_message=pending.assistant_message,
        tool_results=actor_results,
    )
