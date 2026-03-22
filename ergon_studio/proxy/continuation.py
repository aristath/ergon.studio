from __future__ import annotations

from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall
from ergon_studio.proxy.pending_store import PendingCallRecord, PendingStore

_TOKEN_PREFIX = "ergon:"
_TOKEN_VERSION = 3


@dataclass(frozen=True)
class PendingToolContext:
    pending_id: str
    session_id: str
    actor: str
    active_channel_id: str | None
    tool_calls: tuple[ProxyToolCall, ...]
    tool_results: tuple[ProxyInputMessage, ...] = ()


@dataclass(frozen=True)
class PendingContinuation:
    session_id: str
    items: tuple[PendingToolContext, ...]


def encode_continuation_tool_call(
    tool_call: ProxyToolCall,
    *,
    pending_id: str,
) -> ProxyToolCall:
    return ProxyToolCall(
        id=f"{_TOKEN_PREFIX}{_TOKEN_VERSION}:{pending_id}:{tool_call.id}",
        name=tool_call.name,
        arguments_json=tool_call.arguments_json,
    )


def decode_pending_id_from_tool_call_id(tool_call_id: str) -> str | None:
    parts = _token_parts(tool_call_id)
    if parts is None:
        return None
    _version, pending_id, _original = parts
    return pending_id or None


def original_tool_call_id(tool_call_id: str) -> str | None:
    parts = _token_parts(tool_call_id)
    if parts is None:
        return None
    _version, _pending_id, original = parts
    return original or None


def latest_pending_continuation(
    messages: tuple[ProxyInputMessage, ...],
    *,
    pending_store: PendingStore,
) -> PendingContinuation | None:
    if not messages or messages[-1].role != "tool":
        return None

    tool_results_reversed: list[ProxyInputMessage] = []
    index = len(messages) - 1
    while index >= 0 and messages[index].role == "tool":
        tool_results_reversed.append(messages[index])
        index -= 1
    tool_results = tuple(reversed(tool_results_reversed))

    grouped_results: dict[str, list[ProxyInputMessage]] = {}
    pending_order: list[str] = []
    records: dict[str, PendingCallRecord] = {}
    for tool_result in tool_results:
        pending_id = decode_pending_id_from_tool_call_id(tool_result.tool_call_id or "")
        if pending_id is None:
            return None
        record = pending_store.get(pending_id)
        if record is None:
            return None
        if pending_id not in grouped_results:
            grouped_results[pending_id] = []
            pending_order.append(pending_id)
            records[pending_id] = record
        grouped_results[pending_id].append(tool_result)

    if not pending_order:
        return None

    session_ids = {records[pending_id].session_id for pending_id in pending_order}
    if len(session_ids) != 1:
        return None

    items = tuple(
        PendingToolContext(
            pending_id=pending_id,
            session_id=records[pending_id].session_id,
            actor=records[pending_id].actor,
            active_channel_id=records[pending_id].active_channel_id,
            tool_calls=records[pending_id].tool_calls,
            tool_results=tuple(grouped_results[pending_id]),
        )
        for pending_id in pending_order
    )
    for pending_id in pending_order:
        pending_store.discard(pending_id)
    return PendingContinuation(
        session_id=items[0].session_id,
        items=items,
    )


def continuation_tool_calls(
    pending: PendingToolContext,
) -> tuple[ProxyToolCall, ...]:
    result_ids = {
        original_tool_call_id(message.tool_call_id or "")
        for message in pending.tool_results
    }
    return tuple(
        tool_call
        for tool_call in pending.tool_calls
        if tool_call.id in result_ids
    )


def continuation_result_map(pending: PendingToolContext) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for message in pending.tool_results:
        original_id = original_tool_call_id(message.tool_call_id or "")
        if original_id is None:
            continue
        mapped[original_id] = message.content
    return mapped


def pending_actors(pending: PendingContinuation) -> tuple[str, ...]:
    return tuple(item.actor for item in pending.items)


def pending_for_actor(
    pending: PendingContinuation,
    actor: str,
) -> PendingToolContext | None:
    for item in pending.items:
        if item.actor == actor:
            return item
    return None


def _token_parts(tool_call_id: str) -> tuple[str, str, str] | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    parts = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 2)
    if len(parts) != 3:
        return None
    version, pending_id, original = parts
    if version != str(_TOKEN_VERSION):
        return None
    if not pending_id:
        return None
    return version, pending_id, original
