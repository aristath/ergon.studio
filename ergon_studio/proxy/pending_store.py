from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ergon_studio.proxy.models import ProxyToolCall


@dataclass(frozen=True)
class PendingCallRecord:
    pending_id: str
    session_id: str
    actor: str
    active_channel_id: str | None
    tool_calls: tuple[ProxyToolCall, ...]


class PendingStore:
    def __init__(self) -> None:
        self._records: dict[str, PendingCallRecord] = {}

    def create(
        self,
        *,
        session_id: str,
        actor: str,
        active_channel_id: str | None = None,
        tool_calls: tuple[ProxyToolCall, ...],
    ) -> PendingCallRecord:
        record = PendingCallRecord(
            pending_id=f"pending_{uuid4().hex}",
            session_id=session_id,
            actor=actor,
            active_channel_id=active_channel_id,
            tool_calls=tool_calls,
        )
        self._records[record.pending_id] = record
        return record

    def get(self, pending_id: str) -> PendingCallRecord | None:
        return self._records.get(pending_id)

    def discard(self, pending_id: str) -> None:
        self._records.pop(pending_id, None)
