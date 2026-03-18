from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def validate_unix_time(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a Unix time int")
    return value


@dataclass(frozen=True)
class SessionRecord:
    id: str
    project_uuid: str
    created_at: int

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class ThreadRecord:
    id: str
    session_id: str
    kind: str
    created_at: int
    updated_at: int
    assigned_agent_id: str | None = None
    summary: str | None = None
    parent_task_id: str | None = None
    parent_thread_id: str | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")
        validate_unix_time(self.updated_at, "updated_at")


@dataclass(frozen=True)
class TaskRecord:
    id: str
    session_id: str
    title: str
    state: str
    created_at: int
    updated_at: int
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")
        validate_unix_time(self.updated_at, "updated_at")


@dataclass(frozen=True)
class MessageRecord:
    id: str
    thread_id: str
    sender: str
    kind: str
    body_path: Path
    created_at: int
    task_id: str | None = None
    artifact_id: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.body_path, Path):
            raise TypeError("body_path must be a Path")
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class EventRecord:
    id: str
    session_id: str
    kind: str
    summary: str
    created_at: int
    thread_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class ApprovalRecord:
    id: str
    session_id: str
    requester: str
    action: str
    risk_class: str
    reason: str
    status: str
    created_at: int

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class MemoryFactRecord:
    id: str
    scope: str
    kind: str
    content: str
    created_at: int

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    session_id: str
    kind: str
    title: str
    file_path: Path
    created_at: int
    thread_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.file_path, Path):
            raise TypeError("file_path must be a Path")
        validate_unix_time(self.created_at, "created_at")
