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
    title: str
    created_at: int
    updated_at: int
    archived_at: int | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")
        validate_unix_time(self.updated_at, "updated_at")
        if self.archived_at is not None:
            validate_unix_time(self.archived_at, "archived_at")


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
class WorkflowRunRecord:
    id: str
    session_id: str
    workflow_id: str
    state: str
    created_at: int
    updated_at: int
    root_task_id: str | None = None
    current_step_index: int = 0
    last_thread_id: str | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")
        validate_unix_time(self.updated_at, "updated_at")
        if type(self.current_step_index) is not int:
            raise TypeError("current_step_index must be an int")


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
    thread_id: str | None = None
    task_id: str | None = None
    payload_path: Path | None = None

    def __post_init__(self) -> None:
        if self.payload_path is not None and not isinstance(self.payload_path, Path):
            raise TypeError("payload_path must be a Path or None")
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class MemoryFactRecord:
    id: str
    scope: str
    kind: str
    content: str
    created_at: int
    source: str | None = None
    confidence: float | None = None
    tags: tuple[str, ...] = ()
    last_used_at: int | None = None

    def __post_init__(self) -> None:
        validate_unix_time(self.created_at, "created_at")
        if self.confidence is not None and type(self.confidence) not in {int, float}:
            raise TypeError("confidence must be a number or None")
        if self.last_used_at is not None:
            validate_unix_time(self.last_used_at, "last_used_at")
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be a tuple")


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


@dataclass(frozen=True)
class CommandRunRecord:
    id: str
    session_id: str
    command: str
    cwd: str
    exit_code: int
    status: str
    output_path: Path
    created_at: int
    thread_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.exit_code) is not int:
            raise TypeError("exit_code must be an int")
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be a Path")
        validate_unix_time(self.created_at, "created_at")


@dataclass(frozen=True)
class ToolCallRecord:
    id: str
    session_id: str
    tool_name: str
    status: str
    request_path: Path
    created_at: int
    response_path: Path | None = None
    thread_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_path, Path):
            raise TypeError("request_path must be a Path")
        if self.response_path is not None and not isinstance(self.response_path, Path):
            raise TypeError("response_path must be a Path or None")
        validate_unix_time(self.created_at, "created_at")
