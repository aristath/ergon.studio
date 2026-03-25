from __future__ import annotations

import json
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyFunctionTool, ProxyToolCall
from ergon_studio.proxy.orchestrator_tools import MalformedToolCallError

WORKSPACE_TOOL_NAMES = frozenset({"read_file", "write_file", "list_files"})


@dataclass(frozen=True)
class ReadFileAction:
    path: str


@dataclass(frozen=True)
class WriteFileAction:
    path: str
    content: str


@dataclass(frozen=True)
class ListFilesAction:
    directory: str


def is_workspace_tool_name(name: str) -> bool:
    return name in WORKSPACE_TOOL_NAMES


def parse_read_file_action(tool_call: ProxyToolCall) -> ReadFileAction:
    payload = _parse_tool_payload(tool_call)
    return ReadFileAction(path=_required_text(payload.get("path"), field="path"))


def parse_write_file_action(tool_call: ProxyToolCall) -> WriteFileAction:
    payload = _parse_tool_payload(tool_call)
    path = _required_text(payload.get("path"), field="path")
    raw_content = payload.get("content")
    if not isinstance(raw_content, str):
        raise ValueError("content must be a string")
    return WriteFileAction(path=path, content=raw_content)


def parse_list_files_action(tool_call: ProxyToolCall) -> ListFilesAction:
    payload = _parse_tool_payload(tool_call)
    return ListFilesAction(
        directory=_required_text(payload.get("directory"), field="directory")
    )


def build_workspace_tools() -> tuple[ProxyFunctionTool, ...]:
    return (
        ProxyFunctionTool(
            name="read_file",
            description="Read a file from the filesystem by absolute path.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        ProxyFunctionTool(
            name="write_file",
            description=(
                "Write content to a file at the given absolute path. "
                "The file is written to a session workspace overlay; "
                "the real filesystem is never modified."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        ProxyFunctionTool(
            name="list_files",
            description="List files in a directory by absolute path.",
            parameters={
                "type": "object",
                "properties": {"directory": {"type": "string"}},
                "required": ["directory"],
            },
        ),
    )


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
