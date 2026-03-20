from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    thread_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None


_CURRENT_TOOL_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "ergon_studio_tool_execution_context",
    default=None,
)


def current_tool_execution_context() -> ToolExecutionContext | None:
    return _CURRENT_TOOL_CONTEXT.get()


@contextmanager
def use_tool_execution_context(context: ToolExecutionContext) -> Iterator[None]:
    previous = _CURRENT_TOOL_CONTEXT.get()
    _CURRENT_TOOL_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_CONTEXT.set(previous)
