from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from agent_framework import FunctionInvocationContext, function_middleware

from ergon_studio.event_store import EventStore
from ergon_studio.tool_call_store import ToolCallStore
from ergon_studio.tool_context import current_tool_execution_context


def build_tool_call_middleware(
    *,
    tool_call_store: ToolCallStore,
    event_store: EventStore,
    now: Callable[[], int],
):
    @function_middleware
    async def log_tool_call(context: FunctionInvocationContext, call_next) -> None:
        tool_context = current_tool_execution_context()
        if tool_context is None:
            await call_next()
            return

        created_at = now()
        try:
            await call_next()
        except Exception as exc:
            record = tool_call_store.record_tool_call(
                session_id=tool_context.session_id,
                tool_call_id=f"tool-call-{uuid4().hex[:8]}",
                tool_name=context.function.name,
                arguments=_normalize_arguments(context.arguments),
                result=None,
                status="failed",
                created_at=created_at,
                thread_id=tool_context.thread_id,
                task_id=tool_context.task_id,
                agent_id=tool_context.agent_id,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            event_store.append_event(
                session_id=tool_context.session_id,
                event_id=f"event-{uuid4().hex}",
                kind="tool_call",
                summary=f"{tool_context.agent_id} called {context.function.name} [failed]",
                created_at=created_at,
                thread_id=tool_context.thread_id,
                task_id=tool_context.task_id,
            )
            context.metadata["tool_call_id"] = record.id
            raise

        record = tool_call_store.record_tool_call(
            session_id=tool_context.session_id,
            tool_call_id=f"tool-call-{uuid4().hex[:8]}",
            tool_name=context.function.name,
            arguments=_normalize_arguments(context.arguments),
            result=context.result,
            status="completed",
            created_at=created_at,
            thread_id=tool_context.thread_id,
            task_id=tool_context.task_id,
            agent_id=tool_context.agent_id,
        )
        event_store.append_event(
            session_id=tool_context.session_id,
            event_id=f"event-{uuid4().hex}",
            kind="tool_call",
            summary=f"{tool_context.agent_id} called {context.function.name} [completed]",
            created_at=created_at,
            thread_id=tool_context.thread_id,
            task_id=tool_context.task_id,
        )
        context.metadata["tool_call_id"] = record.id

    return log_tool_call


def _normalize_arguments(arguments: object) -> object:
    if hasattr(arguments, "model_dump"):
        return arguments.model_dump()
    if hasattr(arguments, "dict"):
        return arguments.dict()
    return arguments
