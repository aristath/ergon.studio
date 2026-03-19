from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_framework import FunctionInvocationContext, tool

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.event_store import EventStore
from ergon_studio.tool_call_logging import build_tool_call_middleware
from ergon_studio.tool_call_store import ToolCallStore
from ergon_studio.tool_context import ToolExecutionContext, use_tool_execution_context


class ToolCallLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_middleware_records_completed_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            tool_call_store = ToolCallStore(paths)
            event_store = EventStore(paths)
            middleware = build_tool_call_middleware(
                tool_call_store=tool_call_store,
                event_store=event_store,
                now=lambda: 10,
            )

            @tool(name="read_file", approval_mode="never_require")
            def read_file(path: str) -> str:
                return f"read:{path}"

            context = FunctionInvocationContext(read_file, {"path": "README.md"})

            async def call_next() -> None:
                context.result = "read:README.md"

            with use_tool_execution_context(
                ToolExecutionContext(
                    session_id="session-main",
                    thread_id="thread-main",
                    task_id="task-1",
                    agent_id="coder",
                )
            ):
                await middleware(context, call_next)

            records = tool_call_store.list_tool_calls("session-main")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tool_name, "read_file")
            self.assertEqual(records[0].status, "completed")
            self.assertIn("tool_call", [event.kind for event in event_store.list_events("session-main")])

    async def test_middleware_preserves_non_completed_tool_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            tool_call_store = ToolCallStore(paths)
            event_store = EventStore(paths)
            middleware = build_tool_call_middleware(
                tool_call_store=tool_call_store,
                event_store=event_store,
                now=lambda: 20,
            )

            @tool(name="run_command", approval_mode="never_require")
            def run_command(command: str) -> dict[str, object]:
                del command
                return {"status": "budget_exhausted", "stderr": "too many commands"}

            context = FunctionInvocationContext(run_command, {"command": "python3 calc.py 1 + 1"})

            async def call_next() -> None:
                context.result = run_command("python3 calc.py 1 + 1")

            with use_tool_execution_context(
                ToolExecutionContext(
                    session_id="session-main",
                    thread_id="thread-agent-coder-1",
                    task_id="task-1",
                    agent_id="coder",
                )
            ):
                await middleware(context, call_next)

            records = tool_call_store.list_tool_calls("session-main")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].tool_name, "run_command")
            self.assertEqual(records[0].status, "budget_exhausted")
            events = event_store.list_events("session-main")
            self.assertEqual(events[-1].summary, "coder called run_command [budget_exhausted]")
