from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_framework import AgentSession

from ergon_studio.runtime import load_runtime


class TuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_renders_core_workspace_panels(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                self.assertEqual(app.title, "ergon.studio")
                self.assertIsNotNone(app.query_one("#main-chat"))
                self.assertIsNotNone(app.query_one("#tasks"))
                self.assertIsNotNone(app.query_one("#workflows"))
                self.assertIsNotNone(app.query_one("#workflow-runs"))
                self.assertIsNotNone(app.query_one("#threads"))
                self.assertIsNotNone(app.query_one("#activity"))
                self.assertIsNotNone(app.query_one("#commands"))
                self.assertIsNotNone(app.query_one("#artifacts"))
                self.assertIsNotNone(app.query_one("#approvals"))
                self.assertIsNotNone(app.query_one("#tool-calls"))
                self.assertIsNotNone(app.query_one("#memory"))
                self.assertIsNotNone(app.query_one("#team"))
                self.assertIsNotNone(app.query_one("#settings"))
                self.assertIn("thread-main", app.query_one("#threads", Panel).body)
                self.assertIn("> orchestrator", app.query_one("#team", Panel).body)
                self.assertIn("> standard-build", app.query_one("#workflows", Panel).body)
                self.assertIn("No workflow runs yet.", app.query_one("#workflow-runs", Panel).body)
                self.assertIn("No tasks yet.", app.query_one("#tasks", Panel).body)
                self.assertIn("No activity yet.", app.query_one("#activity", Panel).body)
                self.assertIn("No commands yet.", app.query_one("#commands", Panel).body)
                self.assertIn("No approvals pending.", app.query_one("#approvals", Panel).body)
                self.assertIn("No tool calls yet.", app.query_one("#tool-calls", Panel).body)
                self.assertIn("No memory facts yet.", app.query_one("#memory", Panel).body)
                self.assertIn("No artifacts yet.", app.query_one("#artifacts", Panel).body)
                self.assertIn("orchestrator", app.query_one("#settings", Panel).body)
                self.assertIn("standard-build", app.query_one("#settings", Panel).body)
                self.assertIn("Orchestrator: not configured", app.query_one("#settings", Panel).body)

    async def test_app_renders_persisted_main_thread_messages(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Hello from the persisted main thread.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                main_chat = app.query_one("#main-chat", Panel)
                threads = app.query_one("#threads", Panel)
                self.assertIn("Hello from the persisted main thread.", main_chat.body)
                self.assertIn("thread-main", threads.body)

    async def test_app_renders_persisted_tasks(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_task(
                task_id="task-1",
                title="Build task sidebar",
                state="in_progress",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                tasks = app.query_one("#tasks", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("> task-1", tasks.body)
                self.assertIn("State: in_progress", tasks.body)
                self.assertIn("Build task sidebar", tasks.body)
                self.assertIn("task_created", activity.body)

    async def test_app_can_switch_selected_activity_event(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_event(
                kind="event_one",
                summary="First event",
                created_at=1_710_755_200,
            )
            runtime.append_event(
                kind="event_two",
                summary="Second event",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                activity = app.query_one("#activity", Panel)
                self.assertIn("> event_two: Second event", activity.body)

                app.action_previous_activity_event()

                activity = app.query_one("#activity", Panel)
                self.assertIn("> event_one: First event", activity.body)
                self.assertIn("Summary: First event", activity.body)

    async def test_app_renders_additional_threads(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
                summary="Review thread",
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                threads = app.query_one("#threads", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("thread-review-1", threads.body)
                self.assertIn("thread_created", activity.body)

    async def test_app_renders_persisted_command_runs(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.run_workspace_command(
                "pwd",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                commands = app.query_one("#commands", Panel)
                self.assertIn("> command-run-", commands.body)
                self.assertIn("[completed/0] pwd", commands.body)
                self.assertIn("Cwd:", commands.body)
                self.assertIn("# Command Run", commands.body)

    async def test_app_renders_tool_calls(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.tool_call_store.record_tool_call(
                session_id=runtime.main_session_id,
                tool_call_id="tool-call-1",
                tool_name="read_file",
                arguments={"path": "README.md"},
                result={"content": "hello"},
                status="completed",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="orchestrator",
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                tool_calls = app.query_one("#tool-calls", Panel)
                self.assertIn("> tool-call-1", tool_calls.body)
                self.assertIn("Tool: read_file", tool_calls.body)
                self.assertIn('"path": "README.md"', tool_calls.body)

    async def test_app_can_switch_selected_command_run(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.run_workspace_command(
                "printf first",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )
            runtime.run_workspace_command(
                "printf second",
                created_at=1_710_755_201,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                commands = app.query_one("#commands", Panel)
                self.assertIn("printf second", commands.body)

                app.action_previous_command_run()

                commands = app.query_one("#commands", Panel)
                self.assertIn("printf first", commands.body)
                self.assertIn("Command: `printf first`", commands.body)

    async def test_app_can_open_selected_agent_thread(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)
            app.selected_agent_id = "architect"

            async with app.run_test():
                app.action_open_selected_agent_thread()

                tasks = app.query_one("#tasks", Panel)
                threads = app.query_one("#threads", Panel)
                selected_thread = app.query_one("#selected-thread", Panel)
                self.assertIn("Agent thread: architect", tasks.body)
                self.assertIn("agent_direct:architect", threads.body)
                self.assertIn("Kind: agent_direct", selected_thread.body)
                self.assertIn("Agent: architect", selected_thread.body)
                self.assertIn("Task: task-", selected_thread.body)
                self.assertIn("No messages yet.", selected_thread.body)

    async def test_app_renders_pending_approvals(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                approvals = app.query_one("#approvals", Panel)
                self.assertIn("> approval-1", approvals.body)
                self.assertIn("write_file", approvals.body)
                self.assertIn("Update README", approvals.body)

    async def test_app_can_approve_selected_approval(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                app.action_approve_selected_approval()

                approvals = app.query_one("#approvals", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("No approvals pending.", approvals.body)
                self.assertIn("approval_approved", activity.body)
                self.assertEqual(runtime.list_approvals()[0].status, "approved")

    async def test_app_approving_command_approval_executes_command(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="user",
                action="run_command",
                risk_class="high",
                reason="Run pwd",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                payload={
                    "command": "pwd",
                    "timeout": 60,
                    "cwd": str(project_root.resolve()),
                    "thread_id": runtime.main_thread_id,
                    "task_id": None,
                    "agent_id": "user",
                },
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                approvals = app.query_one("#approvals", Panel)
                self.assertIn("Command: pwd", approvals.body)

                app.action_approve_selected_approval()

                approvals = app.query_one("#approvals", Panel)
                commands = app.query_one("#commands", Panel)
                self.assertIn("No approvals pending.", approvals.body)
                self.assertIn("pwd", commands.body)

    async def test_app_approving_file_write_approval_writes_the_file(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="user",
                action="write_file",
                risk_class="moderate",
                reason="Write notes.txt",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                payload={
                    "path": "notes.txt",
                    "content": "hello\n",
                    "thread_id": runtime.main_thread_id,
                    "task_id": None,
                    "agent_id": "user",
                },
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                approvals = app.query_one("#approvals", Panel)
                self.assertIn("Path: notes.txt", approvals.body)

                app.action_approve_selected_approval()

                approvals = app.query_one("#approvals", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("No approvals pending.", approvals.body)
                self.assertEqual((project_root / "notes.txt").read_text(encoding="utf-8"), "hello\n")
                self.assertIn("file_written", activity.body)

    async def test_selected_workflow_run_scopes_approvals_panel(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()
                workflow_run = runtime.list_workflow_runs()[0]
                workflow_threads = runtime.list_threads_for_workflow_run(workflow_run.id)
                runtime.request_approval(
                    approval_id="approval-1",
                    requester="coder",
                    action="write_file",
                    risk_class="moderate",
                    reason="Update workflow file",
                    created_at=1_710_755_210,
                    task_id=workflow_threads[0].parent_task_id,
                    thread_id=workflow_threads[0].id,
                )
                runtime.request_approval(
                    approval_id="approval-2",
                    requester="orchestrator",
                    action="run_command",
                    risk_class="high",
                    reason="Install dependencies",
                    created_at=1_710_755_211,
                )
                app._refresh_panels()

                approvals = app.query_one("#approvals", Panel)
                self.assertIn("Run: workflow-run-", approvals.body)
                self.assertIn("approval-1", approvals.body)
                self.assertNotIn("approval-2", approvals.body)

    async def test_app_can_reject_selected_approval(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                app.action_reject_selected_approval()

                approvals = app.query_one("#approvals", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("No approvals pending.", approvals.body)
                self.assertIn("approval_rejected", activity.body)
                self.assertEqual(runtime.list_approvals()[0].status, "rejected")

    async def test_app_can_switch_selected_approval(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )
            runtime.request_approval(
                approval_id="approval-2",
                requester="orchestrator",
                action="run_command",
                risk_class="high",
                reason="Install project dependencies",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                approvals = app.query_one("#approvals", Panel)
                self.assertIn("> approval-1", approvals.body)

                app.action_next_approval()

                approvals = app.query_one("#approvals", Panel)
                self.assertIn("> approval-2", approvals.body)
                self.assertIn("Install project dependencies", approvals.body)

    async def test_app_renders_memory_facts(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.add_memory_fact(
                fact_id="fact-1",
                scope="project",
                kind="decision",
                content="Use Textual for the TUI.",
                created_at=1_710_755_200,
                source="plan",
                confidence=0.9,
                tags=("ui",),
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                memory = app.query_one("#memory", Panel)
                self.assertIn("> fact-1", memory.body)
                self.assertIn("Scope: project", memory.body)
                self.assertIn("Source: plan", memory.body)
                self.assertIn("Use Textual for the TUI.", memory.body)

    async def test_app_can_switch_selected_memory_fact(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.add_memory_fact(
                fact_id="fact-1",
                scope="project",
                kind="decision",
                content="Use Textual for the TUI.",
                created_at=1_710_755_200,
            )
            runtime.add_memory_fact(
                fact_id="fact-2",
                scope="user",
                kind="preference",
                content="Keep replies short.",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                memory = app.query_one("#memory", Panel)
                self.assertIn("> fact-1", memory.body)
                self.assertIn("Use Textual for the TUI.", memory.body)

                app.action_next_memory_fact()

                memory = app.query_one("#memory", Panel)
                self.assertIn("> fact-2", memory.body)
                self.assertIn("Scope: user", memory.body)
                self.assertIn("Keep replies short.", memory.body)

    async def test_app_renders_selected_task_whiteboard(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_task(
                task_id="task-1",
                title="Build memory system",
                state="planned",
                created_at=1_710_755_200,
            )
            runtime.save_task_whiteboard_text(
                task_id="task-1",
                text="""---
task_id: task-1
title: Build memory system
updated_at: 1710755201
---
## Goal
Build the framework-native memory layer.

## Constraints
Use Agent Framework context providers.

## Plan
Wire whiteboards into agent runs.

## Decisions
Persist whiteboards as markdown.

## Open Questions

## Acceptance Criteria
Selected tasks show their whiteboard in the TUI.
""",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                memory = app.query_one("#memory", Panel)
                self.assertIn("Whiteboard: task-1", memory.body)
                self.assertIn("Build the framework-native memory layer.", memory.body)
                self.assertIn("Persist whiteboards as markdown.", memory.body)

    async def test_app_renders_artifacts(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_artifact(
                artifact_id="artifact-1",
                kind="design-note",
                title="Architecture Notes",
                content="Use Textual with a runtime-first architecture.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                artifacts = app.query_one("#artifacts", Panel)
                self.assertIn("> artifact-1", artifacts.body)
                self.assertIn("Architecture Notes", artifacts.body)
                self.assertIn("Use Textual with a runtime-first architecture.", artifacts.body)

    async def test_app_can_switch_selected_artifact(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_artifact(
                artifact_id="artifact-1",
                kind="design-note",
                title="Architecture Notes",
                content="First artifact body.",
                created_at=1_710_755_200,
            )
            runtime.create_artifact(
                artifact_id="artifact-2",
                kind="review-note",
                title="Review Notes",
                content="Second artifact body.",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                artifacts = app.query_one("#artifacts", Panel)
                self.assertIn("> artifact-1", artifacts.body)
                self.assertIn("First artifact body.", artifacts.body)

                app.action_next_artifact()

                artifacts = app.query_one("#artifacts", Panel)
                self.assertIn("> artifact-2", artifacts.body)
                self.assertIn("Second artifact body.", artifacts.body)

    async def test_app_can_switch_selected_agent(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                team = app.query_one("#team", Panel)
                self.assertIn("> orchestrator", team.body)
                self.assertIn("Role: orchestrator", team.body)
                self.assertIn("Tools: read_file, write_file, patch_file, run_command", team.body)

                app.action_next_agent()

                self.assertIn("> researcher", team.body)
                self.assertIn("Role: researcher", team.body)
                self.assertIn("Tools: search_files, web_lookup", team.body)

    async def test_app_can_switch_selected_workflow(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                workflows = app.query_one("#workflows", Panel)
                self.assertIn("> standard-build", workflows.body)
                self.assertIn("Orchestration: sequential", workflows.body)
                self.assertIn("Steps: architect -> coder -> reviewer", workflows.body)
                self.assertIn("Run the normal plan-build-review-fix loop.", workflows.body)

                app.action_next_workflow()

                self.assertIn("> test-driven-repair", workflows.body)
                self.assertIn("Orchestration: sequential", workflows.body)

    async def test_app_can_start_selected_workflow(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()

                runs = app.query_one("#workflow-runs", Panel)
                tasks = app.query_one("#tasks", Panel)
                threads = app.query_one("#threads", Panel)
                self.assertIn("standard-build", runs.body)
                self.assertIn("> workflow-run-", runs.body)
                self.assertIn("[blocked]", runs.body)
                self.assertIn("Workflow: standard-build", tasks.body)
                self.assertIn("[blocked] standard-build: architect", tasks.body)
                self.assertIn("agent_direct:architect", threads.body)
                self.assertIn("Root: task-", runs.body)
                self.assertIn("Next agent: architect", runs.body)

    async def test_app_shows_parallel_workflow_group_details(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)
            app.selected_workflow_id = "best-of-n"

            async with app.run_test():
                workflows = app.query_one("#workflows", Panel)
                self.assertIn("Steps: coder + coder + coder -> reviewer", workflows.body)

                workflow_run, _ = runtime.start_workflow_run(
                    workflow_id="best-of-n",
                    created_at=1_710_755_200,
                )
                app.selected_workflow_run_id = workflow_run.id
                app._refresh_panels()

                runs = app.query_one("#workflow-runs", Panel)
                tasks = app.query_one("#tasks", Panel)
                self.assertIn("Next agent: coder + coder + coder", runs.body)
                self.assertIn("best-of-n: coder x3", tasks.body)

    async def test_app_starting_workflow_can_kick_off_first_agent_thread(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text="Architecture kickoff received.")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                async with app.run_test():
                    await app.action_start_selected_workflow()

                    selected_thread = app.query_one("#selected-thread", Panel)
                    self.assertIn("Workflow kickoff: standard-build", selected_thread.body)
                    self.assertIn("Build the next feature.", selected_thread.body)
                    self.assertIn("Architecture kickoff received.", selected_thread.body)

    async def test_app_can_advance_selected_workflow_run(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)
            fake_agents = {
                "architect": FakeAgent("Architecture kickoff received."),
                "coder": FakeAgent("Implementation underway."),
                "reviewer": FakeAgent("Review done."),
            }

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ):
                async with app.run_test():
                    await app.action_start_selected_workflow()
                    await app.action_advance_selected_workflow_run()

                    runs = app.query_one("#workflow-runs", Panel)
                    selected_thread = app.query_one("#selected-thread", Panel)
                    self.assertIn("step=2 standard-build", runs.body)
                    self.assertIn("[orchestrator] Continue workflow: standard-build", selected_thread.body)
                    self.assertIn("[coder] Implementation underway.", selected_thread.body)
                    self.assertIn("[completed] standard-build: coder", app.query_one("#tasks", Panel).body)

    async def test_app_shows_completion_summary_in_main_chat(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(f"{agent_id} complete."),
            ):
                async with app.run_test():
                    app.selected_workflow_id = "single-agent-execution"
                    await app.action_start_selected_workflow()

                    main_chat = app.query_one("#main-chat", Panel)
                    artifacts = app.query_one("#artifacts", Panel)
                    self.assertIn("Workflow complete: single-agent-execution", main_chat.body)
                    self.assertIn("Final output from coder:", main_chat.body)
                    self.assertIn("workflow-report", artifacts.body)
                    self.assertIn("Workflow Report: single-agent-execution", artifacts.body)

    async def test_app_can_request_fix_cycle_for_selected_workflow_run(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)
            fake_agents = {
                "architect": FakeAgent("Architecture ready."),
                "coder": FakeAgent("Implementation ready."),
                "reviewer": FakeAgent("Needs fixes."),
                "fixer": FakeAgent("Fixes applied."),
            }

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ):
                async with app.run_test():
                    await app.action_start_selected_workflow()
                    await app.action_advance_selected_workflow_run()
                    await app.action_advance_selected_workflow_run()
                    app.action_request_fix_cycle_for_selected_workflow_run()

                    runs = app.query_one("#workflow-runs", Panel)
                    threads = app.query_one("#threads", Panel)
                    tasks = app.query_one("#tasks", Panel)
                    self.assertIn("[repairing]", runs.body)
                    self.assertIn("agent_direct:fixer", threads.body)
                    self.assertIn("[planned] standard-build: fixer", tasks.body)

    async def test_app_can_switch_selected_thread_view(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
            )
            runtime.append_message_to_thread(
                thread_id="thread-review-1",
                message_id="message-1",
                sender="reviewer",
                kind="review",
                body="Please rename this method.",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                selected_thread = app.query_one("#selected-thread", Panel)
                threads = app.query_one("#threads", Panel)

                self.assertIn("thread-main", selected_thread.body)
                self.assertIn("> thread-main", threads.body)

                app.action_next_thread()

                self.assertIn("thread-review-1", selected_thread.body)
                self.assertIn("Please rename this method.", selected_thread.body)
                self.assertIn("> thread-review-1", threads.body)

    async def test_submitting_input_persists_a_user_message(self) -> None:
        from textual.widgets import Input

        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                composer = app.query_one("#composer-input", Input)
                composer.value = "Ship the next slice."
                await pilot.press("enter")

                main_chat = app.query_one("#main-chat", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("Ship the next slice.", main_chat.body)
                self.assertIn("message_created", activity.body)
                self.assertEqual(len(runtime.list_main_messages()), 1)

    async def test_submitting_input_can_render_an_orchestrator_reply(self) -> None:
        from textual.widgets import Input

        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text="I’m on it.")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                async with app.run_test() as pilot:
                    composer = app.query_one("#composer-input", Input)
                    composer.value = "Ship the next slice."
                    await pilot.press("enter")

                    main_chat = app.query_one("#main-chat", Panel)
                    self.assertIn("Ship the next slice.", main_chat.body)
                    self.assertIn("I’m on it.", main_chat.body)

    async def test_submitting_input_in_agent_thread_renders_agent_reply(self) -> None:
        from textual.widgets import Input

        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text="Architecture outline ready.")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)
            app.selected_agent_id = "architect"

            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                async with app.run_test() as pilot:
                    app.action_open_selected_agent_thread()
                    composer = app.query_one("#composer-input", Input)
                    composer.value = "Design the next component."
                    await pilot.press("enter")

                    selected_thread = app.query_one("#selected-thread", Panel)
                    self.assertIn("[orchestrator] Design the next component.", selected_thread.body)
                    self.assertIn("[architect] Architecture outline ready.", selected_thread.body)

    async def test_app_can_edit_orchestrator_definition(self) -> None:
        from textual.widgets import TextArea

        from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                app.action_edit_orchestrator_definition()
                await pilot.pause()

                self.assertIsInstance(app.screen, DefinitionEditorScreen)
                editor = app.screen.query_one("#definition-editor", TextArea)
                editor.load_text(
                    """---
id: orchestrator
name: Orchestrator
role: orchestrator
temperature: 0.2
tools:
  - read_file
---
## Identity
Lead engineer for the AI firm.

## Output Style
Be extremely concise.
"""
                )
                app.screen.action_save()
                await pilot.pause()

                self.assertIn("definition_saved", app.query_one("#activity", Panel).body)
                self.assertEqual(
                    runtime.registry.agent_definitions["orchestrator"].sections["Output Style"],
                    "Be extremely concise.",
                )

    async def test_app_can_edit_selected_agent_definition(self) -> None:
        from textual.widgets import TextArea

        from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)
            app.selected_agent_id = "architect"

            async with app.run_test() as pilot:
                app.action_edit_selected_agent_definition()
                await pilot.pause()

                self.assertIsInstance(app.screen, DefinitionEditorScreen)
                editor = app.screen.query_one("#definition-editor", TextArea)
                editor.load_text(
                    """---
id: architect
name: Architect
role: architect
temperature: 0.2
tools:
  - read_file
  - search_files
---
## Identity
You are the architect for the AI firm.

## Output Style
Be concise and structural.
"""
                )
                app.screen.action_save()
                await pilot.pause()

                team = app.query_one("#team", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("> architect", team.body)
                self.assertIn("definition_saved", activity.body)
                self.assertEqual(
                    runtime.registry.agent_definitions["architect"].sections["Output Style"],
                    "Be concise and structural.",
                )

    async def test_app_can_edit_global_config(self) -> None:
        from textual.widgets import TextArea

        from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                app.action_edit_global_config()
                await pilot.pause()

                self.assertIsInstance(app.screen, DefinitionEditorScreen)
                editor = app.screen.query_one("#definition-editor", TextArea)
                editor.load_text(
                    """{
  "providers": {
                    "local": {
      "type": "openai_chat",
      "base_url": "http://localhost:8080/v1",
      "api_key": "not-needed",
      "model": "qwen2.5-coder",
      "capabilities": {
        "tool_calling": true,
        "structured_output": true
      }
    }
  },
  "role_assignments": {
    "orchestrator": "local"
  },
  "approvals": {},
  "ui": {}
}
"""
                )
                app.screen.action_save()
                await pilot.pause()

                settings = app.query_one("#settings", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("Orchestrator: ready via local (qwen2.5-coder)", settings.body)
                self.assertIn("local: openai_chat qwen2.5-coder @ http://localhost:8080/v1", settings.body)
                self.assertIn("tool_calling=True", settings.body)
                self.assertIn("Assignments: orchestrator->local", settings.body)
                self.assertIn("config_saved", activity.body)

    async def test_app_can_run_workspace_command_from_editor(self) -> None:
        from textual.widgets import TextArea

        from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                app.action_run_workspace_command()
                await pilot.pause()

                self.assertIsInstance(app.screen, DefinitionEditorScreen)
                editor = app.screen.query_one("#definition-editor", TextArea)
                editor.load_text("pwd")
                app.screen.action_save()
                await pilot.pause()

                commands = app.query_one("#commands", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("> command-run-", commands.body)
                self.assertIn("[completed/0] pwd", commands.body)
                self.assertIn("command_run", activity.body)

    async def test_app_can_edit_selected_workflow_definition(self) -> None:
        from textual.widgets import TextArea

        from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)
            app.selected_workflow_id = "standard-build"

            async with app.run_test() as pilot:
                app.action_edit_selected_workflow_definition()
                await pilot.pause()

                self.assertIsInstance(app.screen, DefinitionEditorScreen)
                editor = app.screen.query_one("#definition-editor", TextArea)
                editor.load_text(
                    """---
id: standard-build
name: Standard Build
kind: workflow
orchestration: sequential
---
## Purpose
Ship standard implementation work.

## Exit Conditions
Return reviewed code and a clear summary.
"""
                )
                app.screen.action_save()
                await pilot.pause()

                workflows = app.query_one("#workflows", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIn("> standard-build", workflows.body)
                self.assertIn("definition_saved", activity.body)
                self.assertEqual(
                    runtime.registry.workflow_definitions["standard-build"].sections["Exit Conditions"],
                    "Return reviewed code and a clear summary.",
                )

    async def test_app_renders_selected_workflow_run_as_task_tree(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()

                tasks = app.query_one("#tasks", Panel)
                self.assertIn("Run: workflow-run-", tasks.body)
                self.assertIn("Root: > task-", tasks.body)
                self.assertIn("State: in_progress", tasks.body)
                self.assertIn("standard-build: architect", tasks.body)
                self.assertIn("thread-agent-architect-", tasks.body)

    async def test_app_can_switch_selected_task(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_task(
                task_id="task-1",
                title="First task",
                state="planned",
                created_at=1_710_755_200,
            )
            runtime.create_task(
                task_id="task-2",
                title="Second task",
                state="blocked",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                tasks = app.query_one("#tasks", Panel)
                self.assertIn("> task-1", tasks.body)
                self.assertIn("First task", tasks.body)

                app.action_next_task()

                tasks = app.query_one("#tasks", Panel)
                self.assertIn("> task-2", tasks.body)
                self.assertIn("State: blocked", tasks.body)
                self.assertIn("Second task", tasks.body)

    async def test_switching_workflow_runs_focuses_the_related_thread(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()
                first_run = runtime.list_workflow_runs()[0]
                first_thread_id = first_run.last_thread_id

                await app.action_start_selected_workflow()
                second_run = runtime.list_workflow_runs()[1]
                self.assertEqual(app.selected_workflow_run_id, second_run.id)
                self.assertEqual(app.selected_thread_id, second_run.last_thread_id)

                app.action_previous_workflow_run()

                selected_thread = app.query_one("#selected-thread", Panel)
                self.assertEqual(app.selected_workflow_run_id, first_run.id)
                self.assertEqual(app.selected_thread_id, first_thread_id)
                self.assertIn(first_thread_id, selected_thread.body)

    async def test_selected_workflow_run_scopes_threads_panel(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
                summary="Unrelated review",
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()

                threads = app.query_one("#threads", Panel)
                self.assertIn("Run: workflow-run-", threads.body)
                self.assertIn("thread-agent-architect-", threads.body)
                self.assertNotIn("thread-review-1", threads.body)

    async def test_thread_navigation_uses_selected_workflow_run_threads(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()

                first_thread_id = app.selected_thread_id
                app.action_next_thread()
                second_thread_id = app.selected_thread_id

                self.assertNotEqual(first_thread_id, second_thread_id)
                self.assertNotEqual(second_thread_id, runtime.main_thread_id)
                self.assertIn(second_thread_id, app.query_one("#selected-thread", Panel).body)

    async def test_switching_workflow_runs_updates_artifacts_panel(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        responses = iter(["First run done.", "Second run done."])

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(next(responses)),
            ):
                async with app.run_test():
                    app.selected_workflow_id = "single-agent-execution"
                    await app.action_start_selected_workflow()
                    first_run = runtime.list_workflow_runs()[0]
                    first_artifact_id = runtime.list_artifacts_for_workflow_run(first_run.id)[0].id

                    await app.action_start_selected_workflow()
                    second_run = runtime.list_workflow_runs()[1]
                    second_artifact_id = runtime.list_artifacts_for_workflow_run(second_run.id)[0].id

                    artifacts = app.query_one("#artifacts", Panel)
                    self.assertIn(second_run.id, artifacts.body)
                    self.assertIn(second_artifact_id, artifacts.body)
                    self.assertNotIn(first_artifact_id, artifacts.body)

                    app.action_previous_workflow_run()

                    artifacts = app.query_one("#artifacts", Panel)
                    self.assertIn(first_run.id, artifacts.body)
                    self.assertIn(first_artifact_id, artifacts.body)
                    self.assertNotIn(second_artifact_id, artifacts.body)

    async def test_selected_workflow_run_scopes_activity_panel(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        responses = iter(["First run done.", "Second run done."])

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            runtime.append_event(
                kind="unrelated_event",
                summary="Unrelated activity",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(next(responses)),
            ):
                async with app.run_test():
                    app.selected_workflow_id = "single-agent-execution"
                    await app.action_start_selected_workflow()
                    first_run = runtime.list_workflow_runs()[0]

                    await app.action_start_selected_workflow()
                    second_run = runtime.list_workflow_runs()[1]

                    activity = app.query_one("#activity", Panel)
                    self.assertIn(second_run.id, activity.body)
                    self.assertNotIn("unrelated_event", activity.body)

                    app.action_previous_workflow_run()

                    activity = app.query_one("#activity", Panel)
                    self.assertIn(first_run.id, activity.body)
                    self.assertNotIn("unrelated_event", activity.body)

    async def test_selected_workflow_run_scopes_commands_panel(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        responses = iter(["First run done.", "Second run done."])

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(next(responses)),
            ):
                async with app.run_test():
                    app.selected_workflow_id = "single-agent-execution"
                    await app.action_start_selected_workflow()
                    first_run = runtime.list_workflow_runs()[0]
                    first_thread = runtime.list_threads_for_workflow_run(first_run.id)[0]
                    runtime.run_workspace_command(
                        "printf first_only",
                        created_at=1_710_755_300,
                        thread_id=first_thread.id,
                        task_id=first_thread.parent_task_id,
                        agent_id="tester",
                    )

                    await app.action_start_selected_workflow()
                    second_run = runtime.list_workflow_runs()[1]
                    second_thread = runtime.list_threads_for_workflow_run(second_run.id)[0]
                    runtime.run_workspace_command(
                        "printf second_only",
                        created_at=1_710_755_301,
                        thread_id=second_thread.id,
                        task_id=second_thread.parent_task_id,
                        agent_id="tester",
                    )
                    app._refresh_panels()

                    commands = app.query_one("#commands", Panel)
                    self.assertIn(second_run.id, commands.body)
                    self.assertIn("second_only", commands.body)
                    self.assertNotIn("first_only", commands.body)

                    app.action_previous_workflow_run()

                    commands = app.query_one("#commands", Panel)
                    self.assertIn(first_run.id, commands.body)
                    self.assertIn("first_only", commands.body)
                    self.assertNotIn("second_only", commands.body)

    async def test_app_can_clear_workflow_run_focus(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
                summary="Unrelated review",
            )
            runtime.append_event(
                kind="unrelated_event",
                summary="Unrelated activity",
                created_at=1_710_755_201,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                await app.action_start_selected_workflow()
                self.assertIsNotNone(app.selected_workflow_run_id)

                app.action_clear_workflow_run_focus()

                tasks = app.query_one("#tasks", Panel)
                threads = app.query_one("#threads", Panel)
                activity = app.query_one("#activity", Panel)
                self.assertIsNone(app.selected_workflow_run_id)
                self.assertNotIn("Run: workflow-run-", tasks.body)
                self.assertIn("thread-main", threads.body)
                self.assertIn("thread-review-1", threads.body)
                self.assertNotIn("Run: workflow-run-", activity.body)
