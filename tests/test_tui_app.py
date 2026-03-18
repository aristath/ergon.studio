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
                self.assertIsNotNone(app.query_one("#threads"))
                self.assertIsNotNone(app.query_one("#activity"))
                self.assertIsNotNone(app.query_one("#artifacts"))
                self.assertIsNotNone(app.query_one("#approvals"))
                self.assertIsNotNone(app.query_one("#memory"))
                self.assertIsNotNone(app.query_one("#settings"))
                self.assertIn("thread-main", app.query_one("#threads", Panel).body)
                self.assertIn("No tasks yet.", app.query_one("#tasks", Panel).body)
                self.assertIn("No activity yet.", app.query_one("#activity", Panel).body)
                self.assertIn("No approvals pending.", app.query_one("#approvals", Panel).body)
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
                self.assertIn("task-1", tasks.body)
                self.assertIn("Build task sidebar", tasks.body)
                self.assertIn("task_created", activity.body)

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
                self.assertIn("approval-1", approvals.body)
                self.assertIn("write_file", approvals.body)

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
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                memory = app.query_one("#memory", Panel)
                self.assertIn("fact-1", memory.body)
                self.assertIn("Use Textual for the TUI.", memory.body)

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
                self.assertIn("artifact-1", artifacts.body)
                self.assertIn("Architecture Notes", artifacts.body)

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
