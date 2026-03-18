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
                self.assertIsNotNone(app.query_one("#artifacts"))
                self.assertIsNotNone(app.query_one("#approvals"))
                self.assertIsNotNone(app.query_one("#memory"))
                self.assertIsNotNone(app.query_one("#team"))
                self.assertIsNotNone(app.query_one("#settings"))
                self.assertIn("thread-main", app.query_one("#threads", Panel).body)
                self.assertIn("> orchestrator", app.query_one("#team", Panel).body)
                self.assertIn("> standard-build", app.query_one("#workflows", Panel).body)
                self.assertIn("No workflow runs yet.", app.query_one("#workflow-runs", Panel).body)
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

                app.action_next_agent()

                self.assertIn("> researcher", team.body)

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

                app.action_next_workflow()

                self.assertIn("> test-driven-repair", workflows.body)

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
      "model": "qwen2.5-coder"
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
                self.assertIn("config_saved", activity.body)

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
                self.assertIn("Root: task-", tasks.body)
                self.assertIn("standard-build: architect", tasks.body)
                self.assertIn("thread-agent-architect-", tasks.body)

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
