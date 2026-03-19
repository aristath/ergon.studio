from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from textual.widgets import Collapsible, Input, RichLog, Static

from ergon_studio.tui.widgets import ComposerTextArea

from ergon_studio.runtime import load_runtime
from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp
from ergon_studio.tui.widgets import AgentStatusBar, InfoBar, SideThreadBlock


def _richlog_text(app: ErgonStudioApp, widget_id: str = "#main-chat") -> str:
    """Extract plain text from a RichLog widget for assertions."""
    log = app.query_one(widget_id, RichLog)
    lines: list[str] = []
    for y in range(200):
        try:
            strip = log.render_line(y)
            text = strip.text.strip()
            if text:
                lines.append(text)
        except Exception:
            break
    return "\n".join(lines)


class FakeAgent:
    def __init__(self, reply: str = "agent reply") -> None:
        self._reply = reply

    def create_session(self, *, session_id: str | None = None, **_: object):
        from agent_framework import AgentSession

        return AgentSession(session_id=session_id)

    async def run(self, messages=None, *, session=None, **_: object):
        return SimpleNamespace(text=self._reply)


def _make_env():
    tmp = tempfile.mkdtemp()
    base = Path(tmp)
    project_root = base / "repo"
    home_dir = base / "home"
    project_root.mkdir()
    home_dir.mkdir()
    runtime = load_runtime(project_root=project_root, home_dir=home_dir)
    app = ErgonStudioApp(runtime)
    return tmp, runtime, app


class TestAppRendering(IsolatedAsyncioTestCase):
    async def test_app_renders_status_bar(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            self.assertIsNotNone(bar)

    async def test_app_renders_main_chat(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            chat = app.query_one("#main-chat", RichLog)
            self.assertIsNotNone(chat)
            text = _richlog_text(app)
            self.assertIn("Workspace", text)

    async def test_app_renders_input_and_info_bar(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIsNotNone(inp)
            self.assertIn("orchestrator", str(inp.placeholder))
            info = app.query_one("#info-bar", InfoBar)
            self.assertIsNotNone(info)

    async def test_info_bar_shows_current_session_title(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            info = app.query_one("#info-bar", InfoBar)
            self.assertIn("Main Session", str(info.content))

    async def test_app_renders_existing_messages(self):
        _, runtime, app = _make_env()
        runtime.append_message_to_main_thread(
            message_id="msg-1", sender="user", kind="text",
            body="hello there", created_at=1000,
        )
        runtime.append_message_to_main_thread(
            message_id="msg-2", sender="orchestrator", kind="text",
            body="hi back", created_at=1001,
        )
        async with app.run_test() as pilot:
            text = _richlog_text(app)
            self.assertIn("hello there", text)
            self.assertIn("hi back", text)

    async def test_app_renders_side_threads_as_collapsibles(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-arch", kind="agent_direct", created_at=1000,
            assigned_agent_id="architect", summary="design",
        )
        async with app.run_test() as pilot:
            blocks = app.query(SideThreadBlock)
            self.assertEqual(len(blocks), 1)


class TestMessages(IsolatedAsyncioTestCase):
    async def test_submitting_input_persists_message(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "test message"
            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                await pilot.press("enter")
                await pilot.pause()
            messages = runtime.list_main_messages()
            user_bodies = [
                runtime.conversation_store.read_message_body(m)
                for m in messages if m.sender == "user"
            ]
            self.assertTrue(any("test message" in b for b in user_bodies))

    async def test_submitting_input_renders_orchestrator_reply(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "hello"
            with patch.object(type(runtime), "build_agent", return_value=FakeAgent("orchestrator says hi")):
                await pilot.press("enter")
                await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("orchestrator says hi", text)

    async def test_input_placeholder_shows_orchestrator_target(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIn("orchestrator", str(inp.placeholder))


class TestApprovals(IsolatedAsyncioTestCase):
    async def test_pending_approval_appears_in_chat(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1", requester="coder", action="write_file",
            risk_class="moderate", reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            text = _richlog_text(app)
            self.assertIn("Approval", text)
            self.assertIn("write_file", text)

    async def test_ctrl_y_approves_pending_approval(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1", requester="coder", action="write_file",
            risk_class="moderate", reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            app.action_approve_pending()
            await pilot.pause()
            approvals = runtime.list_approvals()
            self.assertEqual(approvals[0].status, "approved")
            text = _richlog_text(app)
            self.assertIn("Approved", text)

    async def test_ctrl_r_rejects_pending_approval(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1", requester="coder", action="write_file",
            risk_class="moderate", reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            app.action_reject_pending()
            await pilot.pause()
            approvals = runtime.list_approvals()
            self.assertEqual(approvals[0].status, "rejected")
            text = _richlog_text(app)
            self.assertIn("Rejected", text)

    async def test_info_bar_shows_pending_approval_count(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1", requester="coder", action="write_file",
            risk_class="moderate", reason="update readme",
            created_at=1000,
        )
        runtime.request_approval(
            approval_id="appr-2", requester="coder", action="run_command",
            risk_class="high", reason="install dep",
            created_at=1001,
        )
        async with app.run_test() as pilot:
            info = app.query_one("#info-bar", InfoBar)
            info.refresh_from_runtime()
            await pilot.pause()


class TestSideThreads(IsolatedAsyncioTestCase):
    async def test_side_thread_appears_as_collapsible(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-coder", kind="agent_direct", created_at=1000,
            assigned_agent_id="coder", summary="implement auth",
        )
        async with app.run_test() as pilot:
            blocks = app.query(SideThreadBlock)
            self.assertEqual(len(blocks), 1)

    async def test_collapsible_title_shows_agent_info(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-coder", kind="agent_direct", created_at=1000,
            assigned_agent_id="coder", summary="implement auth",
        )
        runtime.append_message_to_thread(
            thread_id="t-coder", message_id="m-1", sender="orchestrator",
            kind="text", body="implement auth module", created_at=1001,
        )
        async with app.run_test() as pilot:
            blocks = app.query(SideThreadBlock)
            self.assertEqual(len(blocks), 1)
            title = blocks.first().title
            self.assertIn("coder", title)
            self.assertIn("1 msgs", title)

    async def test_expanding_thread_shows_messages(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-coder", kind="agent_direct", created_at=1000,
            assigned_agent_id="coder", summary="implement auth",
        )
        runtime.append_message_to_thread(
            thread_id="t-coder", message_id="m-1", sender="orchestrator",
            kind="text", body="implement the auth module", created_at=1001,
        )
        async with app.run_test() as pilot:
            block = app.query(SideThreadBlock).first()
            block.collapsed = False
            block.refresh_messages()
            await pilot.pause()


class TestSlashCommands(IsolatedAsyncioTestCase):
    async def test_help_shows_commands(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("/session", text)
            self.assertIn("/sessions", text)
            self.assertIn("/config", text)
            self.assertIn("/workflows", text)

    async def test_session_shows_current_session(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/session"
            await pilot.press("enter")
            await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("Main Session", text)
            self.assertIn("session-main", text)

    async def test_sessions_lists_project_sessions(self):
        _, runtime, app = _make_env()
        second = load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        self.assertNotEqual(second.main_session_id, runtime.main_session_id)
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("Main Session", text)
            self.assertIn("Parallel lane", text)

    async def test_new_session_creates_and_switches_runtime(self):
        _, runtime, app = _make_env()
        original_session_id = runtime.main_session_id
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/new-session Parallel lane"
            await pilot.press("enter")
            await pilot.pause()
            self.assertNotEqual(app.runtime.main_session_id, original_session_id)
            self.assertEqual(app.runtime.current_session().title, "Parallel lane")
            text = _richlog_text(app)
            self.assertIn("Switched", text)
            self.assertIn("Parallel lane", text)

    async def test_switch_session_reopens_specific_session(self):
        _, runtime, app = _make_env()
        first_session_id = runtime.main_session_id
        runtime.append_message_to_main_thread(
            message_id="msg-main",
            sender="user",
            kind="text",
            body="main session message",
            created_at=10,
        )
        second = load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        async with app.run_test() as pilot:
            app._replace_runtime(second)
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = f"/switch-session {first_session_id}"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(app.runtime.main_session_id, first_session_id)
            text = _richlog_text(app)
            self.assertIn("main session message", text)

    async def test_workflows_lists_inline(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/workflows"
            await pilot.press("enter")
            await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("Workflows", text)

    async def test_workflow_selects_workflow(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            wf_ids = runtime.list_workflow_ids()
            if wf_ids:
                inp = app.query_one("#composer-input", ComposerTextArea)
                app.set_focus(inp)
                inp.value = f"/workflow {wf_ids[0]}"
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.selected_workflow_id, wf_ids[0])

    async def test_agent_opens_thread(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            agent_ids = runtime.list_agent_ids()
            if agent_ids:
                inp = app.query_one("#composer-input", ComposerTextArea)
                app.set_focus(inp)
                inp.value = f"/agent {agent_ids[0]}"
                await pilot.press("enter")
                await pilot.pause()
                blocks = app.query(SideThreadBlock)
                self.assertTrue(len(blocks) >= 1)

    async def test_unknown_command_shows_error(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/nonexistent"
            await pilot.press("enter")
            await pilot.pause()
            text = _richlog_text(app)
            self.assertIn("Unknown command", text)

    async def test_slash_shows_command_list(self):
        from textual.widgets import OptionList

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/"
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertTrue(cmd_list.has_class("visible"))
            self.assertTrue(cmd_list.option_count > 0)

    async def test_slash_filters_as_you_type(self):
        from textual.widgets import OptionList

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/con"
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertTrue(cmd_list.has_class("visible"))
            # Should only show /config
            self.assertEqual(cmd_list.option_count, 1)

    async def test_slash_list_hides_on_space(self):
        from textual.widgets import OptionList

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/workflow "
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertFalse(cmd_list.has_class("visible"))


class TestStatusBar(IsolatedAsyncioTestCase):
    async def test_status_bar_shows_agents(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            bar.refresh_from_runtime()
            await pilot.pause()
            self.assertIsNotNone(bar)

    async def test_status_bar_set_agent_state(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            bar.set_agent_state("coder", "active")
            self.assertEqual(bar._agent_states["coder"], "active")


class TestEditorModals(IsolatedAsyncioTestCase):
    async def test_edit_global_config_opens_wizard(self):
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            app.action_edit_global_config()
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, ConfigWizardScreen)

    async def test_slash_config_opens_wizard(self):
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/config"
            await pilot.press("enter")
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, ConfigWizardScreen)

    async def test_edit_orchestrator_definition(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            app.action_edit_orchestrator_definition()
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, DefinitionEditorScreen)

    async def test_run_workspace_command(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            app.action_run_workspace_command()
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, DefinitionEditorScreen)


class TestConfigWizard(IsolatedAsyncioTestCase):
    async def test_provider_editor_saves_provider(self):
        from ergon_studio.tui.config_wizard import ProviderEditorScreen

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            result_holder: list[dict | None] = []

            def capture(result):
                result_holder.append(result)

            screen = ProviderEditorScreen(
                name="local",
                base_url="http://localhost:11434/v1",
                model="qwen3:8b",
            )
            app.push_screen(screen, capture)
            await pilot.pause()

            # The screen should be mounted with pre-filled values
            name_input = screen.query_one("#name-input", Input)
            self.assertEqual(name_input.value, "local")

            # Trigger save
            screen._save()
            await pilot.pause()
            self.assertEqual(len(result_holder), 1)
            result = result_holder[0]
            self.assertIsNotNone(result)
            self.assertEqual(result["name"], "local")
            self.assertEqual(result["model"], "qwen3:8b")
            self.assertEqual(result["base_url"], "http://localhost:11434/v1")

    async def test_provider_editor_requires_fields(self):
        from ergon_studio.tui.config_wizard import ProviderEditorScreen

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            result_holder: list[dict | None] = []
            screen = ProviderEditorScreen()
            app.push_screen(screen, lambda r: result_holder.append(r))
            await pilot.pause()

            # Try to save with empty fields — should not dismiss
            screen._save()
            await pilot.pause()
            self.assertEqual(len(result_holder), 0)

    async def test_wizard_auto_assigns_first_provider(self):
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            screen = ConfigWizardScreen(runtime)
            app.push_screen(screen)
            await pilot.pause()

            # Simulate a provider being saved
            screen._on_provider_saved({
                "name": "local",
                "type": "openai_chat",
                "model": "test-model",
                "base_url": "http://localhost:8080/v1",
            })
            await pilot.pause()

            config = json.loads(runtime.read_global_config_text())
            self.assertIn("local", config["providers"])
            # Should auto-assign to all roles
            assignments = config.get("role_assignments", {})
            self.assertTrue(len(assignments) > 0)
            for role, provider in assignments.items():
                self.assertEqual(provider, "local")


class TestThreadTargeting(IsolatedAsyncioTestCase):
    async def test_expanding_thread_changes_input_target(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-arch", kind="agent_direct", created_at=1000,
            assigned_agent_id="architect", summary="design",
        )
        async with app.run_test() as pilot:
            block = app.query(SideThreadBlock).first()
            block.collapsed = False
            app.on_collapsible_expanded(Collapsible.Expanded(block))
            await pilot.pause()
            self.assertEqual(app._target_thread_id, "t-arch")
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIn("architect", str(inp.placeholder))

    async def test_collapsing_thread_resets_to_main(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-arch", kind="agent_direct", created_at=1000,
            assigned_agent_id="architect", summary="design",
        )
        async with app.run_test() as pilot:
            block = app.query(SideThreadBlock).first()
            app.on_collapsible_expanded(Collapsible.Expanded(block))
            app.on_collapsible_collapsed(Collapsible.Collapsed(block))
            await pilot.pause()
            self.assertIsNone(app._target_thread_id)
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIn("orchestrator", str(inp.placeholder))
