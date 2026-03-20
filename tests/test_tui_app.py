from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from agent_framework import AgentSession, ResponseStream
from textual.widgets import Input, OptionList

from ergon_studio.runtime import load_runtime
from ergon_studio.tui.app import DefinitionEditorScreen, ErgonStudioApp, SessionPickerScreen
from ergon_studio.tui.inspectors import InspectorScreen
from ergon_studio.tui.timeline_widgets import TimelineView
from ergon_studio.tui.widgets import AgentStatusBar, ComposerTextArea, InfoBar, ThinkingIndicator


def _timeline_text(app: ErgonStudioApp) -> str:
    return app.query_one("#main-timeline", TimelineView).plain_text()


class FakeAgent:
    def __init__(self, reply: str = "agent reply") -> None:
        self._reply = reply

    def create_session(self, *, session_id: str | None = None, **_: object):
        from agent_framework import AgentSession

        return AgentSession(session_id=session_id)

    async def run(self, messages=None, *, session=None, **_: object):
        return SimpleNamespace(text=self._reply)


class FakeStreamingAgent:
    def __init__(self, deltas: list[str], *, delay: float = 0.01) -> None:
        self._deltas = deltas
        self._delay = delay

    def create_session(self, *, session_id: str | None = None, **_: object):
        return AgentSession(session_id=session_id)

    def run(self, messages=None, *, session=None, **_: object):
        del messages, session

        async def _updates():
            for delta in self._deltas:
                await asyncio.sleep(self._delay)
                yield SimpleNamespace(text=delta)

        return ResponseStream(
            _updates(),
            finalizer=lambda updates: SimpleNamespace(text="".join(update.text for update in updates)),
        )


class GateStreamingAgent:
    def __init__(
        self,
        deltas: list[str],
        *,
        release_event: asyncio.Event | None = None,
    ) -> None:
        self._deltas = deltas
        self.started = asyncio.Event()
        self._release_event = release_event or asyncio.Event()

    def create_session(self, *, session_id: str | None = None, **_: object):
        return AgentSession(session_id=session_id)

    def run(self, messages=None, *, session=None, **_: object):
        del messages, session

        async def _updates():
            self.started.set()
            await self._release_event.wait()
            for delta in self._deltas:
                yield SimpleNamespace(text=delta)

        return ResponseStream(
            _updates(),
            finalizer=lambda updates: SimpleNamespace(text="".join(update.text for update in updates)),
        )


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
    async def test_app_selects_metadata_driven_default_workflow(self):
        _, runtime, app = _make_env()
        async with app.run_test():
            self.assertEqual(app.selected_workflow_id, "standard-build")

    async def test_app_renders_status_bar(self):
        _, _, app = _make_env()
        async with app.run_test():
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            self.assertIsNotNone(bar)

    async def test_app_renders_main_timeline(self):
        _, _, app = _make_env()
        async with app.run_test():
            timeline = app.query_one("#main-timeline", TimelineView)
            self.assertIsNotNone(timeline)
            text = _timeline_text(app)
            self.assertIn("Workspace", text)

    async def test_app_renders_input_and_info_bar(self):
        _, _, app = _make_env()
        async with app.run_test():
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIn("orchestrator", str(inp.placeholder))
            info = app.query_one("#info-bar", InfoBar)
            self.assertIsNotNone(info)

    async def test_info_bar_shows_current_session_title(self):
        _, runtime, app = _make_env()
        async with app.run_test():
            info = app.query_one("#info-bar", InfoBar)
            self.assertIn(runtime.current_session().title, str(info.content))

    async def test_app_can_open_session_picker_on_mount(self):
        _, runtime, _ = _make_env()
        load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        app = ErgonStudioApp(runtime, open_session_picker_on_mount=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, SessionPickerScreen)
            options = app.screen.query_one("#session-picker-options", OptionList)
            option_text = "\n".join(str(options.get_option_at_index(i).prompt) for i in range(options.option_count))
            self.assertIn("Parallel lane", option_text)

    async def test_existing_messages_render_in_timeline(self):
        _, runtime, app = _make_env()
        runtime.append_message_to_main_thread(
            message_id="msg-1",
            sender="user",
            kind="text",
            body="hello there",
            created_at=1000,
        )
        runtime.append_message_to_main_thread(
            message_id="msg-2",
            sender="orchestrator",
            kind="text",
            body="hi back",
            created_at=1001,
        )
        async with app.run_test():
            text = _timeline_text(app)
            self.assertIn("hello there", text)
            self.assertIn("hi back", text)

    async def test_internal_threads_render_inline_in_timeline(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-arch",
            kind="agent_direct",
            created_at=1000,
            assigned_agent_id="architect",
            summary="design",
        )
        runtime.append_message_to_thread(
            thread_id="t-arch",
            message_id="m-1",
            sender="orchestrator",
            kind="text",
            body="User wants feature A. How should we do it?",
            created_at=1001,
        )
        runtime.append_message_to_thread(
            thread_id="t-arch",
            message_id="m-2",
            sender="architect",
            kind="text",
            body="Start by decomposing it into B and C.",
            created_at=1002,
        )
        async with app.run_test():
            text = _timeline_text(app)
            self.assertIn("Orchestrator <-> architect", text)
            self.assertIn("How should we do it?", text)
            self.assertIn("B and C", text)


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
                runtime.conversation_store.read_message_body(message)
                for message in messages
                if message.sender == "user"
            ]
            self.assertTrue(any("test message" in body for body in user_bodies))

    async def test_submitting_input_renders_orchestrator_reply(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "hello"
            with patch.object(type(runtime), "build_agent", return_value=FakeAgent("orchestrator says hi")):
                await pilot.press("enter")
                await pilot.pause()
            text = _timeline_text(app)
            self.assertIn("orchestrator says hi", text)

    async def test_send_to_orchestrator_renders_live_streaming_reply(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            with patch.object(
                type(runtime),
                "build_agent",
                return_value=FakeStreamingAgent(["stream", "ing reply"], delay=0.15),
            ):
                task = asyncio.create_task(app._send_to_orchestrator("hello"))
                text = ""
                saw_live = False
                saw_stream = False
                for _ in range(8):
                    await asyncio.sleep(0.03)
                    await pilot.pause()
                    text = _timeline_text(app)
                    saw_live = saw_live or "<live>" in text
                    saw_stream = saw_stream or "stream" in text
                    if saw_live and saw_stream:
                        break
                self.assertTrue(saw_stream)
                self.assertTrue(saw_live)
                await task
                await pilot.pause()
            text = _timeline_text(app)
            self.assertIn("streaming reply", text)
            self.assertNotIn("<live>", text)

    async def test_workflow_turn_renders_live_internal_workroom_activity(self):
        from ergon_studio.runtime import OrchestratorTurnDecision

        _, runtime, app = _make_env()

        async def decide(_runtime, *, body: str, created_at: int):
            del _runtime, created_at
            return OrchestratorTurnDecision(
                mode="workflow",
                workflow_id="standard-build",
                goal=body,
                deliverable_expected=True,
            )

        async def run_workflow(
            _runtime,
            *,
            workflow_id: str,
            goal: str,
            created_at: int | None = None,
            parent_thread_id: str | None = None,
        ):
            del workflow_id, goal, parent_thread_id
            thread = runtime.create_agent_thread(
                agent_id="coder",
                created_at=(created_at or 10) + 1,
            )
            runtime.live_state.start_draft(
                draft_id="draft-workflow-1",
                thread_id=thread.id,
                sender="coder",
                kind="chat",
                created_at=(created_at or 10) + 2,
            )
            await asyncio.sleep(0.02)
            runtime.live_state.append_delta(
                draft_id="draft-workflow-1",
                delta="Drafting implementation",
                created_at=(created_at or 10) + 3,
            )
            await asyncio.sleep(0.15)
            reply = runtime.append_message_to_thread(
                thread_id=thread.id,
                message_id="message-workflow-coder",
                sender="coder",
                kind="chat",
                body="Drafting implementation",
                created_at=(created_at or 10) + 4,
            )
            runtime.live_state.complete_draft(
                draft_id="draft-workflow-1",
                message_id=reply.id,
                created_at=(created_at or 10) + 4,
            )
            return {
                "status": "completed",
                "workflow_run_id": "workflow-run-1",
                "review_summary": "ACCEPTED: implemented",
                "last_thread_id": thread.id,
            }

        async with app.run_test() as pilot:
            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                task = asyncio.create_task(app._send_to_orchestrator("build it"))
                text = ""
                saw_live = False
                saw_draft = False
                saw_workroom = False
                for _ in range(8):
                    await asyncio.sleep(0.03)
                    await pilot.pause()
                    text = _timeline_text(app)
                    saw_workroom = saw_workroom or "Orchestrator <-> coder" in text
                    saw_draft = saw_draft or "Drafting implementation" in text
                    saw_live = saw_live or "<live>" in text
                    if saw_workroom and saw_draft and saw_live:
                        break
                self.assertTrue(saw_workroom)
                self.assertTrue(saw_draft)
                self.assertTrue(saw_live)
                await task
                await pilot.pause()
            text = _timeline_text(app)
            self.assertIn("standard-build", text)
            self.assertNotIn("<live>", text)

    async def test_submitting_while_turn_is_running_queues_the_next_message(self):
        from ergon_studio.runtime import DeliveryAuditDecision, OrchestratorTurnDecision

        _, runtime, app = _make_env()
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        orchestrator_agents = [
            GateStreamingAgent(["first reply"], release_event=first_release),
            GateStreamingAgent(["second reply"], release_event=second_release),
        ]

        def build_agent(_runtime, agent_id: str):
            if agent_id == "orchestrator" and orchestrator_agents:
                return orchestrator_agents.pop(0)
            return FakeAgent(f"{agent_id} ready")

        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=build_agent,
            ), patch.object(
                type(runtime),
                "_decide_orchestrator_turn",
                autospec=True,
                return_value=OrchestratorTurnDecision(mode="act", request=""),
            ), patch.object(
                type(runtime),
                "_audit_orchestrator_delivery_turn",
                autospec=True,
                return_value=DeliveryAuditDecision(
                    deliverable_expected=False,
                    reconsider=False,
                    reason="",
                ),
            ):
                inp.value = "first"
                await pilot.press("enter")
                for _ in range(20):
                    await asyncio.sleep(0.02)
                    await pilot.pause()
                    if app._active_turn_task is not None and runtime.list_live_message_drafts():
                        break
                inp.value = "second"
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(len(app._queued_turns), 1)
                self.assertIn("user: second", _timeline_text(app))
                self.assertIn("Queued message for the orchestrator", _timeline_text(app))
                first_release.set()
                second_release.set()
                for _ in range(20):
                    await asyncio.sleep(0.03)
                    await pilot.pause()
                    if len(runtime.list_main_messages()) >= 4 and not app._queued_turns and app._active_turn_task is None:
                        break

            bodies = [
                runtime.conversation_store.read_message_body(message).strip()
                for message in runtime.list_main_messages()
            ]
            self.assertEqual(bodies, ["first", "second", "first reply", "second reply"])

    async def test_background_current_hides_spinner_while_turn_keeps_running(self):
        from ergon_studio.runtime import DeliveryAuditDecision, OrchestratorTurnDecision

        _, runtime, app = _make_env()
        release = asyncio.Event()
        orchestrator_agent = GateStreamingAgent(["backgrounded reply"], release_event=release)

        def build_agent(_runtime, agent_id: str):
            if agent_id == "orchestrator":
                return orchestrator_agent
            return FakeAgent(f"{agent_id} ready")

        async with app.run_test() as pilot:
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=build_agent,
            ), patch.object(
                type(runtime),
                "_decide_orchestrator_turn",
                autospec=True,
                return_value=OrchestratorTurnDecision(mode="act", request=""),
            ), patch.object(
                type(runtime),
                "_audit_orchestrator_delivery_turn",
                autospec=True,
                return_value=DeliveryAuditDecision(
                    deliverable_expected=False,
                    reconsider=False,
                    reason="",
                ),
            ):
                task = asyncio.create_task(app._send_to_orchestrator("hello"))
                for _ in range(20):
                    await asyncio.sleep(0.02)
                    await pilot.pause()
                    if app._active_turn_task is not None and orchestrator_agent.started.is_set():
                        break
                thinking = app.query_one("#thinking", ThinkingIndicator)
                self.assertTrue(thinking.has_class("visible"))
                app.action_background_current()
                await pilot.pause()
                self.assertFalse(thinking.has_class("visible"))
                self.assertIsNotNone(app._active_turn_task)
                self.assertIn("Operation continues in background", _timeline_text(app))
                release.set()
                await task
                await pilot.pause()

    async def test_session_switch_is_blocked_while_turn_is_running(self):
        from ergon_studio.runtime import DeliveryAuditDecision, OrchestratorTurnDecision

        _, runtime, app = _make_env()
        release = asyncio.Event()
        orchestrator_agent = GateStreamingAgent(["still working"], release_event=release)

        def build_agent(_runtime, agent_id: str):
            if agent_id == "orchestrator":
                return orchestrator_agent
            return FakeAgent(f"{agent_id} ready")

        async with app.run_test() as pilot:
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=build_agent,
            ), patch.object(
                type(runtime),
                "_decide_orchestrator_turn",
                autospec=True,
                return_value=OrchestratorTurnDecision(mode="act", request=""),
            ), patch.object(
                type(runtime),
                "_audit_orchestrator_delivery_turn",
                autospec=True,
                return_value=DeliveryAuditDecision(
                    deliverable_expected=False,
                    reconsider=False,
                    reason="",
                ),
            ):
                task = asyncio.create_task(app._send_to_orchestrator("hello"))
                for _ in range(20):
                    await asyncio.sleep(0.02)
                    await pilot.pause()
                    if app._active_turn_task is not None and orchestrator_agent.started.is_set():
                        break
                original_session_id = runtime.main_session_id
                inp = app.query_one("#composer-input", ComposerTextArea)
                app.set_focus(inp)
                inp.value = "/new-session Parallel lane"
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.runtime.main_session_id, original_session_id)
                self.assertIn("Finish or wait for running orchestrator work", _timeline_text(app))
                release.set()
                await task
                await pilot.pause()

    async def test_input_always_targets_orchestrator(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-arch",
            kind="agent_direct",
            created_at=1000,
            assigned_agent_id="architect",
            summary="design",
        )
        async with app.run_test():
            inp = app.query_one("#composer-input", ComposerTextArea)
            self.assertIn("orchestrator", str(inp.placeholder))

    async def test_text_change_handler_ignores_unmounted_composer_events(self):
        _, _, app = _make_env()
        app.on_text_area_changed(
            SimpleNamespace(text_area=SimpleNamespace(id="composer-input", text="/he"))
        )

    async def test_clear_command_keeps_notice_and_hides_old_messages(self):
        _, runtime, app = _make_env()
        runtime.append_message_to_main_thread(
            message_id="msg-1",
            sender="user",
            kind="text",
            body="old message",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/clear"
            await pilot.press("enter")
            await pilot.pause()
            text = _timeline_text(app)
            self.assertNotIn("old message", text)
            self.assertIn("Conversation cleared", text)


class TestApprovals(IsolatedAsyncioTestCase):
    async def test_pending_approval_appears_in_timeline(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1",
            requester="coder",
            action="write_file",
            risk_class="moderate",
            reason="update readme",
            created_at=1000,
        )
        async with app.run_test():
            text = _timeline_text(app)
            self.assertIn("Approval Required", text)
            self.assertIn("write_file", text)

    async def test_ctrl_y_approves_pending_approval(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1",
            requester="coder",
            action="write_file",
            risk_class="moderate",
            reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            app.action_approve_pending()
            await pilot.pause()
            approvals = runtime.list_approvals()
            self.assertEqual(approvals[0].status, "approved")
            text = _timeline_text(app)
            self.assertIn("Approved", text)

    async def test_ctrl_r_rejects_pending_approval(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1",
            requester="coder",
            action="write_file",
            risk_class="moderate",
            reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            app.action_reject_pending()
            await pilot.pause()
            approvals = runtime.list_approvals()
            self.assertEqual(approvals[0].status, "rejected")
            text = _timeline_text(app)
            self.assertIn("Rejected", text)


class TestSlashCommands(IsolatedAsyncioTestCase):
    async def test_help_shows_commands(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            text = _timeline_text(app)
            self.assertIn("/session", text)
            self.assertIn("/sessions", text)
            self.assertIn("/config", text)

    async def test_session_shows_current_session(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/session"
            await pilot.press("enter")
            await pilot.pause()
            text = _timeline_text(app)
            self.assertIn(runtime.current_session().title, text)
            self.assertIn(runtime.main_session_id, text)

    async def test_sessions_lists_project_sessions(self):
        _, runtime, app = _make_env()
        load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause()
            text = _timeline_text(app)
            self.assertIn(runtime.current_session().title, text)
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
            text = _timeline_text(app)
            self.assertIn("Switched", text)
            self.assertIn("Parallel lane", text)

    async def test_rename_session_updates_current_session_title(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/rename-session Focus lane"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(app.runtime.current_session().title, "Focus lane")
            text = _timeline_text(app)
            self.assertIn("Renamed", text)
            self.assertIn("Focus lane", text)

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
            text = _timeline_text(app)
            self.assertIn("main session message", text)

    async def test_switch_session_without_id_opens_picker(self):
        _, runtime, app = _make_env()
        load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/switch-session"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, SessionPickerScreen)

    async def test_session_picker_can_switch_sessions(self):
        _, runtime, app = _make_env()
        second = load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        async with app.run_test() as pilot:
            app._open_session_picker()
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, SessionPickerScreen)
            screen.dismiss(second.main_session_id)
            await pilot.pause()
            self.assertEqual(app.runtime.main_session_id, second.main_session_id)
            text = _timeline_text(app)
            self.assertIn("Parallel lane", text)

    async def test_archive_current_session_switches_to_fresh_session(self):
        _, runtime, app = _make_env()
        archived_session_id = runtime.main_session_id
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/archive-session"
            await pilot.press("enter")
            await pilot.pause()
            self.assertNotEqual(app.runtime.main_session_id, archived_session_id)
            self.assertTrue(app.runtime.main_session_id.startswith("session-"))
            text = _timeline_text(app)
            self.assertIn("Archived", text)
            self.assertIn("Switched", text)

    async def test_workflows_lists_inline(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/workflows"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Workflow Definitions")

    async def test_runs_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.workflow_store.create_workflow_run(
            session_id=runtime.main_session_id,
            workflow_run_id="run-1",
            workflow_id="standard-build",
            state="running",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/runs"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Workflow Runs")

    async def test_threads_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.create_thread(
            thread_id="t-coder",
            kind="agent_direct",
            created_at=1000,
            assigned_agent_id="coder",
            summary="implement auth",
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/threads"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Threads")

    async def test_tasks_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.create_task(
            task_id="task-1",
            title="Build feature",
            state="in_progress",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/tasks"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Tasks")

    async def test_artifacts_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.create_artifact(
            artifact_id="artifact-1",
            kind="report",
            title="Workflow report",
            content="Completed successfully.",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/artifacts"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Artifacts")

    async def test_workflow_selects_workflow(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            workflow_ids = runtime.list_workflow_ids()
            if workflow_ids:
                inp = app.query_one("#composer-input", ComposerTextArea)
                app.set_focus(inp)
                inp.value = f"/workflow {workflow_ids[0]}"
                await pilot.press("enter")
                await pilot.pause()
                self.assertEqual(app.selected_workflow_id, workflow_ids[0])

    async def test_agent_opens_internal_workroom_and_keeps_composer_on_orchestrator(self):
        _, runtime, app = _make_env()
        async with app.run_test() as pilot:
            agent_ids = runtime.list_agent_ids()
            if agent_ids:
                inp = app.query_one("#composer-input", ComposerTextArea)
                app.set_focus(inp)
                inp.value = f"/agent {agent_ids[0]}"
                await pilot.press("enter")
                await pilot.pause()
                threads = [thread for thread in runtime.list_threads() if thread.id != runtime.main_thread_id]
                self.assertTrue(threads)
                text = _timeline_text(app)
                self.assertIn("Workroom opened", text)
                self.assertIn("orchestrator", str(inp.placeholder))

    async def test_memory_opens_inspector(self):
        _, runtime, app = _make_env()
        runtime.add_memory_fact(
            fact_id="fact-1",
            scope="project",
            kind="decision",
            content="Use migrations for schema changes.",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/memory"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Memory Facts")

    async def test_approvals_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.request_approval(
            approval_id="appr-1",
            requester="coder",
            action="write_file",
            risk_class="moderate",
            reason="update readme",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/approvals"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Approvals")

    async def test_events_open_inspector(self):
        _, runtime, app = _make_env()
        runtime.append_event(
            kind="note",
            summary="Something happened.",
            created_at=1000,
        )
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/events"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, InspectorScreen)
            self.assertEqual(app.screen.title, "Events")

    async def test_unknown_command_shows_error(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/nonexistent"
            await pilot.press("enter")
            await pilot.pause()
            text = _timeline_text(app)
            self.assertIn("Unknown command", text)

    async def test_slash_shows_command_list(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/"
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertTrue(cmd_list.has_class("visible"))
            self.assertGreater(cmd_list.option_count, 0)

    async def test_slash_filters_as_you_type(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/con"
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertTrue(cmd_list.has_class("visible"))
            self.assertEqual(cmd_list.option_count, 2)

    async def test_slash_list_hides_on_space(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/workflow "
            await pilot.pause()
            cmd_list = app.query_one("#slash-commands", OptionList)
            self.assertFalse(cmd_list.has_class("visible"))


class TestStatusBar(IsolatedAsyncioTestCase):
    async def test_status_bar_shows_agents(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            bar.refresh_from_runtime()
            await pilot.pause()
            self.assertIsNotNone(bar)

    async def test_status_bar_set_agent_state(self):
        _, _, app = _make_env()
        async with app.run_test():
            bar = app.query_one("#agent-status-bar", AgentStatusBar)
            bar.set_agent_state("coder", "active")
            self.assertEqual(bar._agent_states["coder"], "active")


class TestEditorModals(IsolatedAsyncioTestCase):
    async def test_open_session_picker_binding(self):
        _, runtime, app = _make_env()
        load_runtime(
            project_root=runtime.paths.project_root,
            home_dir=runtime.paths.home_dir,
            create_session=True,
            session_title="Parallel lane",
        )
        async with app.run_test() as pilot:
            await pilot.press("ctrl+o")
            await pilot.pause()
            self.assertIsInstance(app.screen, SessionPickerScreen)

    async def test_edit_global_config_opens_wizard(self):
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        _, _, app = _make_env()
        async with app.run_test() as pilot:
            app.action_edit_global_config()
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfigWizardScreen)

    async def test_slash_config_opens_wizard(self):
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        _, _, app = _make_env()
        async with app.run_test() as pilot:
            inp = app.query_one("#composer-input", ComposerTextArea)
            app.set_focus(inp)
            inp.value = "/config"
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfigWizardScreen)

    async def test_edit_orchestrator_definition(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            app.action_edit_orchestrator_definition()
            await pilot.pause()
            self.assertIsInstance(app.screen, DefinitionEditorScreen)

    async def test_run_workspace_command(self):
        _, _, app = _make_env()
        async with app.run_test() as pilot:
            app.action_run_workspace_command()
            await pilot.pause()
            self.assertIsInstance(app.screen, DefinitionEditorScreen)


class TestConfigWizard(IsolatedAsyncioTestCase):
    async def test_provider_editor_saves_provider(self):
        from ergon_studio.tui.config_wizard import ProviderEditorScreen

        _, _, app = _make_env()
        async with app.run_test() as pilot:
            result_holder: list[dict | None] = []

            screen = ProviderEditorScreen(
                name="local",
                base_url="http://localhost:11434/v1",
                model="qwen3:8b",
            )
            app.push_screen(screen, lambda result: result_holder.append(result))
            await pilot.pause()

            name_input = screen.query_one("#name-input", Input)
            self.assertEqual(name_input.value, "local")

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

        _, _, app = _make_env()
        async with app.run_test() as pilot:
            result_holder: list[dict | None] = []
            screen = ProviderEditorScreen()
            app.push_screen(screen, lambda result: result_holder.append(result))
            await pilot.pause()
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

            screen._on_provider_saved(
                {
                    "name": "local",
                    "type": "openai_chat",
                    "model": "test-model",
                    "base_url": "http://localhost:8080/v1",
                }
            )
            await pilot.pause()

            config = json.loads(runtime.read_global_config_text())
            self.assertIn("local", config["providers"])
            assignments = config.get("role_assignments", {})
            self.assertTrue(assignments)
            for provider in assignments.values():
                self.assertEqual(provider, "local")
