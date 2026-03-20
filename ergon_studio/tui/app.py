from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from uuid import uuid4

from textual.app import App, ComposeResult, ScreenStackError
from textual.css.query import NoMatches
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static, TextArea

from ergon_studio.live_runtime import LiveRuntimeEvent, LiveRuntimeSubscription
from ergon_studio.provider_health import (
    AgentReadinessResult,
    ProviderHealthResult,
    assess_agent_readiness,
    probe_all_providers,
)
from ergon_studio.runtime import RuntimeContext, load_runtime
from ergon_studio.storage.models import MessageRecord
from ergon_studio.tui.inspectors import (
    InspectorScreen,
    build_approval_entries,
    build_artifact_entries,
    build_event_entries,
    build_memory_entries,
    build_task_entries,
    build_team_entries,
    build_thread_entry,
    build_thread_entries,
    build_workflow_definition_entries,
    build_workflow_run_entries,
)
from ergon_studio.tui.timeline_builder import build_session_timeline
from ergon_studio.tui.timeline_models import NoticeItem
from ergon_studio.tui.timeline_widgets import TimelineView, TimelineWorkroomSegmentWidget
from ergon_studio.tui.widgets import AgentStatusBar, ComposerTextArea, InfoBar, ThinkingIndicator

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/clear", "Clear conversation, keep config"),
    ("/compact", "Summarize conversation to save context"),
    ("/context", "Show token/context usage"),
    ("/main", "Return the composer to the orchestrator"),
    ("/session", "Show the current session"),
    ("/sessions", "List project sessions"),
    ("/new-session", "Create and switch to a new session"),
    ("/rename-session", "Rename the current session"),
    ("/archive-session", "Archive a session"),
    ("/switch-session", "Switch to an existing session"),
    ("/config", "Open configuration wizard"),
    ("/model", "Switch model for a role"),
    ("/status", "Check provider health and agent readiness"),
    ("/doctor", "Diagnose configuration issues"),
    ("/team", "Show agent roster with status"),
    ("/workflows", "List workflow definitions"),
    ("/runs", "Inspect workflow runs"),
    ("/workflow", "Select a workflow by name"),
    ("/agent", "Open a direct thread with an agent"),
    ("/tasks", "Inspect session tasks"),
    ("/artifacts", "Inspect session artifacts"),
    ("/memory", "Show memory facts"),
    ("/threads", "List all threads"),
    ("/approvals", "Show approval history"),
    ("/events", "Show recent activity"),
    ("/init", "Initialize project configuration"),
]


@dataclass
class ComposeTarget:
    kind: str
    label: str
    thread_id: str | None = None
    agent_id: str | None = None


@dataclass
class PendingTurn:
    user_message: MessageRecord
    body: str
    created_at: int
    target: ComposeTarget
    completion: asyncio.Future[None]


@dataclass(frozen=True)
class SlashSuggestion:
    label: str
    replacement: str
    auto_submit: bool = False


class DefinitionEditorScreen(ModalScreen[None]):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("escape", "cancel", "Cancel"),
    ]
    CSS = """
    #definition-editor-container {
      width: 90%;
      height: 90%;
      border: round $accent;
      background: transparent;
      padding: 1;
    }

    #definition-editor {
      height: 1fr;
      margin: 1 0;
    }

    #definition-error {
      color: $error;
      height: auto;
    }
    """

    def __init__(self, *, title: str, initial_text: str, on_save, language: str = "markdown") -> None:
        super().__init__()
        self.title = title
        self.initial_text = initial_text
        self.on_save = on_save
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="definition-editor-container"):
            yield Static(f"[b]{self.title}[/b]\n`Ctrl+S` to save, `Esc` to cancel.")
            yield TextArea(self.initial_text, id="definition-editor", language=self.language)
            yield Static("", id="definition-error")

    def action_cancel(self) -> None:
        self.dismiss()

    def action_save(self) -> None:
        editor = self.query_one("#definition-editor", TextArea)
        error = self.query_one("#definition-error", Static)
        try:
            self.on_save(editor.text)
        except ValueError as exc:
            error.update(str(exc))
            return
        self.dismiss()


class SessionPickerScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    CSS = """
    #session-picker-container {
      width: 70%;
      height: auto;
      max-height: 80%;
      border: round $accent;
      background: transparent;
      padding: 1;
    }

    #session-picker-options {
      height: auto;
      max-height: 20;
      margin: 1 0 0 0;
    }
    """

    def __init__(self, *, runtime: RuntimeContext, sessions, current_session_id: str) -> None:
        super().__init__()
        self.runtime = runtime
        self.sessions = list(sessions)
        self.current_session_id = current_session_id

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-container"):
            yield Static("[b]Switch Session[/b]\nSelect a session and press Enter.")
            options = []
            for session in self.sessions:
                marker = "•" if session.id == self.current_session_id else " "
                preview = self.runtime.session_preview(session.id)
                options.append(
                    f"{marker} {session.title}  [dim]{preview} · {session.id}[/dim]"
                )
            yield OptionList(*options, id="session-picker-options")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "session-picker-options":
            return
        index = event.option_list.highlighted
        if index is None or index < 0 or index >= len(self.sessions):
            self.dismiss(None)
            return
        self.dismiss(self.sessions[index].id)


class ErgonStudioApp(App[None]):
    TITLE = "ergon.studio"
    BINDINGS = [
        ("enter", "activate_focused_workroom", "Use Workroom"),
        ("ctrl+i", "inspect_focused_workroom", "Inspect Workroom"),
        ("ctrl+o", "open_session_picker", "Sessions"),
        ("ctrl+y", "approve_pending", "Approve"),
        ("ctrl+r", "reject_pending", "Reject"),
        ("ctrl+g", "edit_global_config", "Edit Config"),
        ("ctrl+e", "edit_orchestrator_definition", "Edit Orchestrator"),
        ("ctrl+x", "run_workspace_command", "Run Command"),
        ("ctrl+c", "quit", "Quit"),
        ("shift+tab", "cycle_permission_mode", "Permission Mode"),
        ("ctrl+b", "background_current", "Background"),
    ]
    CSS = """
    Screen {
      layout: vertical;
    }

    #main-timeline {
      height: 1fr;
      background: transparent;
    }

    #composer-input {
      height: auto;
      max-height: 10;
      min-height: 3;
      margin: 0 1;
      padding: 0 1;
      border: none;
    }

    #slash-commands {
      height: auto;
      max-height: 10;
      margin: 0 1;
      display: none;
      background: transparent;
      border: round $accent;
    }

    #slash-commands.visible {
      display: block;
    }
    """

    def __init__(
        self,
        runtime: RuntimeContext,
        *,
        open_session_picker_on_mount: bool = False,
        open_config_wizard_on_mount: bool = False,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.open_session_picker_on_mount = open_session_picker_on_mount
        self.open_config_wizard_on_mount = open_config_wizard_on_mount
        self.selected_workflow_id = self._default_workflow_id()
        self.selected_workflow_run_id: str | None = None
        self._timeline_notices: list[NoticeItem] = []
        self._hidden_main_message_ids: set[str] = set()
        self._timeline_cutoff_created_at: int | None = None
        self._time_cursor = int(time.time())
        self._last_escape_time: float = 0.0
        self._permission_mode: str = "default"  # default, auto-approve, plan
        self._compacting: bool = False
        self._active_turn: PendingTurn | None = None
        self._active_turn_task: asyncio.Task[None] | None = None
        self._queued_turns: list[PendingTurn] = []
        self._turn_backgrounded: bool = False
        self._live_subscription: LiveRuntimeSubscription | None = None
        self._live_subscription_task: asyncio.Task[None] | None = None
        self._live_refresh_task: asyncio.Task[None] | None = None
        self._compose_target = self._orchestrator_target()
        self._slash_suggestions: list[SlashSuggestion] = []

    def _default_workflow_id(self) -> str | None:
        summaries = self.runtime.list_workflow_summaries()
        for summary in summaries:
            hints = tuple(summary.get("selection_hints", ()))
            if "staged_delivery" in hints:
                return str(summary["id"])
        for summary in summaries:
            if bool(summary.get("delivery_candidate")):
                return str(summary["id"])
        workflow_ids = self.runtime.list_workflow_ids()
        if workflow_ids:
            return workflow_ids[0]
        return None

    def _orchestrator_target(self) -> ComposeTarget:
        return ComposeTarget(
            kind="orchestrator",
            label="orchestrator",
            agent_id="orchestrator",
        )

    def _set_compose_target(self, target: ComposeTarget) -> None:
        self._compose_target = target
        self._update_composer_placeholder()
        self._refresh_info()

    def _set_compose_target_to_orchestrator(self) -> None:
        self._set_compose_target(self._orchestrator_target())

    def _set_compose_target_to_thread(self, thread_id: str) -> None:
        thread = self.runtime.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        label = thread.assigned_agent_id or thread.summary or thread.id
        self._set_compose_target(
            ComposeTarget(
                kind="thread",
                label=label,
                thread_id=thread.id,
                agent_id=thread.assigned_agent_id,
            )
        )

    def _composer_placeholder(self) -> str:
        if self._compose_target.thread_id is None:
            return "❯ Message the orchestrator..."
        return f"❯ Message {self._compose_target.label} in this workroom..."

    def _update_composer_placeholder(self) -> None:
        if not self.is_mounted:
            return
        try:
            composer = self.query_one("#composer-input", ComposerTextArea)
        except (NoMatches, ScreenStackError):
            return
        composer.placeholder = self._composer_placeholder()

    def _slash_command_suggestions(self, value: str) -> list[SlashSuggestion]:
        text = value.strip()
        if not text.startswith("/"):
            return []

        if " " not in text:
            prefix = text.lower()
            return [
                SlashSuggestion(
                    label=f"{cmd}  [dim]{desc}[/dim]",
                    replacement=cmd,
                    auto_submit=cmd not in {
                        "/workflow",
                        "/agent",
                        "/new-session",
                        "/rename-session",
                        "/archive-session",
                        "/switch-session",
                    },
                )
                for cmd, desc in SLASH_COMMANDS
                if cmd.startswith(prefix)
            ]

        command, raw_arg = text.split(" ", 1)
        arg_prefix = raw_arg.strip().lower()
        suggestions: list[SlashSuggestion] = []

        if command == "/agent":
            for agent_id in self.runtime.list_agent_ids():
                if arg_prefix and not agent_id.lower().startswith(arg_prefix):
                    continue
                suggestions.append(
                    SlashSuggestion(
                        label=f"{command} {agent_id}  [dim]open a direct thread[/dim]",
                        replacement=f"{command} {agent_id}",
                        auto_submit=True,
                    )
                )
        elif command == "/workflow":
            for workflow_id in self.runtime.list_workflow_ids():
                if arg_prefix and not workflow_id.lower().startswith(arg_prefix):
                    continue
                suggestions.append(
                    SlashSuggestion(
                        label=f"{command} {workflow_id}  [dim]select workflow[/dim]",
                        replacement=f"{command} {workflow_id}",
                        auto_submit=True,
                    )
                )
        elif command in {"/switch-session", "/archive-session"}:
            for session in self.runtime.list_sessions(include_archived=command == "/switch-session"):
                haystack = f"{session.id} {session.title}".lower()
                if arg_prefix and arg_prefix not in haystack:
                    continue
                suggestions.append(
                    SlashSuggestion(
                        label=f"{command} {session.id}  [dim]{session.title}[/dim]",
                        replacement=f"{command} {session.id}",
                        auto_submit=True,
                    )
                )

        return suggestions

    def _target_notice_label(self, target: ComposeTarget) -> str:
        if target.thread_id is None:
            return "the orchestrator"
        return target.label

    def _thinking_label(self, turn: PendingTurn) -> str:
        if turn.target.thread_id is None:
            return "Thinking"
        return f"Working with {turn.target.label}"

    async def _provider_health(self) -> list[ProviderHealthResult]:
        return await asyncio.to_thread(
            probe_all_providers,
            self.runtime.registry.config,
            timeout=5,
        )

    def _agent_readiness(
        self,
        provider_health: list[ProviderHealthResult],
    ) -> list[AgentReadinessResult]:
        return assess_agent_readiness(
            self.runtime.list_agent_ids(),
            assigned_provider_name=self.runtime.assigned_provider_name,
            agent_unavailable_reason=self.runtime.agent_unavailable_reason,
            agent_status_summary=self.runtime.agent_status_summary,
            provider_health=provider_health,
            provider_details=self.runtime.provider_details,
        )

    async def _status_notice_body(self) -> str:
        providers = self.runtime.list_provider_ids()
        provider_lines: list[str]
        if not providers:
            health_results: list[ProviderHealthResult] = []
            provider_lines = ["[bold]Provider status:[/bold]", "  [red]No providers configured.[/red] Use /config"]
        else:
            health_results = await self._provider_health()
            if not health_results:
                provider_lines = ["[bold]Provider status:[/bold]", "  [red]No valid provider definitions found.[/red]"]
            else:
                provider_lines = ["[bold]Provider status:[/bold]"]
        for result in health_results:
            if result.ok:
                suffix = f"{result.model} @ {result.base_url}"
                if result.model_count:
                    suffix += f" [dim]({result.model_count} models)[/dim]"
                provider_lines.append(f"  [green]●[/green] {result.name}: {suffix}")
            else:
                provider_lines.append(
                    f"  [red]●[/red] {result.name}: {result.model or 'unknown-model'} @ {result.base_url or '(missing url)'}"
                )
                if result.error:
                    provider_lines.append(f"    [dim]{result.error}[/dim]")

        readiness = self._agent_readiness(health_results)
        readiness_lines = ["", "[bold]Agent readiness:[/bold]"]
        if not readiness:
            readiness_lines.append("  [yellow]No agents defined.[/yellow]")
        for result in readiness:
            if result.ok:
                suffix = result.summary
                readiness_lines.append(f"  [green]●[/green] {result.name}: {suffix}")
                continue
            if result.provider_name is None:
                readiness_lines.append(f"  [red]●[/red] {result.name}: not configured")
            else:
                readiness_lines.append(
                    f"  [red]●[/red] {result.name}: {result.summary}"
                )
            if result.error:
                readiness_lines.append(f"    [dim]{result.error}[/dim]")

        return "\n".join(provider_lines + readiness_lines)

    async def _doctor_notice(self) -> tuple[str, str, str]:
        issues: list[str] = []
        suggestions: list[str] = []
        health_results = await self._provider_health()

        if not self.runtime.list_provider_ids():
            issues.append("[red]✗[/red] No providers configured")
            suggestions.append("Use /config to add at least one OpenAI-compatible provider.")

        for result in health_results:
            if not result.ok:
                issues.append(f"[red]✗[/red] Provider {result.name}: {result.error or 'health check failed'}")
                suggestions.append(f"Check {result.name} at {result.base_url or '(missing url)'} and verify its model list.")

        for agent_id in self.runtime.list_agent_ids():
            reason = self.runtime.agent_unavailable_reason(agent_id)
            if reason is None:
                continue
            issues.append(f"[red]✗[/red] {agent_id}: {reason}")
            if "no provider assigned" in reason:
                suggestions.append(f"Assign a provider to {agent_id} in /config.")
            elif "not defined" in reason:
                suggestions.append(f"Fix the provider assignment for {agent_id} in /config.")

        if not issues:
            return (
                "Doctor",
                "All checks passed.\n\n[green]Providers are reachable and every agent can be constructed.[/green]",
                "success",
            )

        lines = ["[bold]Issues found:[/bold]", *issues]
        if suggestions:
            deduped = list(dict.fromkeys(suggestions))
            lines.extend(["", "[bold]Suggested fixes[/bold]"])
            lines.extend(f"- {suggestion}" for suggestion in deduped)
        return ("Doctor", "\n".join(lines), "info")

    def _maybe_surface_unavailable_target(self, target: ComposeTarget) -> None:
        if target.agent_id is None:
            return
        reason = self.runtime.agent_unavailable_reason(target.agent_id)
        if reason is None:
            return
        if target.thread_id is None:
            self._add_notice(
                f"{reason}\nUse /config to add a provider and role assignment for the orchestrator.",
                level="error",
                title="Setup needed",
            )
            return
        self._add_notice(
            f"{target.label} is unavailable: {reason}\nUse /config to finish the team setup, or /main to return to the orchestrator.",
            level="error",
            title="Workroom unavailable",
        )

    def compose(self) -> ComposeResult:
        yield AgentStatusBar(self.runtime, id="agent-status-bar")
        yield TimelineView(id="main-timeline")
        yield ThinkingIndicator(id="thinking")
        yield OptionList(id="slash-commands")
        yield ComposerTextArea(placeholder="❯ Message the orchestrator...", id="composer-input")
        yield InfoBar(self.runtime, id="info-bar")

    def on_mount(self) -> None:
        self._update_composer_placeholder()
        self.set_focus(self.query_one("#composer-input", ComposerTextArea))
        self._start_live_subscription()
        self._refresh_timeline()
        self._refresh_info()
        if self.open_session_picker_on_mount:
            self._open_session_picker()
        elif self.open_config_wizard_on_mount:
            self._open_config_wizard()

    def on_unmount(self) -> None:
        self._stop_live_subscription()
        if self._active_turn_task is not None:
            self._active_turn_task.cancel()
            self._active_turn_task = None
        if self._active_turn is not None and not self._active_turn.completion.done():
            self._active_turn.completion.cancel()
            self._active_turn = None
        for turn in self._queued_turns:
            if not turn.completion.done():
                turn.completion.cancel()
        self._queued_turns = []

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer-input":
            return
        if not self.is_mounted:
            return
        value = event.text_area.text
        try:
            cmd_list = self.query_one("#slash-commands", OptionList)
        except (NoMatches, ScreenStackError):
            return
        self._slash_suggestions = self._slash_command_suggestions(value)
        cmd_list.clear_options()
        if self._slash_suggestions:
            for suggestion in self._slash_suggestions:
                cmd_list.add_option(suggestion.label)
            cmd_list.add_class("visible")
            cmd_list.highlighted = 0
        else:
            cmd_list.remove_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "slash-commands":
            return
        if not self.is_mounted:
            return
        index = event.option_list.highlighted
        if index is None or index < 0 or index >= len(self._slash_suggestions):
            return
        suggestion = self._slash_suggestions[index]
        try:
            inp = self.query_one("#composer-input", ComposerTextArea)
        except (NoMatches, ScreenStackError):
            return
        inp.value = suggestion.replacement
        event.option_list.remove_class("visible")
        self.set_focus(inp)
        if suggestion.auto_submit:
            inp.post_message(ComposerTextArea.Submitted(inp, suggestion.replacement))

    def on_key(self, event) -> None:
        """Route arrow keys to command list; double-escape to rewind."""
        # Handle double-escape
        if event.key == "escape":
            now = time.monotonic()
            composer = self.query_one("#composer-input", ComposerTextArea)
            if composer.text:
                # Composer handles its own escape (clear text)
                pass
            elif now - self._last_escape_time < 0.5:
                self._last_escape_time = 0.0
                self._rewind_last_exchange()
                event.prevent_default()
                return
            else:
                self._last_escape_time = now

        cmd_list = self.query_one("#slash-commands", OptionList)
        if not cmd_list.has_class("visible"):
            return
        inp = self.query_one("#composer-input", ComposerTextArea)
        if self.focused is inp and event.key in ("up", "down"):
            self.set_focus(cmd_list)
            event.prevent_default()
        elif self.focused is cmd_list and event.key == "escape":
            cmd_list.remove_class("visible")
            self.set_focus(inp)
            event.prevent_default()

    async def on_composer_text_area_submitted(self, event: ComposerTextArea.Submitted) -> None:
        # Hide command list on submit
        self.query_one("#slash-commands", OptionList).remove_class("visible")

        text = event.value.strip()
        if not text:
            event.text_area.value = ""
            return

        if text.startswith("/"):
            event.text_area.disabled = True
            try:
                await self._handle_slash_command(text)
                event.text_area.value = ""
            finally:
                event.text_area.disabled = False
                self.set_focus(event.text_area)
            return

        self._submit_turn(text)
        event.text_area.value = ""
        self.set_focus(event.text_area)

    async def _send_to_orchestrator(self, body: str) -> None:
        turn = self._submit_orchestrator_turn(body)
        await turn.completion

    async def _send_to_agent_thread(self, *, thread_id: str, body: str) -> None:
        thread = self.runtime.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"unknown thread: {thread_id}")
        turn = self._submit_turn_for_target(
            body,
            ComposeTarget(
                kind="thread",
                label=thread.assigned_agent_id or thread_id,
                thread_id=thread_id,
                agent_id=thread.assigned_agent_id,
            ),
        )
        await turn.completion

    def _submit_turn(self, body: str) -> PendingTurn:
        return self._submit_turn_for_target(body, self._compose_target)

    def _submit_orchestrator_turn(self, body: str) -> PendingTurn:
        return self._submit_turn_for_target(body, self._orchestrator_target())

    def _submit_turn_for_target(self, body: str, target: ComposeTarget) -> PendingTurn:
        created_at = self._next_timestamp()
        if target.thread_id is None:
            user_message = self.runtime.record_user_message_to_main_thread(
                body=body,
                created_at=created_at,
            )
        else:
            user_message = self.runtime.record_user_message_to_thread(
                thread_id=target.thread_id,
                body=body,
                created_at=created_at,
            )
        turn = PendingTurn(
            user_message=user_message,
            body=body,
            created_at=created_at,
            target=target,
            completion=asyncio.get_running_loop().create_future(),
        )
        self._refresh_timeline()
        if self._active_turn_task is not None and not self._active_turn_task.done():
            self._queued_turns.append(turn)
            self._add_notice(
                f"Queued message for {self._target_notice_label(target)}. {len(self._queued_turns)} waiting.",
                level="info",
            )
            self._refresh_timeline()
            self._refresh_info()
            return turn
        self._start_turn(turn)
        return turn

    def _start_orchestrator_turn(self, turn: PendingTurn) -> None:
        self._start_turn(turn)

    def _start_turn(self, turn: PendingTurn) -> None:
        thinking = self.query_one("#thinking", ThinkingIndicator)
        self._turn_backgrounded = False
        thinking.show(self._thinking_label(turn))
        self._active_turn = turn
        self._active_turn_task = asyncio.create_task(
            self._run_turn(
                turn=turn,
            )
        )
        self._refresh_timeline()
        self._refresh_info()

    async def _run_turn(
        self,
        *,
        turn: PendingTurn,
    ) -> None:
        thinking = self.query_one("#thinking", ThinkingIndicator)
        current_task = asyncio.current_task()
        try:
            if turn.target.thread_id is None:
                stream = self.runtime.stream_user_message_to_orchestrator(
                    body=turn.body,
                    created_at=turn.created_at,
                    user_message=turn.user_message,
                )
            else:
                stream = self.runtime.stream_user_message_to_agent_thread(
                    thread_id=turn.target.thread_id,
                    body=turn.body,
                    created_at=turn.created_at,
                    user_message=turn.user_message,
                )
            self._refresh_timeline()
            async for _event in stream:
                pass
            _, reply_message = await stream.get_final_response()
        except asyncio.CancelledError:
            thinking.hide()
            self._refresh_timeline()
            if not turn.completion.done():
                turn.completion.cancel()
            raise
        except Exception as exc:
            thinking.hide()
            self._add_notice(f"Error: {exc}", level="error", title="Send failed")
            self._refresh_timeline()
            if not turn.completion.done():
                turn.completion.set_result(None)
        else:
            thinking.hide()
            if reply_message is None and turn.target.thread_id is not None:
                self._maybe_surface_unavailable_target(turn.target)
            self._refresh_timeline()
            await self._check_auto_compaction()
            if not turn.completion.done():
                turn.completion.set_result(None)
        finally:
            if self._active_turn_task is current_task:
                self._active_turn_task = None
            if self._active_turn is turn:
                self._active_turn = None
            self._turn_backgrounded = False
            self._refresh_info()
        self._start_next_queued_turn()

    def _start_next_queued_turn(self) -> None:
        if self._active_turn_task is not None and not self._active_turn_task.done():
            return
        if not self._queued_turns:
            return
        turn = self._queued_turns.pop(0)
        self._add_notice("Continuing with the next queued message.", level="info")
        self._start_orchestrator_turn(turn)

    async def _handle_slash_command(self, text: str) -> None:
        parts = text.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            lines = ["[bold]Available commands:[/bold]"]
            for cmd, desc in SLASH_COMMANDS:
                lines.append(f"  {cmd:20s} {desc}")
            self._add_notice("\n".join(lines), title="Help")
        elif command == "/clear":
            self._timeline_cutoff_created_at = self._next_timestamp()
            self._timeline_notices = []
            self._hidden_main_message_ids = set()
            self._add_notice("Conversation cleared from the current view.", level="info")
        elif command == "/compact":
            focus = args.strip() or None
            self._add_notice("Compacting conversation...", level="info")
            thinking = self.query_one("#thinking", ThinkingIndicator)
            thinking.show("Compacting")
            # Reset circuit breaker for manual compaction
            object.__setattr__(self.runtime, "_compaction_failure_count", 0)
            try:
                summary = await self._compact_conversation(focus)
                thinking.hide()
                if summary:
                    self._add_notice("Context compacted.", level="success")
                else:
                    self._add_notice("Nothing to compact.", level="warning")
            except Exception as exc:
                thinking.hide()
                self._add_notice(f"Compaction failed: {exc}", level="error")
        elif command == "/context":
            messages = self.runtime.list_main_messages()
            total_chars = 0
            for msg in messages:
                total_chars += len(self.runtime.conversation_store.read_message_body(msg))
            tracked = self.runtime.accumulated_tokens()
            ctx_window = self.runtime.context_window_size()
            pct = int(tracked / ctx_window * 100) if ctx_window > 0 else 0
            self._add_notice(
                f"[bold]Context usage:[/bold]\n"
                f"  Messages: {len(messages)}\n"
                f"  Characters: {total_chars:,}\n"
                f"  Tracked tokens: {tracked:,}\n"
                f"  Context window: {ctx_window:,} ({ctx_window // 1024}k)\n"
                f"  Usage: {pct}%\n"
                f"  Auto-compact at: 95%",
                title="Context",
            )
        elif command == "/main":
            self._set_compose_target_to_orchestrator()
            self._add_notice(
                "Composer now targets the orchestrator.",
                level="info",
                title="Main chat",
            )
        elif command == "/session":
            session = self.runtime.current_session()
            if session is None:
                self._add_notice("No active session.", level="info")
            else:
                self._add_notice(
                    f"[bold]Session:[/bold] {session.title}\n"
                    f"[dim]{session.id}[/dim]",
                    title="Session",
                )
        elif command == "/sessions":
            sessions = self.runtime.list_sessions(include_archived=True)
            if not sessions:
                self._add_notice("No sessions yet.", level="info")
            else:
                lines = ["[bold]Sessions:[/bold]"]
                for session in sessions:
                    marker = ">" if session.id == self.runtime.main_session_id else " "
                    archived = " archived" if session.archived_at is not None else ""
                    lines.append(
                        f"  {marker} {session.title} [dim]({session.id})[/dim]{archived}"
                    )
                self._add_notice("\n".join(lines), title="Sessions")
        elif command == "/new-session":
            if self._has_inflight_turns():
                self._add_notice("Finish or wait for running orchestrator work before switching sessions.", level="warning")
                self._refresh_timeline()
                return
            title = args.strip() or None
            new_runtime = load_runtime(
                project_root=self.runtime.paths.project_root,
                home_dir=self.runtime.paths.home_dir,
                create_session=True,
                session_title=title,
            )
            self._replace_runtime(
                new_runtime,
                notice=f"[green]Switched[/green] to session {new_runtime.current_session().title}",
            )
        elif command == "/rename-session":
            title = args.strip()
            if not title:
                self._add_notice("Usage: /rename-session <title>", level="error")
            else:
                renamed = self.runtime.rename_session(
                    session_id=self.runtime.main_session_id,
                    title=title,
                    created_at=self._next_timestamp(),
                )
                self._add_notice(f"Renamed current session to {renamed.title}", level="success")
                self._refresh_info()
        elif command == "/archive-session":
            if self._has_inflight_turns():
                self._add_notice("Finish or wait for running orchestrator work before switching sessions.", level="warning")
                self._refresh_timeline()
                return
            target = args.strip() or self.runtime.main_session_id
            try:
                self.runtime.archive_session(
                    session_id=target,
                    created_at=self._next_timestamp(),
                )
            except ValueError:
                self._add_notice(f"Unknown session: {target}", level="error")
            else:
                if target == self.runtime.main_session_id:
                    new_runtime = load_runtime(
                        project_root=self.runtime.paths.project_root,
                        home_dir=self.runtime.paths.home_dir,
                    )
                    self._replace_runtime(
                        new_runtime,
                        notice=(
                            f"[yellow]Archived[/yellow] session {target}\n"
                            f"[green]Switched[/green] to session {new_runtime.current_session().title}"
                        ),
                    )
                else:
                    self._add_notice(f"Archived session {target}", level="warning")
        elif command == "/switch-session":
            if self._has_inflight_turns():
                self._add_notice("Finish or wait for running orchestrator work before switching sessions.", level="warning")
                self._refresh_timeline()
                return
            target = args.strip()
            if not target:
                self._open_session_picker()
            else:
                try:
                    new_runtime = load_runtime(
                        project_root=self.runtime.paths.project_root,
                        home_dir=self.runtime.paths.home_dir,
                        session_id=target,
                    )
                except ValueError:
                    self._add_notice(f"Unknown session: {target}", level="error")
                else:
                    self._replace_runtime(
                        new_runtime,
                        notice=f"[green]Switched[/green] to session {new_runtime.current_session().title}",
                    )
        elif command == "/config":
            self._open_config_wizard()
        elif command == "/workflows":
            self._open_workflow_definition_inspector()
        elif command == "/runs":
            self._open_workflow_run_inspector()
        elif command == "/workflow":
            if not args:
                self._add_notice("Usage: /workflow <name>", level="error")
            else:
                workflow_id = self.runtime.resolve_workflow_reference(args)
                if workflow_id is None:
                    self._add_notice(f"Unknown workflow: {args}", level="error")
                else:
                    self.selected_workflow_id = workflow_id
                    self._add_notice(f"Selected workflow: {workflow_id}", level="info")
                    self._refresh_info()
        elif command == "/agent":
            if not args:
                self._add_notice("Usage: /agent <name>", level="error")
            elif args in self.runtime.list_agent_ids():
                self._open_agent_thread(args)
            else:
                self._add_notice(f"Unknown agent: {args}", level="error")
        elif command == "/memory":
            self._open_memory_inspector()
        elif command == "/threads":
            self._open_thread_inspector()
        elif command == "/tasks":
            self._open_task_inspector()
        elif command == "/artifacts":
            self._open_artifact_inspector()
        elif command == "/model":
            if not args:
                lines = ["[bold]Model assignments:[/bold]"]
                for agent_id in self.runtime.list_agent_ids():
                    provider = self.runtime.assigned_provider_name(agent_id)
                    if provider:
                        details = self.runtime.provider_details(provider)
                        model = details.get("model", "?") if details else "?"
                        lines.append(f"  {agent_id}: {model} [dim]via {provider}[/dim]")
                    else:
                        lines.append(f"  {agent_id}: [red]not assigned[/red]")
                self._add_notice("\n".join(lines), title="Models")
            else:
                self._add_notice("Use /config to change model assignments.", level="info")
        elif command == "/status":
            thinking = self.query_one("#thinking", ThinkingIndicator)
            thinking.show("Checking providers")
            try:
                body = await self._status_notice_body()
            finally:
                thinking.hide()
            self._add_notice(body, title="Status")
        elif command == "/doctor":
            thinking = self.query_one("#thinking", ThinkingIndicator)
            thinking.show("Running doctor")
            try:
                title, body, level = await self._doctor_notice()
            finally:
                thinking.hide()
            self._add_notice(body, title=title, level=level)
        elif command == "/team":
            self._open_team_inspector()
        elif command == "/approvals":
            self._open_approval_inspector()
        elif command == "/events":
            self._open_event_inspector()
        elif command == "/init":
            self._add_notice(
                "[dim]Project already initialized.\n"
                f"UUID: {self.runtime.paths.project_uuid}\n"
                f"Data: {self.runtime.paths.project_data_dir}[/dim]",
                title="Project",
            )
        else:
            self._add_notice(f"Unknown command: {command}. Type /help for available commands.", level="error")
        self._refresh_timeline()

    def _open_agent_thread(self, agent_id: str) -> None:
        existing_threads = [
            thread
            for thread in self.runtime.list_threads()
            if thread.kind == "agent_direct" and thread.assigned_agent_id == agent_id
        ]
        if existing_threads:
            thread = max(existing_threads, key=lambda item: (item.created_at, item.id))
        else:
            created_at = self._next_timestamp()
            task = self.runtime.create_task(
                task_id=f"task-{uuid4().hex[:8]}",
                title=f"Agent thread: {agent_id}",
                state="in_progress",
                created_at=created_at,
            )
            thread = self.runtime.create_agent_thread(
                agent_id=agent_id,
                created_at=created_at + 1,
                parent_task_id=task.id,
            )
        self._set_compose_target_to_thread(thread.id)
        reason = self.runtime.agent_unavailable_reason(agent_id)
        self._add_notice(
            (
                f"Composer now targets {agent_id}. Your next message will go to this direct thread.\n"
                "Use /main to return to the orchestrator."
            )
            if reason is None
            else (
                f"Composer now targets {agent_id}, but that agent is not ready yet.\n"
                f"{reason}\nUse /config to finish setup, or /main to return."
            ),
            level="info" if reason is None else "warning",
            title="Direct thread",
        )
        self._refresh_timeline()

    def action_approve_pending(self) -> None:
        approval = self._selected_pending_approval()
        if approval is None:
            return
        self.runtime.resolve_approval(
            approval_id=approval.id,
            status="approved",
            created_at=self._next_timestamp(),
        )
        self._add_notice(f"Approved {approval.action} by {approval.requester}", level="success")
        self._refresh_timeline()
        self._refresh_info()

    def action_reject_pending(self) -> None:
        approval = self._selected_pending_approval()
        if approval is None:
            return
        self.runtime.resolve_approval(
            approval_id=approval.id,
            status="rejected",
            created_at=self._next_timestamp(),
        )
        self._add_notice(f"Rejected {approval.action} by {approval.requester}", level="error")
        self._refresh_timeline()
        self._refresh_info()

    def _selected_pending_approval(self):
        pending = self.runtime.list_pending_approvals()
        if not pending:
            return None
        focused = self.focused
        approval_id = getattr(focused, "approval_id", None)
        if isinstance(approval_id, str):
            for approval in pending:
                if approval.id == approval_id:
                    return approval
        if len(pending) == 1:
            return pending[0]
        self._add_notice(
            "Select an approval in the timeline before approving or rejecting when more than one is pending.",
            level="warning",
            title="Select approval",
        )
        self._refresh_timeline()
        return None

    def action_background_current(self) -> None:
        """Move the current thinking operation to background."""
        thinking = self.query_one("#thinking", ThinkingIndicator)
        if self._active_turn_task is None or self._active_turn_task.done():
            self._add_notice("Nothing is running to background.", level="info")
            self._refresh_timeline()
            return
        self._turn_backgrounded = True
        thinking.hide()
        self._add_notice("Operation continues in background. You can keep typing.", level="info")
        self._refresh_timeline()
        self._refresh_info()
        self.set_focus(self.query_one("#composer-input", ComposerTextArea))

    def action_cycle_permission_mode(self) -> None:
        modes = ["default", "auto-approve", "plan"]
        current_idx = modes.index(self._permission_mode) if self._permission_mode in modes else 0
        self._permission_mode = modes[(current_idx + 1) % len(modes)]
        config = self.runtime.registry.config
        if self._permission_mode == "default":
            config["approvals"] = {"default": "ask"}
        elif self._permission_mode == "auto-approve":
            config["approvals"] = {"default": "auto"}
        elif self._permission_mode == "plan":
            config["approvals"] = {"default": "block"}
        mode_labels = {
            "default": "[yellow]Default[/yellow] (ask for approvals)",
            "auto-approve": "[green]Auto-approve[/green] (approve all automatically)",
            "plan": "[red]Plan mode[/red] (block all write operations)",
        }
        self._add_notice(f"[dim]Permission mode:[/dim] {mode_labels[self._permission_mode]}", level="info")
        self._refresh_timeline()
        self._refresh_info()

    def action_edit_global_config(self) -> None:
        self._open_config_wizard()

    def action_open_session_picker(self) -> None:
        if self._has_inflight_turns():
            self._add_notice("Finish or wait for running orchestrator work before switching sessions.", level="warning")
            self._refresh_timeline()
            return
        self._open_session_picker()

    def action_activate_focused_workroom(self) -> None:
        focused = self.focused
        if not isinstance(focused, TimelineWorkroomSegmentWidget):
            return
        if focused.thread_kind == "agent_direct" and focused.assigned_agent_id:
            self._set_compose_target_to_thread(focused.thread_id)
            self._add_notice(
                f"Composer now targets {focused.assigned_agent_id}. Use /main to return to the orchestrator.",
                level="info",
                title="Direct thread",
            )
            self._refresh_timeline()
            self.set_focus(self.query_one("#composer-input", ComposerTextArea))
            return
        self._open_thread_detail(focused.thread_id)

    def action_inspect_focused_workroom(self) -> None:
        focused = self.focused
        if not isinstance(focused, TimelineWorkroomSegmentWidget):
            return
        self._open_thread_detail(focused.thread_id)

    def _open_config_wizard(self) -> None:
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        def on_dismiss(result: None) -> None:
            self._refresh_info()
            self.query_one(AgentStatusBar).refresh_from_runtime()

        self.push_screen(ConfigWizardScreen(self.runtime), on_dismiss)

    def _open_thread_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Threads",
                entries=build_thread_entries(self.runtime),
                empty_message="No internal threads in this session yet.",
            )
        )

    def _open_thread_detail(self, thread_id: str) -> None:
        entry = build_thread_entry(self.runtime, thread_id)
        if entry is None:
            self._add_notice(f"Unknown thread: {thread_id}", level="error")
            self._refresh_timeline()
            return
        self.push_screen(
            InspectorScreen(
                title="Thread",
                entries=[entry],
                empty_message="Thread not found.",
            )
        )

    def _open_task_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Tasks",
                entries=build_task_entries(self.runtime),
                empty_message="No tasks in this session yet.",
            )
        )

    def _open_team_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Team",
                entries=build_team_entries(self.runtime),
                empty_message="No agents loaded.",
            )
        )

    def _open_workflow_run_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Workflow Runs",
                entries=build_workflow_run_entries(self.runtime),
                empty_message="No workflow runs in this session yet.",
            )
        )

    def _open_workflow_definition_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Workflow Definitions",
                entries=build_workflow_definition_entries(self.runtime),
                empty_message="No workflow definitions loaded.",
            )
        )

    def _open_artifact_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Artifacts",
                entries=build_artifact_entries(self.runtime),
                empty_message="No artifacts in this session yet.",
            )
        )

    def _open_memory_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Memory Facts",
                entries=build_memory_entries(self.runtime),
                empty_message="No memory facts available yet.",
            )
        )

    def _open_approval_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Approvals",
                entries=build_approval_entries(self.runtime),
                empty_message="No approvals in this session yet.",
            )
        )

    def _open_event_inspector(self) -> None:
        self.push_screen(
            InspectorScreen(
                title="Events",
                entries=build_event_entries(self.runtime),
                empty_message="No events in this session yet.",
            )
        )

    def _open_session_picker(self) -> None:
        sessions = self.runtime.list_sessions()
        if not sessions:
            self._add_notice("No sessions available.", level="info")
            self._refresh_timeline()
            return

        def on_dismiss(result: str | None) -> None:
            if not result or result == self.runtime.main_session_id:
                return
            new_runtime = load_runtime(
                project_root=self.runtime.paths.project_root,
                home_dir=self.runtime.paths.home_dir,
                session_id=result,
            )
            session = new_runtime.current_session()
            notice = "[green]Switched[/green] session"
            if session is not None:
                notice = f"[green]Switched[/green] to session {session.title}"
            self._replace_runtime(new_runtime, notice=notice)

        self.push_screen(
            SessionPickerScreen(
                runtime=self.runtime,
                sessions=sessions,
                current_session_id=self.runtime.main_session_id,
            ),
            on_dismiss,
        )

    def action_edit_orchestrator_definition(self) -> None:
        initial_text = self.runtime.read_agent_definition_text("orchestrator")
        self.push_screen(
            DefinitionEditorScreen(
                title="Edit Orchestrator Definition",
                initial_text=initial_text,
                on_save=lambda text: self._save_agent_definition("orchestrator", text),
            )
        )

    def action_run_workspace_command(self) -> None:
        self.push_screen(
            DefinitionEditorScreen(
                title="Run Workspace Command",
                initial_text="",
                on_save=self._run_workspace_command_from_editor,
                language="bash",
            )
        )

    def _save_global_config(self, text: str) -> None:
        self.runtime.save_global_config_text(
            text=text, created_at=self._next_timestamp(),
        )
        self._refresh_info()
        self.query_one("#agent-status-bar", AgentStatusBar).refresh_from_runtime()

    def _save_agent_definition(self, agent_id: str, text: str) -> None:
        self.runtime.save_agent_definition_text(
            agent_id=agent_id, text=text, created_at=self._next_timestamp(),
        )
        self.query_one("#agent-status-bar", AgentStatusBar).refresh_from_runtime()

    def _run_workspace_command_from_editor(self, text: str) -> None:
        command = text.strip()
        if not command:
            raise ValueError("Command cannot be empty.")
        result = self.runtime.run_workspace_command(
            command, created_at=self._next_timestamp(), agent_id="user",
        )
        approval_id = result.get("approval_id")
        if isinstance(approval_id, str):
            self._add_notice(f"Command `{command}` is waiting for approval.", level="warning")
        else:
            self._add_notice(f"$ {command}", level="info", title="Command")
        self._refresh_timeline()
        self._refresh_info()

    def _refresh_timeline(self) -> None:
        if not self.is_mounted:
            return
        try:
            view = self.query_one("#main-timeline", TimelineView)
        except (NoMatches, ScreenStackError):
            return
        items = list(
            build_session_timeline(
                self.runtime,
                notices=tuple(self._timeline_notices),
                hidden_main_message_ids=self._hidden_main_message_ids,
            )
        )
        if self._timeline_cutoff_created_at is not None:
            items = [
                item
                for item in items
                if item.created_at >= self._timeline_cutoff_created_at
            ]
        if not items:
            items = [self._startup_notice()]
        view.set_items(items)

    def _refresh_info(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#info-bar", InfoBar).refresh_from_runtime(
                selected_workflow_run_id=self.selected_workflow_run_id,
                selected_workflow_id=self.selected_workflow_id,
                permission_mode=self._permission_mode,
                turn_status=self._turn_status_text(),
                compose_target_label=self._compose_target.label,
            )
            status_bar = self.query_one("#agent-status-bar", AgentStatusBar)
            status_bar.refresh_from_runtime()
            active_agent_id = self._active_turn.target.agent_id if self._active_turn is not None else None
            if active_agent_id is None:
                status_bar.set_agent_state("orchestrator", self._orchestrator_status_bar_state())
            else:
                status_bar.set_agent_state(active_agent_id, self._active_status_bar_state())
        except (NoMatches, ScreenStackError):
            return

    def _replace_runtime(self, runtime: RuntimeContext, *, notice: str | None = None) -> None:
        self._stop_live_subscription()
        self.runtime = runtime
        self._start_live_subscription()
        self._compose_target = self._orchestrator_target()
        self.selected_workflow_run_id = None
        self._timeline_notices = []
        self._hidden_main_message_ids = set()
        self._timeline_cutoff_created_at = None
        self._time_cursor = int(time.time())

        status_bar = self.query_one("#agent-status-bar", AgentStatusBar)
        status_bar.runtime = self.runtime
        status_bar._agent_states = {}

        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.runtime = self.runtime

        self._update_composer_placeholder()
        self._refresh_timeline()
        self._refresh_info()

        if notice:
            self._add_notice(notice, level="success")
            self._refresh_timeline()

    def _rewind_last_exchange(self) -> None:
        """Hide the last visible user exchange from the timeline."""
        messages = self.runtime.list_main_messages()
        if len(messages) < 2:
            self._add_notice("Nothing to rewind.", level="info")
            self._refresh_timeline()
            return
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].sender == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            self._add_notice("No user message to rewind.", level="info")
            self._refresh_timeline()
            return
        for message in messages[last_user_idx:]:
            self._hidden_main_message_ids.add(message.id)
        self._add_notice("Rewound the last visible exchange.", level="info")
        self._refresh_timeline()

    async def _compact_conversation(self, focus: str | None = None) -> str:
        """Summarize the conversation using the runtime's compaction system."""
        result = await self.runtime.auto_compact(
            focus=focus, created_at=self._next_timestamp(),
        )
        return result or "Compaction produced no output."

    async def _check_auto_compaction(self) -> None:
        """Check if auto-compaction is needed and trigger it."""
        if self._compacting:
            return
        if not self.runtime.needs_compaction():
            return
        self._compacting = True
        thinking = self.query_one("#thinking", ThinkingIndicator)
        ctx = self.runtime.context_window_size()
        used = self.runtime.accumulated_tokens()
        pct = int(used / ctx * 100) if ctx > 0 else 0
        self._add_notice(
            f"Context at {pct}% ({used:,}/{ctx:,} tokens). Auto-compacting...",
            level="info",
        )
        self._refresh_timeline()
        thinking.show("Compacting")
        try:
            summary = await self.runtime.auto_compact(created_at=self._next_timestamp())
        except Exception as exc:
            thinking.hide()
            self._compacting = False
            self._add_notice(f"Auto-compaction failed: {exc}", level="error")
            self._refresh_timeline()
            return
        thinking.hide()
        self._compacting = False
        if summary:
            self._add_notice("Context compacted.", level="success")
        self._refresh_timeline()
        self._refresh_info()

    def _add_notice(self, body: str, *, level: str = "info", title: str | None = None) -> None:
        self._timeline_notices.append(
            NoticeItem(
                item_id=f"notice-{uuid4().hex}",
                title=title,
                body=body,
                level=level,
                created_at=self._next_timestamp(),
            )
        )
        if len(self._timeline_notices) > 40:
            self._timeline_notices = self._timeline_notices[-40:]

    def _startup_notice(self) -> NoticeItem:
        session = self.runtime.current_session()
        session_line = ""
        if session is not None:
            session_line = f"Session: {session.title} ({session.id})\n"
        setup_reason = self.runtime.agent_unavailable_reason("orchestrator")
        body = (
            f"Project: {self.runtime.paths.project_uuid}\n"
            f"{session_line}"
            f"Workspace: {self.runtime.paths.project_root}\n"
        )
        if setup_reason is None:
            body += "Type a message to start."
        else:
            body += (
                f"Orchestrator unavailable: {setup_reason}\n"
                "Use /config to add a provider and role assignment before you start."
            )
        return NoticeItem(
            item_id="startup-notice",
            title="Workspace",
            body=body,
            level="info",
            created_at=0,
        )

    def _has_inflight_turns(self) -> bool:
        active = self._active_turn_task is not None and not self._active_turn_task.done()
        return active or bool(self._queued_turns)

    def _turn_status_text(self) -> str | None:
        active = self._active_turn_task is not None and not self._active_turn_task.done()
        if not active and not self._queued_turns:
            return None
        if self._active_turn is not None:
            label_name = self._active_turn.target.label
        elif self._queued_turns:
            label_name = self._queued_turns[0].target.label
        else:
            label_name = "orchestrator"
        if self._turn_backgrounded:
            label = f"{label_name}: backgrounded"
        elif active:
            label = f"{label_name}: working"
        else:
            label = f"{label_name}: queued"
        if self._queued_turns:
            label = f"{label} (+{len(self._queued_turns)} queued)"
        return label

    def _active_status_bar_state(self) -> str:
        active = self._active_turn_task is not None and not self._active_turn_task.done()
        if self._turn_backgrounded and active:
            return "waiting"
        if active:
            return "working"
        return "ready"

    def _orchestrator_status_bar_state(self) -> str:
        summary = self.runtime.agent_status_summary("orchestrator")
        active = self._active_turn_task is not None and not self._active_turn_task.done()
        if active and self._active_turn is not None and self._active_turn.target.agent_id == "orchestrator":
            return self._active_status_bar_state()
        if "not configured" in summary or "error" in summary:
            return "error"
        if "ready" in summary:
            return "ready"
        return "idle"

    def _start_live_subscription(self) -> None:
        self._stop_live_subscription()
        self._live_subscription = self.runtime.live_state.subscribe()
        self._live_subscription_task = asyncio.create_task(self._watch_live_runtime())

    def _stop_live_subscription(self) -> None:
        if self._live_subscription is not None:
            self._live_subscription.close()
            self._live_subscription = None
        if self._live_subscription_task is not None:
            self._live_subscription_task.cancel()
            self._live_subscription_task = None
        if self._live_refresh_task is not None:
            self._live_refresh_task.cancel()
            self._live_refresh_task = None

    async def _watch_live_runtime(self) -> None:
        subscription = self._live_subscription
        if subscription is None:
            return
        try:
            async for event in subscription:
                self._handle_live_runtime_event(event)
                self._queue_live_refresh()
        except asyncio.CancelledError:
            return

    def _queue_live_refresh(self) -> None:
        if self._live_refresh_task is not None and not self._live_refresh_task.done():
            return
        self._live_refresh_task = asyncio.create_task(self._flush_live_refresh())

    async def _flush_live_refresh(self) -> None:
        try:
            await asyncio.sleep(0.02)
            self._refresh_timeline()
        except asyncio.CancelledError:
            return
        finally:
            self._live_refresh_task = None

    def _handle_live_runtime_event(self, event: LiveRuntimeEvent) -> None:
        if event.kind != "message_failed":
            return
        if event.thread_id == self.runtime.main_thread_id:
            reason = self.runtime.agent_unavailable_reason("orchestrator")
            if reason is not None:
                self._add_notice(
                    f"{reason}\nUse /config to add a provider and role assignment for the orchestrator.",
                    level="error",
                    title="Setup needed",
                )
                return
            self._add_notice(
                f"Error: {event.error}" if event.error else "The orchestrator could not produce a response for that turn.",
                level="error",
                title="Send failed",
            )
            return
        speaker = event.sender or "agent"
        detail = f": {event.error}" if event.error else "."
        self._add_notice(
            f"{speaker} could not finish a response{detail}",
            level="error",
            title="Workroom failed",
        )

    def _next_timestamp(self) -> int:
        self._time_cursor += 1
        return self._time_cursor
