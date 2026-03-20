from __future__ import annotations

import time
from uuid import uuid4

from rich.markdown import Markdown
from rich.panel import Panel as RichPanel
from rich.text import Text

from textual.app import App, ComposeResult, ScreenStackError
from textual.css.query import NoMatches
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, OptionList, RichLog, Static, TextArea

from ergon_studio.runtime import RuntimeContext, load_runtime
from ergon_studio.tui.widgets import AgentStatusBar, ComposerTextArea, InfoBar, SideThreadBlock, ThinkingIndicator

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show available commands"),
    ("/clear", "Clear conversation, keep config"),
    ("/compact", "Summarize conversation to save context"),
    ("/context", "Show token/context usage"),
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
    ("/workflow", "Select a workflow by name"),
    ("/agent", "Open a direct thread with an agent"),
    ("/memory", "Show memory facts"),
    ("/threads", "List all threads"),
    ("/approvals", "Show approval history"),
    ("/events", "Show recent activity"),
    ("/init", "Initialize project configuration"),
]


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
        ("ctrl+o", "open_session_picker", "Sessions"),
        ("ctrl+y", "approve_pending", "Approve"),
        ("ctrl+r", "reject_pending", "Reject"),
        ("ctrl+g", "edit_global_config", "Edit Config"),
        ("ctrl+e", "edit_orchestrator_definition", "Edit Orchestrator"),
        ("ctrl+x", "run_workspace_command", "Run Command"),
        ("ctrl+c", "quit", "Quit"),
    ]
    CSS = """
    Screen {
      layout: vertical;
    }

    #chat-area {
      height: 1fr;
      background: transparent;
    }

    #side-threads {
      height: auto;
      max-height: 30%;
    }

    #main-chat {
      height: 1fr;
      padding: 0 1;
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

    .thread-content {
      padding: 0 2;
      color: $text-muted;
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

    def __init__(self, runtime: RuntimeContext, *, open_session_picker_on_mount: bool = False) -> None:
        super().__init__()
        self.runtime = runtime
        self.open_session_picker_on_mount = open_session_picker_on_mount
        self.selected_workflow_id = "standard-build"
        self.selected_workflow_run_id: str | None = None
        self._target_thread_id: str | None = None
        self._chat_message_count = 0
        self._known_thread_ids: set[str] = set()
        self._known_approval_ids: set[str] = set()
        self._time_cursor = int(time.time())
        self._last_escape_time: float = 0.0

    def compose(self) -> ComposeResult:
        yield AgentStatusBar(self.runtime, id="agent-status-bar")
        with VerticalScroll(id="chat-area"):
            yield Vertical(id="side-threads")
            yield RichLog(id="main-chat", markup=True, auto_scroll=True, wrap=True)
        yield ThinkingIndicator(id="thinking")
        yield OptionList(id="slash-commands")
        yield ComposerTextArea(placeholder="❯ Message the orchestrator...", id="composer-input")
        yield InfoBar(self.runtime, id="info-bar")

    def on_mount(self) -> None:
        self.set_focus(self.query_one("#composer-input", ComposerTextArea))
        self._load_existing_messages()
        self._load_existing_threads()
        self._load_existing_approvals()
        self._refresh_info()
        if self.open_session_picker_on_mount:
            self._open_session_picker()

    def _load_existing_messages(self) -> None:
        chat = self.query_one("#main-chat", RichLog)
        messages = self.runtime.list_main_messages()
        if not messages:
            session = self.runtime.current_session()
            session_line = ""
            if session is not None:
                session_line = f"Session: {session.title} ({session.id})\n"
            chat.write(
                f"[dim]Project: {self.runtime.paths.project_uuid}\n"
                f"{session_line}"
                f"Workspace: {self.runtime.paths.project_root}\n"
                "Type a message to start.[/dim]"
            )
        for msg in messages:
            body = self.runtime.conversation_store.read_message_body(msg).rstrip("\n")
            chat.write(self._format_message(msg.sender, body))
        self._chat_message_count = len(messages)

    def _load_existing_threads(self) -> None:
        threads = self.runtime.list_threads()
        for thread in threads:
            if thread.id == self.runtime.main_thread_id:
                continue
            if thread.id not in self._known_thread_ids:
                self._mount_side_thread(thread)

    def _load_existing_approvals(self) -> None:
        chat = self.query_one("#main-chat", RichLog)
        for approval in self.runtime.list_pending_approvals():
            if approval.id not in self._known_approval_ids:
                self._known_approval_ids.add(approval.id)
                chat.write(self._format_approval(approval))

    def _mount_side_thread(self, thread) -> None:
        self._known_thread_ids.add(thread.id)
        block = SideThreadBlock(
            thread, self.runtime, id=f"side-thread-{thread.id}"
        )
        self.query_one("#side-threads", Vertical).mount(block)

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        widget = event.collapsible
        if isinstance(widget, SideThreadBlock):
            self._target_thread_id = widget._thread.id
            widget.refresh_messages()
            self._update_composer_placeholder()

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        widget = event.collapsible
        if isinstance(widget, SideThreadBlock):
            if self._target_thread_id == widget._thread.id:
                self._target_thread_id = None
                self._update_composer_placeholder()

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
        if value.startswith("/") and " " not in value:
            prefix = value.lower()
            matches = [
                f"{cmd}  [dim]{desc}[/dim]"
                for cmd, desc in SLASH_COMMANDS
                if cmd.startswith(prefix)
            ]
            cmd_list.clear_options()
            if matches:
                for item in matches:
                    cmd_list.add_option(item)
                cmd_list.add_class("visible")
                cmd_list.highlighted = 0
            else:
                cmd_list.remove_class("visible")
        else:
            cmd_list.remove_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "slash-commands":
            return
        if not self.is_mounted:
            return
        # Extract the command from "  /cmd  description"
        text = str(event.option.prompt).strip()
        cmd = text.split()[0] if text else ""
        try:
            inp = self.query_one("#composer-input", ComposerTextArea)
        except (NoMatches, ScreenStackError):
            return
        # If command takes args, put cursor after it with a space
        if cmd in ("/workflow", "/agent", "/new-session", "/rename-session", "/archive-session", "/switch-session"):
            inp.value = cmd + " "
        else:
            inp.value = cmd
        event.option_list.remove_class("visible")
        self.set_focus(inp)
        # Auto-submit commands that don't need args
        if cmd not in ("/workflow", "/agent", "/new-session", "/rename-session", "/archive-session", "/switch-session"):
            inp.post_message(ComposerTextArea.Submitted(inp, cmd))

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

        event.text_area.disabled = True
        try:
            if text.startswith("/"):
                await self._handle_slash_command(text)
            elif self._target_thread_id is not None:
                await self._send_to_thread(self._target_thread_id, text)
            else:
                await self._send_to_orchestrator(text)
            event.text_area.value = ""
        finally:
            event.text_area.disabled = False

    async def _send_to_orchestrator(self, body: str) -> None:
        chat = self.query_one("#main-chat", RichLog)
        thinking = self.query_one("#thinking", ThinkingIndicator)
        created_at = self._next_timestamp()
        chat.write(self._format_message("you", body))
        self._chat_message_count += 1
        thinking.show("Thinking")
        try:
            user_msg, reply_msg = await self.runtime.send_user_message_to_orchestrator(
                body=body, created_at=created_at,
            )
        except Exception as exc:
            thinking.hide()
            chat.write(f"[bold red]Error:[/bold red] {exc}")
            return
        thinking.hide()
        if reply_msg is not None:
            reply_body = self.runtime.conversation_store.read_message_body(reply_msg).rstrip("\n")
            chat.write(self._format_message(reply_msg.sender, reply_body))
            self._chat_message_count += 1
        self._refresh_chat()

    async def _send_to_thread(self, thread_id: str, body: str) -> None:
        thinking = self.query_one("#thinking", ThinkingIndicator)
        created_at = self._next_timestamp()
        thinking.show("Working")
        try:
            user_msg, reply_msg = await self.runtime.send_message_to_agent_thread(
                thread_id=thread_id, body=body, created_at=created_at,
            )
        except Exception as exc:
            thinking.hide()
            self.query_one("#main-chat", RichLog).write(f"[bold red]Error:[/bold red] {exc}")
            return
        thinking.hide()
        # Refresh the side thread block
        for widget in self.query(SideThreadBlock):
            if widget._thread.id == thread_id:
                widget.refresh_messages()
                break

    async def _handle_slash_command(self, text: str) -> None:
        chat = self.query_one("#main-chat", RichLog)
        parts = text.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            lines = ["[bold]Available commands:[/bold]"]
            for cmd, desc in SLASH_COMMANDS:
                lines.append(f"  {cmd:20s} {desc}")
            chat.write("\n".join(lines))
        elif command == "/clear":
            self.query_one("#main-chat", RichLog).clear()
            self._chat_message_count = len(self.runtime.list_main_messages())
            chat = self.query_one("#main-chat", RichLog)
            chat.write("[dim]Conversation cleared.[/dim]")
        elif command == "/compact":
            focus = args.strip() or None
            chat.write("[dim]Compacting conversation...[/dim]")
            thinking = self.query_one("#thinking", ThinkingIndicator)
            thinking.show("Compacting")
            try:
                summary = await self._compact_conversation(focus)
                thinking.hide()
                chat.write(f"[green]Compacted.[/green] Summary:\n{summary}")
            except Exception as exc:
                thinking.hide()
                chat.write(f"[red]Compaction failed:[/red] {exc}")
        elif command == "/context":
            messages = self.runtime.list_main_messages()
            total_chars = 0
            for msg in messages:
                total_chars += len(self.runtime.conversation_store.read_message_body(msg))
            approx_tokens = total_chars // 4
            chat.write(
                f"[bold]Context usage:[/bold]\n"
                f"  Messages: {len(messages)}\n"
                f"  Characters: {total_chars:,}\n"
                f"  Estimated tokens: ~{approx_tokens:,}"
            )
        elif command == "/session":
            session = self.runtime.current_session()
            if session is None:
                chat.write("[dim]No active session.[/dim]")
            else:
                chat.write(
                    f"[bold]Session:[/bold] {session.title}\n"
                    f"[dim]{session.id}[/dim]"
                )
        elif command == "/sessions":
            sessions = self.runtime.list_sessions(include_archived=True)
            if not sessions:
                chat.write("[dim]No sessions yet.[/dim]")
            else:
                lines = ["[bold]Sessions:[/bold]"]
                for session in sessions:
                    marker = ">" if session.id == self.runtime.main_session_id else " "
                    archived = " archived" if session.archived_at is not None else ""
                    lines.append(
                        f"  {marker} {session.title} [dim]({session.id})[/dim]{archived}"
                    )
                chat.write("\n".join(lines))
        elif command == "/new-session":
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
                chat.write("[red]Usage: /rename-session <title>[/red]")
            else:
                renamed = self.runtime.rename_session(
                    session_id=self.runtime.main_session_id,
                    title=title,
                    created_at=self._next_timestamp(),
                )
                chat.write(f"[green]Renamed[/green] current session to {renamed.title}")
                self._refresh_info()
        elif command == "/archive-session":
            target = args.strip() or self.runtime.main_session_id
            try:
                self.runtime.archive_session(
                    session_id=target,
                    created_at=self._next_timestamp(),
                )
            except ValueError:
                chat.write(f"[red]Unknown session:[/red] {target}")
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
                    chat.write(f"[yellow]Archived[/yellow] session {target}")
        elif command == "/switch-session":
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
                    chat.write(f"[red]Unknown session:[/red] {target}")
                else:
                    self._replace_runtime(
                        new_runtime,
                        notice=f"[green]Switched[/green] to session {new_runtime.current_session().title}",
                    )
        elif command == "/config":
            self._open_config_wizard()
        elif command == "/workflows":
            lines = ["[bold]Workflows:[/bold]"]
            for wf_id in self.runtime.list_workflow_ids():
                marker = "> " if wf_id == self.selected_workflow_id else "  "
                lines.append(f"  {marker}{wf_id}")
            chat.write("\n".join(lines))
        elif command == "/workflow":
            if not args:
                chat.write("[red]Usage: /workflow <name>[/red]")
            elif args in self.runtime.list_workflow_ids():
                self.selected_workflow_id = args
                chat.write(f"[dim]Selected workflow: {args}[/dim]")
                self._refresh_info()
            else:
                chat.write(f"[red]Unknown workflow: {args}[/red]")
        elif command == "/agent":
            if not args:
                chat.write("[red]Usage: /agent <name>[/red]")
            elif args in self.runtime.list_agent_ids():
                self._open_agent_thread(args)
            else:
                chat.write(f"[red]Unknown agent: {args}[/red]")
        elif command == "/memory":
            facts = self.runtime.list_memory_facts()
            if not facts:
                chat.write("[dim]No memory facts yet.[/dim]")
            else:
                lines = ["[bold]Memory facts:[/bold]"]
                for fact in facts[-10:]:
                    lines.append(f"  [{fact.kind}] {fact.content}")
                chat.write("\n".join(lines))
        elif command == "/threads":
            threads = self.runtime.list_threads()
            if not threads:
                chat.write("[dim]No threads yet.[/dim]")
            else:
                lines = ["[bold]Threads:[/bold]"]
                for thread in threads:
                    agent = thread.assigned_agent_id or thread.kind
                    count = len(self.runtime.list_thread_messages(thread.id))
                    lines.append(f"  {agent} ({thread.kind}) [{count} msgs]")
                chat.write("\n".join(lines))
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
                chat.write("\n".join(lines))
            else:
                chat.write("[dim]Use /config to change model assignments.[/dim]")
        elif command == "/status":
            lines = ["[bold]Provider status:[/bold]"]
            for pid in self.runtime.list_provider_ids():
                details = self.runtime.provider_details(pid)
                if details:
                    model = details.get("model", "?")
                    url = details.get("base_url", "?")
                    lines.append(f"  [green]●[/green] {pid}: {model} @ {url}")
                else:
                    lines.append(f"  [red]●[/red] {pid}: invalid config")
            if not self.runtime.list_provider_ids():
                lines.append("  [red]No providers configured.[/red] Use /config")
            chat.write("\n".join(lines))
        elif command == "/doctor":
            issues: list[str] = []
            if not self.runtime.list_provider_ids():
                issues.append("[red]✗[/red] No providers configured")
            for agent_id in self.runtime.list_agent_ids():
                summary = self.runtime.agent_status_summary(agent_id)
                if "not configured" in summary:
                    issues.append(f"[red]✗[/red] {agent_id}: {summary}")
            if not issues:
                chat.write("[green]All checks passed.[/green] Providers configured, agents ready.")
            else:
                chat.write("[bold]Issues found:[/bold]\n" + "\n".join(issues))
        elif command == "/team":
            from ergon_studio.tui.widgets import AGENT_SPRITES, AGENT_COLORS
            lines = ["[bold]Team:[/bold]"]
            for agent_id in self.runtime.list_agent_ids():
                sprite = AGENT_SPRITES.get(agent_id, "  ")
                color = AGENT_COLORS.get(agent_id, "white")
                summary = self.runtime.agent_status_summary(agent_id)
                lines.append(f"  [{color}]{sprite}[/{color}] {agent_id}: {summary}")
            chat.write("\n".join(lines))
        elif command == "/approvals":
            all_approvals = self.runtime.list_approvals()
            if not all_approvals:
                chat.write("[dim]No approvals yet.[/dim]")
            else:
                lines = ["[bold]Approvals:[/bold]"]
                for a in all_approvals[-10:]:
                    status_color = {"approved": "green", "rejected": "red"}.get(a.status, "yellow")
                    lines.append(f"  [{status_color}]{a.status}[/{status_color}] {a.action} by {a.requester}")
                chat.write("\n".join(lines))
        elif command == "/events":
            events = self.runtime.list_events()
            if not events:
                chat.write("[dim]No events yet.[/dim]")
            else:
                lines = ["[bold]Recent events:[/bold]"]
                for e in events[-15:]:
                    lines.append(f"  [dim]{e.kind}:[/dim] {e.summary}")
                chat.write("\n".join(lines))
        elif command == "/init":
            chat.write(
                "[dim]Project already initialized.\n"
                f"UUID: {self.runtime.paths.project_uuid}\n"
                f"Data: {self.runtime.paths.project_data_dir}[/dim]"
            )
        else:
            chat.write(f"[red]Unknown command: {command}[/red]. Type /help for available commands.")

    def _open_agent_thread(self, agent_id: str) -> None:
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
        self._mount_side_thread(thread)
        chat = self.query_one("#main-chat", RichLog)
        chat.write(f"[dim]Opened thread with {agent_id}. Expand it above to chat.[/dim]")

    def action_approve_pending(self) -> None:
        pending = self.runtime.list_pending_approvals()
        if not pending:
            return
        approval = pending[0]
        self.runtime.resolve_approval(
            approval_id=approval.id,
            status="approved",
            created_at=self._next_timestamp(),
        )
        chat = self.query_one("#main-chat", RichLog)
        chat.write(f"[green]Approved[/green] {approval.action} by {approval.requester}")
        self._refresh_info()

    def action_reject_pending(self) -> None:
        pending = self.runtime.list_pending_approvals()
        if not pending:
            return
        approval = pending[0]
        self.runtime.resolve_approval(
            approval_id=approval.id,
            status="rejected",
            created_at=self._next_timestamp(),
        )
        chat = self.query_one("#main-chat", RichLog)
        chat.write(f"[red]Rejected[/red] {approval.action} by {approval.requester}")
        self._refresh_info()

    def action_edit_global_config(self) -> None:
        self._open_config_wizard()

    def action_open_session_picker(self) -> None:
        self._open_session_picker()

    def _open_config_wizard(self) -> None:
        from ergon_studio.tui.config_wizard import ConfigWizardScreen

        def on_dismiss(result: None) -> None:
            self._refresh_info()
            self.query_one(AgentStatusBar).refresh_from_runtime()

        self.push_screen(ConfigWizardScreen(self.runtime), on_dismiss)

    def _open_session_picker(self) -> None:
        sessions = self.runtime.list_sessions()
        if not sessions:
            self.query_one("#main-chat", RichLog).write("[dim]No sessions available.[/dim]")
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
        chat = self.query_one("#main-chat", RichLog)
        approval_id = result.get("approval_id")
        if isinstance(approval_id, str):
            self._load_new_approvals()
        else:
            chat.write(f"[dim]$ {command}[/dim]")
        self._refresh_info()

    def _refresh_chat(self) -> None:
        self._load_new_messages()
        self._load_new_threads()
        self._load_new_approvals()
        self._refresh_info()

    def _load_new_messages(self) -> None:
        chat = self.query_one("#main-chat", RichLog)
        messages = self.runtime.list_main_messages()
        new_messages = messages[self._chat_message_count:]
        for msg in new_messages:
            body = self.runtime.conversation_store.read_message_body(msg).rstrip("\n")
            chat.write(self._format_message(msg.sender, body))
        self._chat_message_count = len(messages)

    def _load_new_threads(self) -> None:
        threads = self.runtime.list_threads()
        for thread in threads:
            if thread.id == self.runtime.main_thread_id:
                continue
            if thread.id not in self._known_thread_ids:
                self._mount_side_thread(thread)
        # Refresh existing thread blocks
        for widget in self.query(SideThreadBlock):
            if not widget.collapsed:
                widget.refresh_messages()

    def _load_new_approvals(self) -> None:
        chat = self.query_one("#main-chat", RichLog)
        for approval in self.runtime.list_pending_approvals():
            if approval.id not in self._known_approval_ids:
                self._known_approval_ids.add(approval.id)
                chat.write(self._format_approval(approval))

    def _refresh_info(self) -> None:
        self.query_one("#info-bar", InfoBar).refresh_from_runtime(
            selected_workflow_run_id=self.selected_workflow_run_id,
            selected_workflow_id=self.selected_workflow_id,
        )
        self.query_one("#agent-status-bar", AgentStatusBar).refresh_from_runtime()

    def _replace_runtime(self, runtime: RuntimeContext, *, notice: str | None = None) -> None:
        self.runtime = runtime
        self.selected_workflow_run_id = None
        self._target_thread_id = None
        self._chat_message_count = 0
        self._known_thread_ids = set()
        self._known_approval_ids = set()
        self._time_cursor = int(time.time())

        self.query_one("#main-chat", RichLog).clear()
        self.query_one("#side-threads", Vertical).remove_children()

        status_bar = self.query_one("#agent-status-bar", AgentStatusBar)
        status_bar.runtime = self.runtime
        status_bar._agent_states = {}

        info_bar = self.query_one("#info-bar", InfoBar)
        info_bar.runtime = self.runtime

        self._load_existing_messages()
        self._load_existing_threads()
        self._load_existing_approvals()
        self._update_composer_placeholder()
        self._refresh_info()

        if notice:
            self.query_one("#main-chat", RichLog).write(notice)

    def _update_composer_placeholder(self) -> None:
        if not self.is_mounted:
            return
        composer = self.query_one("#composer-input", ComposerTextArea)
        if self._target_thread_id is not None:
            thread = self.runtime.get_thread(self._target_thread_id)
            if thread is not None:
                label = thread.assigned_agent_id or thread.kind
                composer.placeholder = f"❯ Message {label} directly..."
                return
        composer.placeholder = "❯ Message the orchestrator..."

    @staticmethod
    def _format_message(sender: str, body: str):
        label = "you" if sender == "user" else sender
        color = "bright_white" if sender == "user" else "bright_cyan"
        header = f"**{label}**"
        # If body contains code fences or markdown formatting, render as Markdown
        if "```" in body or "\n#" in body:
            return Markdown(f"{header}\n\n{body}")
        return f"[bold {color}]{label}[/bold {color}] {body}"

    @staticmethod
    def _format_approval(approval) -> RichPanel:
        payload_text = f"{approval.action} by {approval.requester}"
        return RichPanel(
            f"[bold yellow]Approval Required[/bold yellow]\n"
            f"[{approval.risk_class}] {payload_text}\n"
            f"Reason: {approval.reason}\n"
            f"[dim]Ctrl+Y to approve │ Ctrl+R to reject[/dim]",
            border_style="yellow",
            title="Approval",
            expand=True,
        )

    def _rewind_last_exchange(self) -> None:
        """Remove the last user message and its reply from the chat display."""
        chat = self.query_one("#main-chat", RichLog)
        messages = self.runtime.list_main_messages()
        if len(messages) < 2:
            chat.write("[dim]Nothing to rewind.[/dim]")
            return
        # Find the last user message and everything after it
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].sender == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            chat.write("[dim]No user message to rewind.[/dim]")
            return
        # Clear and reload without the last exchange
        chat.clear()
        self._chat_message_count = 0
        for msg in messages[:last_user_idx]:
            body = self.runtime.conversation_store.read_message_body(msg).rstrip("\n")
            chat.write(self._format_message(msg.sender, body))
            self._chat_message_count += 1
        chat.write("[dim]Rewound last exchange.[/dim]")

    async def _compact_conversation(self, focus: str | None = None) -> str:
        """Summarize the conversation using the orchestrator's LLM."""
        messages = self.runtime.list_main_messages()
        if not messages:
            return "No messages to compact."
        conversation_text = []
        for msg in messages:
            body = self.runtime.conversation_store.read_message_body(msg).rstrip("\n")
            conversation_text.append(f"[{msg.sender}] {body}")
        full_text = "\n\n".join(conversation_text)
        focus_hint = f"\nFocus on: {focus}" if focus else ""
        prompt = (
            "Summarize this conversation for continuity. Preserve:\n"
            "- Current goals and active tasks\n"
            "- Decisions made and their rationale\n"
            "- Key technical facts (file paths, architecture choices, conventions)\n"
            "- Outstanding issues and blockers\n"
            "- User preferences expressed during the session\n"
            "Discard: verbose reasoning, repeated attempts, intermediate tool outputs.\n"
            f"{focus_hint}\n\n"
            f"Conversation:\n{full_text}"
        )
        result = await self.runtime.generate_agent_text_without_tools(
            agent_id="orchestrator",
            body=prompt,
            created_at=self._next_timestamp(),
        )
        return result or "Compaction produced no output."

    def _next_timestamp(self) -> int:
        self._time_cursor += 1
        return self._time_cursor
