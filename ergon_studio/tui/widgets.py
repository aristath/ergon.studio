"""Custom widgets for the ergon.studio TUI."""

from __future__ import annotations

from rich.text import Text

from textual.message import Message
from textual.timer import Timer
from textual.widgets import Collapsible, Static, TextArea

from ergon_studio.runtime import RuntimeContext
from ergon_studio.storage.models import ThreadRecord

THINKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

AGENT_SPRITES: dict[str, str] = {
    "orchestrator": "⢺⡗",
    "architect": "⣎⣱",
    "coder": "⡢⢔",
    "reviewer": "⠺⢗",
    "fixer": "⢹⡀",
    "researcher": "⣑⣄",
    "tester": "⢄⠊",
    "documenter": "⣏⡇",
    "brainstormer": "⡱⢎",
    "designer": "⡠⠃",
    "user": "⢘⡃",
}

AGENT_COLORS: dict[str, str] = {
    "orchestrator": "bright_cyan",
    "architect": "bright_blue",
    "coder": "bright_green",
    "reviewer": "bright_yellow",
    "fixer": "rgb(255,165,0)",
    "researcher": "bright_magenta",
    "tester": "rgb(0,255,128)",
    "documenter": "grey70",
    "brainstormer": "bright_red",
    "designer": "rgb(255,100,255)",
    "user": "bright_white",
}

STATE_COLORS: dict[str, str] = {
    "idle": "grey35",
    "ready": "grey62",
    "active": "bright_green",
    "working": "bright_cyan",
    "waiting": "yellow",
    "error": "bright_red",
}


class AgentStatusBar(Static):
    """Single-line status bar showing braille sprites for each agent."""

    DEFAULT_CSS = """
    AgentStatusBar {
        height: 1;
        dock: top;
        background: transparent;
        padding: 0 1;
    }
    """

    def __init__(self, runtime: RuntimeContext, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime
        self._agent_states: dict[str, str] = {}

    def on_mount(self) -> None:
        self.refresh_from_runtime()

    def refresh_from_runtime(self) -> None:
        agent_ids = self.runtime.list_agent_ids()
        for agent_id in agent_ids:
            if agent_id not in self._agent_states:
                summary = self.runtime.agent_status_summary(agent_id)
                if "not configured" in summary or "error" in summary:
                    self._agent_states[agent_id] = "error"
                elif "ready" in summary:
                    self._agent_states[agent_id] = "ready"
                else:
                    self._agent_states[agent_id] = "idle"
        self._agent_states.setdefault("user", "ready")
        self.update(self._build_bar())

    def set_agent_state(self, agent_id: str, state: str) -> None:
        self._agent_states[agent_id] = state
        self.update(self._build_bar())

    def _build_bar(self) -> Text:
        text = Text()
        agent_ids = list(AGENT_SPRITES.keys())
        for i, agent_id in enumerate(agent_ids):
            sprite = AGENT_SPRITES[agent_id]
            state = self._agent_states.get(agent_id, "idle")
            if state in ("active", "working"):
                color = AGENT_COLORS.get(agent_id, "white")
            else:
                color = STATE_COLORS.get(state, "grey35")
            text.append(sprite, style=color)
            if i < len(agent_ids) - 1:
                text.append(" ")
        return text


class SideThreadBlock(Collapsible):
    """Collapsible block showing a side thread's messages."""

    DEFAULT_CSS = """
    SideThreadBlock {
        height: auto;
        margin: 0;
        padding: 0;
        background: transparent;
        border-top: none;
        padding-left: 0;
        padding-bottom: 0;
    }

    SideThreadBlock:focus-within {
        background-tint: transparent;
    }

    SideThreadBlock > CollapsibleTitle {
        background: transparent;
        padding: 0 1;
    }

    SideThreadBlock > CollapsibleTitle:hover {
        background: transparent;
    }

    SideThreadBlock > CollapsibleTitle:focus {
        background: transparent;
    }

    SideThreadBlock > Contents {
        background: transparent;
        padding: 1 0 0 2;
    }
    """

    def __init__(self, thread: ThreadRecord, runtime: RuntimeContext, **kwargs) -> None:
        self._thread = thread
        self._runtime = runtime
        self._message_count = len(runtime.list_thread_messages(thread.id))
        title = self._build_title()
        self._content_widget = Static("", classes="thread-content")
        super().__init__(self._content_widget, title=title, collapsed=True, **kwargs)

    def _build_title(self) -> str:
        agent = self._thread.assigned_agent_id or self._thread.kind
        sprite = AGENT_SPRITES.get(agent, "")
        summary = self._thread.summary or self._thread.kind
        return f"{sprite} {agent} › {summary} [{self._message_count} msgs]"

    def refresh_messages(self) -> None:
        messages = self._runtime.list_thread_messages(self._thread.id)
        self._message_count = len(messages)
        self.title = self._build_title()
        if not messages:
            self._content_widget.update("No messages yet.")
            return
        lines = []
        for msg in messages:
            body = self._runtime.conversation_store.read_message_body(msg).rstrip("\n")
            lines.append(f"[bold]{msg.sender}[/bold] {body}")
        self._content_widget.update("\n".join(lines))


class InfoBar(Static):
    """Two-line bottom bar with workflow state and command hints."""

    DEFAULT_CSS = """
    InfoBar {
        height: 2;
        background: transparent;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, runtime: RuntimeContext, **kwargs) -> None:
        super().__init__(**kwargs)
        self.runtime = runtime

    def on_mount(self) -> None:
        self.refresh_from_runtime()

    def refresh_from_runtime(
        self,
        selected_workflow_run_id: str | None = None,
        selected_workflow_id: str | None = None,
        permission_mode: str = "default",
    ) -> None:
        line1 = self._build_status_line(selected_workflow_run_id, selected_workflow_id, permission_mode)
        line2 = "/help /config /workflows /agent <name> /memory /threads"
        self.update(f"{line1}\n{line2}")

    def _build_status_line(
        self,
        workflow_run_id: str | None,
        workflow_id: str | None,
        permission_mode: str = "default",
    ) -> str:
        parts: list[str] = []
        session = self.runtime.current_session()

        if session is not None:
            parts.append(f"session: {session.title}")

        if workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(workflow_run_id)
            if run_view is not None:
                run = run_view.workflow_run
                step = run.current_step_index
                total = len(run_view.steps)
                filled = "▪" * step
                empty = "○" * max(0, total - step)
                parts.append(f"{run.workflow_id} step {step}/{total} {filled}{empty} [{run.state}]")
        elif workflow_id:
            parts.append(f"workflow: {workflow_id}")

        pending = self.runtime.list_pending_approvals()
        if pending:
            count = len(pending)
            parts.append(f"{count} pending approval{'s' if count != 1 else ''} (Ctrl+Y/R)")

        if permission_mode != "default":
            mode_labels = {"auto-approve": "auto", "plan": "plan"}
            parts.append(f"mode: {mode_labels.get(permission_mode, permission_mode)}")

        return " │ ".join(parts) if parts else "No active workflow"


class ThinkingIndicator(Static):
    """Animated spinner shown while the LLM is processing."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        padding: 0 1;
        display: none;
    }

    ThinkingIndicator.visible {
        display: block;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer: Timer | None = None

    def show(self, label: str = "Thinking") -> None:
        self._label = label
        self._frame = 0
        self.add_class("visible")
        self._timer = self.set_interval(0.1, self._tick)

    def hide(self) -> None:
        self.remove_class("visible")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.update("")

    def _tick(self) -> None:
        spinner = THINKING_FRAMES[self._frame % len(THINKING_FRAMES)]
        self.update(Text.from_markup(f"[bright_cyan]{spinner}[/bright_cyan] [dim]{self._label}...[/dim]"))
        self._frame += 1


class ComposerTextArea(TextArea):
    """Multi-line text input. Enter submits, Alt+Enter for newline, up/down for history."""

    class Submitted(Message):
        """Posted when the user presses Enter to submit."""

        def __init__(self, text_area: ComposerTextArea, value: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.value = value

    DEFAULT_CSS = """
    ComposerTextArea {
        height: auto;
        max-height: 10;
        min-height: 3;
        margin: 0 1;
    }
    """

    def __init__(self, placeholder: str = "", **kwargs) -> None:
        super().__init__("", show_line_numbers=False, placeholder=placeholder, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.clear()
        if new_value:
            self.insert(new_value)

    def _push_history(self, text: str) -> None:
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = -1
        self._draft = ""

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            text = self.text.strip()
            if text:
                self._push_history(text)
                self.post_message(self.Submitted(self, text))
            return
        if event.key == "alt+enter":
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "escape":
            event.prevent_default()
            if self.text:
                self.clear()
                self._history_index = -1
            return
        if event.key == "up" and "\n" not in self.text and self._history:
            event.prevent_default()
            if self._history_index == -1:
                self._draft = self.text
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self.value = self._history[self._history_index]
            return
        if event.key == "down" and "\n" not in self.text and self._history_index >= 0:
            event.prevent_default()
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
            else:
                self._history_index = -1
                self.value = self._draft
            return
        await super()._on_key(event)
