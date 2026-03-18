from __future__ import annotations

import time
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

from ergon_studio.runtime import RuntimeContext


class Panel(Static):
    def __init__(self, title: str, body: str, *, panel_id: str, classes: str | None = None) -> None:
        self.title_text = title
        self.body = body
        super().__init__(self._render_panel(title, body), id=panel_id, classes=classes)

    @staticmethod
    def _render_panel(title: str, body: str) -> str:
        return f"[b]{title}[/b]\n{body}"

    def set_body(self, body: str) -> None:
        self.body = body
        self.update(self._render_panel(self.title_text, body))


class ErgonStudioApp(App[None]):
    TITLE = "ergon.studio"
    CSS = """
    Screen {
      layout: vertical;
    }

    #workspace {
      height: 1fr;
      layout: horizontal;
    }

    #left-sidebar {
      width: 28;
      layout: vertical;
    }

    #center-column {
      width: 1fr;
      layout: vertical;
    }

    #right-sidebar {
      width: 34;
      layout: vertical;
    }

    .panel {
      border: round $accent;
      padding: 1;
      margin: 0 1 1 0;
      height: 1fr;
    }

    #main-chat {
      height: 2fr;
    }

    #activity {
      height: 1fr;
    }

    #composer-input {
      dock: bottom;
      margin: 0 1 1 0;
    }
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        super().__init__()
        self.runtime = runtime

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="left-sidebar"):
                yield Panel("Tasks", "Task tree will appear here.", panel_id="tasks", classes="panel")
                yield Panel("Threads", self._render_threads_body(), panel_id="threads", classes="panel")
                yield Panel("Activity", "Workflow activity will appear here.", panel_id="activity", classes="panel")
            with Vertical(id="center-column"):
                yield Panel(
                    "Main Chat",
                    self._render_main_chat_body(),
                    panel_id="main-chat",
                    classes="panel",
                )
                yield Panel("Artifacts", "Diffs and generated artifacts will appear here.", panel_id="artifacts", classes="panel")
            with Vertical(id="right-sidebar"):
                yield Panel("Approvals", "Approval requests will appear here.", panel_id="approvals", classes="panel")
                yield Panel(
                    "Memory",
                    (
                        f"Agents: {len(self.runtime.registry.agent_definitions)}\n"
                        f"Workflows: {len(self.runtime.registry.workflow_definitions)}"
                    ),
                    panel_id="memory",
                    classes="panel",
                )
                yield Panel(
                    "Settings",
                    f"Configured providers: {len(self.runtime.registry.config.get('providers', {}))}",
                    panel_id="settings",
                    classes="panel",
                )
        yield Input(placeholder="Message the orchestrator...", id="composer-input")
        yield Footer()

    def _render_threads_body(self) -> str:
        threads = self.runtime.list_threads()
        if not threads:
            return "No threads yet."
        return "\n".join(
            f"{thread.id} ({thread.kind})"
            for thread in threads
        )

    def _render_main_chat_body(self) -> str:
        messages = self.runtime.list_main_messages()
        if not messages:
            return (
                f"Project UUID: {self.runtime.paths.project_uuid}\n"
                f"Workspace: {self.runtime.paths.project_root}\n"
                "No messages yet."
            )

        rendered_messages = []
        for message in messages:
            body = self.runtime.conversation_store.read_message_body(message).rstrip("\n")
            rendered_messages.append(f"[{message.sender}] {body}")
        return "\n\n".join(rendered_messages)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message_body = event.value.strip()
        if not message_body:
            event.input.value = ""
            return

        self.runtime.append_message_to_main_thread(
            message_id=f"message-{uuid4().hex}",
            sender="user",
            kind="chat",
            body=message_body,
            created_at=int(time.time()),
        )
        self.query_one("#main-chat", Panel).set_body(self._render_main_chat_body())
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        event.input.value = ""
