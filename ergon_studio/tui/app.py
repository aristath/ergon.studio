from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static, TextArea

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
      background: $surface;
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

    def __init__(self, *, title: str, initial_text: str, on_save) -> None:
        super().__init__()
        self.title = title
        self.initial_text = initial_text
        self.on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical(id="definition-editor-container"):
            yield Static(f"[b]{self.title}[/b]\n`Ctrl+S` to save, `Esc` to cancel.")
            yield TextArea(self.initial_text, id="definition-editor", language="markdown")
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


class ErgonStudioApp(App[None]):
    TITLE = "ergon.studio"
    BINDINGS = [
        ("ctrl+j", "next_thread", "Next Thread"),
        ("ctrl+k", "previous_thread", "Previous Thread"),
        ("ctrl+e", "edit_orchestrator_definition", "Edit Orchestrator"),
    ]
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
        self.selected_thread_id = runtime.main_thread_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="left-sidebar"):
                yield Panel("Tasks", self._render_tasks_body(), panel_id="tasks", classes="panel")
                yield Panel("Threads", self._render_threads_body(), panel_id="threads", classes="panel")
                yield Panel("Activity", self._render_activity_body(), panel_id="activity", classes="panel")
            with Vertical(id="center-column"):
                yield Panel(
                    "Main Chat",
                    self._render_main_chat_body(),
                    panel_id="main-chat",
                    classes="panel",
                )
                yield Panel(
                    "Selected Thread",
                    self._render_selected_thread_body(),
                    panel_id="selected-thread",
                    classes="panel",
                )
                yield Panel("Artifacts", self._render_artifacts_body(), panel_id="artifacts", classes="panel")
            with Vertical(id="right-sidebar"):
                yield Panel("Approvals", self._render_approvals_body(), panel_id="approvals", classes="panel")
                yield Panel(
                    "Memory",
                    self._render_memory_body(),
                    panel_id="memory",
                    classes="panel",
                )
                yield Panel(
                    "Settings",
                    self._render_settings_body(),
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
            f"{'> ' if thread.id == self.selected_thread_id else '  '}{thread.id} ({thread.kind})"
            for thread in threads
        )

    def _render_tasks_body(self) -> str:
        tasks = self.runtime.list_tasks()
        if not tasks:
            return "No tasks yet."
        return "\n".join(
            f"{task.id} [{task.state}] {task.title}"
            for task in tasks
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

    def _render_selected_thread_body(self) -> str:
        messages = self.runtime.list_thread_messages(self.selected_thread_id)
        if not messages:
            return f"{self.selected_thread_id}\nNo messages yet."

        rendered_messages = [self.selected_thread_id]
        for message in messages:
            body = self.runtime.conversation_store.read_message_body(message).rstrip("\n")
            rendered_messages.append(f"[{message.sender}] {body}")
        return "\n\n".join(rendered_messages)

    def _render_activity_body(self) -> str:
        events = self.runtime.list_events()
        if not events:
            return "No activity yet."
        return "\n".join(
            f"{event.kind}: {event.summary}"
            for event in events[-8:]
        )

    def _render_approvals_body(self) -> str:
        approvals = self.runtime.list_approvals()
        if not approvals:
            return "No approvals pending."
        return "\n".join(
            f"{approval.id} [{approval.risk_class}] {approval.action}"
            for approval in approvals
        )

    def _render_memory_body(self) -> str:
        facts = self.runtime.list_memory_facts()
        if not facts:
            return "No memory facts yet."
        return "\n".join(
            f"{fact.id} [{fact.kind}] {fact.content}"
            for fact in facts[-8:]
        )

    def _render_artifacts_body(self) -> str:
        artifacts = self.runtime.list_artifacts()
        if not artifacts:
            return "No artifacts yet."
        return "\n".join(
            f"{artifact.id} [{artifact.kind}] {artifact.title}"
            for artifact in artifacts[-8:]
        )

    def _render_settings_body(self) -> str:
        providers = self.runtime.list_provider_ids()
        agents = self.runtime.list_agent_ids()
        workflows = self.runtime.list_workflow_ids()
        orchestrator_status = self.runtime.agent_status_summary("orchestrator")

        provider_text = ", ".join(providers) if providers else "none"
        agent_text = ", ".join(agents)
        workflow_text = ", ".join(workflows)

        return (
            f"Config: {self.runtime.paths.config_path}\n"
            f"Agents Dir: {self.runtime.paths.agents_dir}\n"
            f"Workflows Dir: {self.runtime.paths.workflows_dir}\n"
            f"Orchestrator: {orchestrator_status}\n"
            f"Providers: {provider_text}\n"
            f"Agents: {agent_text}\n"
            f"Workflows: {workflow_text}"
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message_body = event.value.strip()
        if not message_body:
            event.input.value = ""
            return

        event.input.disabled = True
        try:
            await self.runtime.send_user_message_to_orchestrator(
                body=message_body,
                created_at=int(time.time()),
            )
            self._refresh_panels()
            event.input.value = ""
        finally:
            event.input.disabled = False

    def action_next_thread(self) -> None:
        self._cycle_thread(1)

    def action_previous_thread(self) -> None:
        self._cycle_thread(-1)

    def action_edit_orchestrator_definition(self) -> None:
        initial_text = self.runtime.read_agent_definition_text("orchestrator")
        self.push_screen(
            DefinitionEditorScreen(
                title="Edit Orchestrator Definition",
                initial_text=initial_text,
                on_save=self._save_orchestrator_definition,
            )
        )

    def _cycle_thread(self, direction: int) -> None:
        threads = self.runtime.list_threads()
        if not threads:
            return

        thread_ids = [thread.id for thread in threads]
        try:
            current_index = thread_ids.index(self.selected_thread_id)
        except ValueError:
            current_index = 0

        self.selected_thread_id = thread_ids[(current_index + direction) % len(thread_ids)]
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        self.query_one("#selected-thread", Panel).set_body(self._render_selected_thread_body())

    def _save_orchestrator_definition(self, text: str) -> None:
        self.runtime.save_agent_definition_text(
            agent_id="orchestrator",
            text=text,
            created_at=int(time.time()),
        )
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        self.query_one("#tasks", Panel).set_body(self._render_tasks_body())
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        self.query_one("#activity", Panel).set_body(self._render_activity_body())
        self.query_one("#main-chat", Panel).set_body(self._render_main_chat_body())
        self.query_one("#selected-thread", Panel).set_body(self._render_selected_thread_body())
        self.query_one("#artifacts", Panel).set_body(self._render_artifacts_body())
        self.query_one("#approvals", Panel).set_body(self._render_approvals_body())
        self.query_one("#memory", Panel).set_body(self._render_memory_body())
        self.query_one("#settings", Panel).set_body(self._render_settings_body())
