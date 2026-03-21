from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from ergon_studio.app_config import ProxyAppConfig, save_app_config
from ergon_studio.server_control import ProxyServerController

_VALID_DEFINITION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class DefinitionListItem(ListItem):
    def __init__(self, path: Path) -> None:
        super().__init__(Label(path.stem))
        self.path = path


class DefinitionEditor(Static):
    def __init__(
        self,
        *,
        title: str,
        definition_kind: str,
        directory: Path,
        on_changed: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.title = title
        self.definition_kind = definition_kind
        self.directory = directory
        self._on_changed = on_changed
        self._selected_path: Path | None = None

    def compose(self) -> ComposeResult:
        kind = self.definition_kind
        yield Horizontal(
            Vertical(
                Label(f"{self.title}"),
                ListView(id=f"{kind}-list"),
                Input(placeholder=f"new {kind} id", id=f"{kind}-new-name"),
                Horizontal(
                    Button("Add", id=f"{kind}-add"),
                    Button("Delete", id=f"{kind}-delete"),
                    Button("Save", id=f"{kind}-save"),
                ),
                Static("", id=f"{kind}-message"),
                id=f"{kind}-sidebar",
            ),
            Vertical(
                Label(f"{self.title} Markdown"),
                TextArea(id=f"{kind}-editor"),
                id=f"{kind}-editor-pane",
            ),
        )

    def on_mount(self) -> None:
        self.refresh_list(select_first=True)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, DefinitionListItem):
            self._load_path(item.path)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, DefinitionListItem):
            self._load_path(item.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == f"{self.definition_kind}-add":
            self._add_definition()
            return
        if button_id == f"{self.definition_kind}-delete":
            self._delete_definition()
            return
        if button_id == f"{self.definition_kind}-save":
            self._save_definition()

    def refresh_list(
        self,
        *,
        selected_filename: str | None = None,
        select_first: bool = False,
    ) -> None:
        list_view = self.query_one(f"#{self.definition_kind}-list", ListView)
        list_view.clear()
        paths = sorted(self.directory.glob("*.md"))
        for path in paths:
            list_view.append(DefinitionListItem(path))
        if not paths:
            self._selected_path = None
            self.query_one(f"#{self.definition_kind}-editor", TextArea).load_text("")
            return
        target_name = selected_filename
        if target_name is None and self._selected_path is not None:
            target_name = self._selected_path.name
        selected_index = 0
        if target_name is not None:
            for index, path in enumerate(paths):
                if path.name == target_name:
                    selected_index = index
                    break
        elif not select_first:
            selected_index = 0
        list_view.index = selected_index
        self._load_path(paths[selected_index])

    def _load_path(self, path: Path) -> None:
        self._selected_path = path
        editor = self.query_one(f"#{self.definition_kind}-editor", TextArea)
        editor.load_text(path.read_text(encoding="utf-8"))
        self._set_message(f"Loaded {path.name}")

    def _add_definition(self) -> None:
        new_name_input = self.query_one(f"#{self.definition_kind}-new-name", Input)
        raw_name = new_name_input.value.strip()
        if not raw_name:
            self._set_message(f"Enter a {self.definition_kind} id first")
            return
        if not _VALID_DEFINITION_ID.fullmatch(raw_name):
            self._set_message(
                "Use only letters, numbers, dashes, and underscores in ids"
            )
            return
        path = self.directory / f"{raw_name}.md"
        if path.exists():
            self._set_message(f"{path.name} already exists")
            return
        path.write_text(
            _new_definition_template(self.definition_kind, raw_name),
            encoding="utf-8",
        )
        new_name_input.value = ""
        self.refresh_list(selected_filename=path.name)
        self._set_message(f"Created {path.name}")
        self._on_changed(f"Created {path.name}")

    def _delete_definition(self) -> None:
        path = self._selected_path
        if path is None:
            self._set_message(f"No {self.definition_kind} selected")
            return
        if self.definition_kind == "agent" and path.stem == "orchestrator":
            self._set_message("The orchestrator definition is required")
            return
        deleted_name = path.name
        path.unlink()
        self.refresh_list(select_first=True)
        self._set_message(f"Deleted {deleted_name}")
        self._on_changed(f"Deleted {deleted_name}")

    def _save_definition(self) -> None:
        path = self._selected_path
        if path is None:
            self._set_message(f"No {self.definition_kind} selected")
            return
        editor = self.query_one(f"#{self.definition_kind}-editor", TextArea)
        path.write_text(editor.text, encoding="utf-8")
        self._set_message(f"Saved {path.name}")
        self._on_changed(f"Saved {path.name}")

    def _set_message(self, message: str) -> None:
        self.query_one(f"#{self.definition_kind}-message", Static).update(message)


class ProxyConfigApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #server-status {
        padding: 0 1;
        height: 2;
    }

    TabbedContent {
        height: 1fr;
    }

    #endpoint-form,
    #agent-sidebar,
    #workflow-sidebar,
    #agent-editor-pane,
    #workflow-editor-pane {
        padding: 1;
    }

    #agent-sidebar,
    #workflow-sidebar {
        width: 30;
    }

    TextArea {
        height: 1fr;
    }

    ListView {
        height: 1fr;
    }

    Button {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        app_dir: Path,
        definitions_dir: Path,
        initial_config: ProxyAppConfig,
        server_controller: ProxyServerController | None = None,
    ) -> None:
        super().__init__()
        self.app_dir = app_dir
        self.definitions_dir = definitions_dir
        self.config_path = app_dir / "config.json"
        self.config = initial_config
        self.server_controller = server_controller or ProxyServerController()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="server-status")
        with TabbedContent(initial="endpoint-tab"):
            with TabPane("Endpoint", id="endpoint-tab"):
                yield self._compose_endpoint_tab()
            with TabPane("Agents", id="agents-tab"):
                yield DefinitionEditor(
                    title="Agent Roles",
                    definition_kind="agent",
                    directory=self.definitions_dir / "agents",
                    on_changed=self._handle_definition_change,
                )
            with TabPane("Workflows", id="workflows-tab"):
                yield DefinitionEditor(
                    title="Workflows",
                    definition_kind="workflow",
                    directory=self.definitions_dir / "workflows",
                    on_changed=self._handle_definition_change,
                )
        yield Footer()

    def _compose_endpoint_tab(self) -> Vertical:
        return Vertical(
            Label("OpenAI-compatible endpoint URL"),
            Input(value=self.config.upstream_base_url, id="endpoint-url"),
            Label("API key"),
            Input(value=self.config.upstream_api_key, password=True, id="endpoint-key"),
            Static(
                "If the API key is blank, ergon will use `not-needed` for the upstream "
                "client.",
                id="endpoint-help",
            ),
            Button("Save Endpoint", id="save-endpoint"),
            Static("", id="endpoint-message"),
            id="endpoint-form",
        )

    def on_mount(self) -> None:
        self._restart_server("Ready")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-endpoint":
            self._save_endpoint()

    def _save_endpoint(self) -> None:
        self.config = ProxyAppConfig(
            upstream_base_url=self.query_one("#endpoint-url", Input).value.strip(),
            upstream_api_key=self.query_one("#endpoint-key", Input).value,
            host=self.config.host,
            port=self.config.port,
            instruction_role=self.config.instruction_role,
            disable_tool_calling=self.config.disable_tool_calling,
        )
        save_app_config(self.config_path, self.config)
        self.query_one("#endpoint-message", Static).update("Saved endpoint settings")
        self._restart_server("Endpoint settings saved")

    def _handle_definition_change(self, message: str) -> None:
        self._restart_server(message)

    def _restart_server(self, reason: str) -> None:
        try:
            status = self.server_controller.start(
                config=self.config,
                definitions_dir=self.definitions_dir,
            )
            self.query_one("#server-status", Static).update(
                _status_text(status.message, status.url, reason)
            )
        except Exception as exc:
            self.server_controller.stop()
            self.query_one("#server-status", Static).update(
                f"Server error: {exc} | {reason}"
            )


def run_config_tui(
    *,
    app_dir: Path,
    definitions_dir: Path,
    initial_config: ProxyAppConfig,
) -> int:
    app = ProxyConfigApp(
        app_dir=app_dir,
        definitions_dir=definitions_dir,
        initial_config=initial_config,
    )
    app.run()
    return 0


def _new_definition_template(definition_kind: str, definition_id: str) -> str:
    if definition_kind == "agent":
        return (
            "---\n"
            f"id: {definition_id}\n"
            f"role: {definition_id}\n"
            "temperature: 0\n"
            "---\n\n"
            "## Identity\n"
            f"You are the {definition_id} specialist.\n\n"
            "## Responsibilities\n"
            "Describe the role here.\n"
        )
    return (
        "---\n"
        f"id: {definition_id}\n"
        f"name: {definition_id.replace('-', ' ').title()}\n"
        "orchestration: sequential\n"
        "steps:\n"
        "  - orchestrator\n"
        "---\n\n"
        "## Purpose\n"
        "Describe when this workflow should be used.\n"
    )


def _status_text(message: str, url: str | None, reason: str) -> str:
    if url:
        return f"{message} at {url} | {reason}"
    return f"{message} | {reason}"
