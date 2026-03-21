from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from textual import work
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
from textual.worker import Worker, WorkerState

from ergon_studio.app_config import (
    ProxyAppConfig,
    save_app_config,
    validate_proxy_host,
    validate_proxy_port,
)
from ergon_studio.file_ops import atomic_write_text
from ergon_studio.registry import load_registry
from ergon_studio.server_control import ProxyServerController, ProxyServerStatus
from ergon_studio.upstream import UpstreamSettings

_VALID_DEFINITION_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class DefinitionMutation:
    action: str
    path: Path
    content: str | None = None


@dataclass(frozen=True)
class DefinitionRollback:
    restore_content: str | None = None
    delete_path: bool = False


@dataclass(frozen=True)
class ServerUpdateResult:
    status: ProxyServerStatus
    reason: str


@dataclass(frozen=True)
class EndpointSaveResult:
    status: ProxyServerStatus
    config: ProxyAppConfig
    endpoint_message: str
    server_reason: str
    applied: bool


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
        apply_mutation: Callable[[DefinitionMutation], str],
    ) -> None:
        super().__init__()
        self.title = title
        self.definition_kind = definition_kind
        self.directory = directory
        self._apply_mutation = apply_mutation
        self._selected_path: Path | None = None
        self._loaded_text = ""
        self._suspend_list_events = False

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
                    Button("Revert", id=f"{kind}-revert"),
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
        if self._suspend_list_events:
            return
        item = event.item
        if isinstance(item, DefinitionListItem):
            self._handle_list_navigation(item.path)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._suspend_list_events:
            return
        item = event.item
        if isinstance(item, DefinitionListItem):
            self._handle_list_navigation(item.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == f"{self.definition_kind}-add":
            self._add_definition()
            return
        if button_id == f"{self.definition_kind}-delete":
            self._delete_definition()
            return
        if button_id == f"{self.definition_kind}-revert":
            self._revert_definition()
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
            self._loaded_text = ""
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
        loaded_text = path.read_text(encoding="utf-8")
        editor.load_text(loaded_text)
        self._loaded_text = loaded_text
        self._set_message(f"Loaded {path.name}")

    def _handle_list_navigation(self, path: Path) -> None:
        if path == self._selected_path:
            return
        if self._has_unsaved_changes():
            self._set_message("Unsaved changes. Save or Revert before switching.")
            self._restore_current_selection()
            return
        self._load_path(path)

    def _add_definition(self) -> None:
        if self._has_unsaved_changes():
            self._set_message("Unsaved changes. Save or Revert before creating.")
            return
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
        try:
            message = self._apply_mutation(
                DefinitionMutation(
                    action="create",
                    path=path,
                    content=_new_definition_template(self.definition_kind, raw_name),
                )
            )
        except ValueError as exc:
            self._set_message(str(exc))
            return
        new_name_input.value = ""
        self.refresh_list(selected_filename=path.name)
        self._set_message(message)

    def _delete_definition(self) -> None:
        path = self._selected_path
        if path is None:
            self._set_message(f"No {self.definition_kind} selected")
            return
        if self._has_unsaved_changes():
            self._set_message("Unsaved changes. Save or Revert before deleting.")
            return
        if self.definition_kind == "agent" and path.stem == "orchestrator":
            self._set_message("The orchestrator definition is required")
            return
        try:
            message = self._apply_mutation(
                DefinitionMutation(action="delete", path=path)
            )
        except ValueError as exc:
            self._set_message(str(exc))
            return
        self.refresh_list(select_first=True)
        self._set_message(message)

    def _save_definition(self) -> None:
        path = self._selected_path
        if path is None:
            self._set_message(f"No {self.definition_kind} selected")
            return
        editor = self.query_one(f"#{self.definition_kind}-editor", TextArea)
        try:
            message = self._apply_mutation(
                DefinitionMutation(action="save", path=path, content=editor.text)
            )
        except ValueError as exc:
            self._set_message(str(exc))
            return
        self._loaded_text = editor.text
        self._set_message(message)

    def _revert_definition(self) -> None:
        path = self._selected_path
        if path is None:
            self._set_message(f"No {self.definition_kind} selected")
            return
        self._load_path(path)
        self._set_message(f"Reverted {path.name}")

    def _set_message(self, message: str) -> None:
        self.query_one(f"#{self.definition_kind}-message", Static).update(message)

    def _has_unsaved_changes(self) -> bool:
        path = self._selected_path
        if path is None:
            return False
        editor = self.query_one(f"#{self.definition_kind}-editor", TextArea)
        return editor.text != self._loaded_text

    def _restore_current_selection(self) -> None:
        path = self._selected_path
        if path is None:
            return
        paths = sorted(self.directory.glob("*.md"))
        for index, candidate in enumerate(paths):
            if candidate != path:
                continue
            list_view = self.query_one(f"#{self.definition_kind}-list", ListView)
            self._suspend_list_events = True
            list_view.index = index
            self._suspend_list_events = False
            return


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
    #workroom-sidebar,
    #agent-editor-pane,
    #workroom-editor-pane {
        padding: 1;
    }

    #agent-sidebar,
    #workroom-sidebar {
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
        self._server_update_in_progress = False

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
                    apply_mutation=self._apply_definition_mutation,
                )
            with TabPane("Workroom Templates", id="workrooms-tab"):
                yield DefinitionEditor(
                    title="Workroom Templates",
                    definition_kind="workroom",
                    directory=self.definitions_dir / "workrooms",
                    apply_mutation=self._apply_definition_mutation,
                )
        yield Footer()

    def _compose_endpoint_tab(self) -> Vertical:
        return Vertical(
            Label("OpenAI-compatible endpoint URL"),
            Input(value=self.config.upstream_base_url, id="endpoint-url"),
            Label("API key"),
            Input(value=self.config.upstream_api_key, password=True, id="endpoint-key"),
            Label("Proxy bind host"),
            Input(value=self.config.host, id="proxy-host"),
            Label("Proxy bind port"),
            Input(value=str(self.config.port), id="proxy-port"),
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
        self._begin_background_restart("Ready")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-endpoint":
            self._save_endpoint()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "server-restart":
            self._handle_server_restart_worker(event)
            return
        if event.worker.name == "endpoint-save":
            self._handle_endpoint_save_worker(event)

    def _save_endpoint(self) -> None:
        if self._server_update_in_progress:
            self.query_one("#endpoint-message", Static).update(
                "Server update in progress. Please wait."
            )
            return
        host_text = self.query_one("#proxy-host", Input).value.strip()
        port_text = self.query_one("#proxy-port", Input).value.strip()
        try:
            host = validate_proxy_host(host_text)
            port = validate_proxy_port(int(port_text))
        except ValueError as exc:
            self.query_one("#endpoint-message", Static).update(str(exc))
            return
        candidate = ProxyAppConfig(
            upstream_base_url=self.query_one("#endpoint-url", Input).value.strip(),
            upstream_api_key=self.query_one("#endpoint-key", Input).value,
            host=host,
            port=port,
            instruction_role=self.config.instruction_role,
            disable_tool_calling=self.config.disable_tool_calling,
        )
        self._server_update_in_progress = True
        self._set_endpoint_controls_disabled(True)
        self.query_one("#endpoint-message", Static).update(
            "Applying endpoint settings..."
        )
        self.query_one("#server-status", Static).update(
            _status_text(
                self.server_controller.status.message,
                self.server_controller.status.url,
                "Applying endpoint settings...",
            )
        )
        self._apply_endpoint_settings(candidate, self.config)

    def _apply_definition_mutation(self, mutation: DefinitionMutation) -> str:
        if self._server_update_in_progress:
            raise ValueError("Server update in progress. Please wait.")
        self._validate_definition_mutation(mutation)
        rollback = self._capture_definition_rollback(mutation)
        try:
            if mutation.action == "delete":
                mutation.path.unlink()
            else:
                atomic_write_text(mutation.path, mutation.content or "")
        except OSError as exc:
            raise ValueError(f"could not persist {mutation.path.name}: {exc}") from exc
        message = _mutation_message(mutation)
        try:
            self._restart_server(message)
        except ValueError:
            self._restore_definition_rollback(mutation.path, rollback)
            raise
        return message

    def _restart_server(self, reason: str) -> None:
        try:
            status = self.server_controller.start(
                config=self.config,
                definitions_dir=self.definitions_dir,
            )
        except Exception as exc:
            self._show_server_status(reason=f"{reason} | Server error: {exc}")
            raise ValueError(f"server restart failed: {exc}") from exc
        self.query_one("#server-status", Static).update(
            _status_text(status.message, status.url, reason)
        )

    def _show_server_status(self, *, reason: str) -> None:
        status = self.server_controller.status
        self.query_one("#server-status", Static).update(
            _status_text(status.message, status.url, reason)
        )

    def _begin_background_restart(self, reason: str) -> None:
        if self._server_update_in_progress:
            return
        self._server_update_in_progress = True
        self._set_endpoint_controls_disabled(True)
        self.query_one("#endpoint-message", Static).update("")
        self._show_server_status(reason=f"{reason} | Starting...")
        self._restart_server_background(reason)

    @work(
        name="server-restart",
        group="server-updates",
        exclusive=True,
        exit_on_error=False,
        thread=True,
    )
    def _restart_server_background(self, reason: str) -> ServerUpdateResult:
        try:
            status = self.server_controller.start(
                config=self.config,
                definitions_dir=self.definitions_dir,
            )
        except Exception as exc:
            return ServerUpdateResult(
                status=self.server_controller.status,
                reason=f"{reason} | Server error: {exc}",
            )
        return ServerUpdateResult(status=status, reason=reason)

    @work(
        name="endpoint-save",
        group="server-updates",
        exclusive=True,
        exit_on_error=False,
        thread=True,
    )
    def _apply_endpoint_settings(
        self,
        candidate: ProxyAppConfig,
        previous_config: ProxyAppConfig,
    ) -> EndpointSaveResult:
        try:
            status = self.server_controller.start(
                config=candidate,
                definitions_dir=self.definitions_dir,
            )
        except Exception as exc:
            return EndpointSaveResult(
                status=self.server_controller.status,
                config=previous_config,
                endpoint_message=f"Could not apply endpoint settings: {exc}",
                server_reason="Endpoint settings unchanged",
                applied=False,
            )
        try:
            save_app_config(self.config_path, candidate)
        except OSError as exc:
            rollback_reason = f"Could not persist endpoint settings: {exc}"
            try:
                rollback_status = self.server_controller.start(
                    config=previous_config,
                    definitions_dir=self.definitions_dir,
                )
                return EndpointSaveResult(
                    status=rollback_status,
                    config=previous_config,
                    endpoint_message=rollback_reason,
                    server_reason="Endpoint settings unchanged",
                    applied=False,
                )
            except Exception as rollback_exc:
                return EndpointSaveResult(
                    status=status,
                    config=previous_config,
                    endpoint_message=rollback_reason,
                    server_reason=(
                        f"{rollback_reason} | rollback failed: {rollback_exc}"
                    ),
                    applied=False,
                )
        return EndpointSaveResult(
            status=status,
            config=candidate,
            endpoint_message="Saved endpoint settings",
            server_reason="Endpoint settings saved",
            applied=True,
        )

    def _handle_server_restart_worker(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            self._server_update_in_progress = False
            self._set_endpoint_controls_disabled(False)
            error = event.worker.error
            self._show_server_status(reason=f"Server error: {error}")
            return
        if event.state != WorkerState.SUCCESS:
            return
        self._server_update_in_progress = False
        self._set_endpoint_controls_disabled(False)
        result = event.worker.result
        if not isinstance(result, ServerUpdateResult):
            return
        self.query_one("#server-status", Static).update(
            _status_text(result.status.message, result.status.url, result.reason)
        )

    def _handle_endpoint_save_worker(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            self._server_update_in_progress = False
            self._set_endpoint_controls_disabled(False)
            error = event.worker.error
            self.query_one("#endpoint-message", Static).update(
                f"Could not apply endpoint settings: {error}"
            )
            self._show_server_status(reason="Endpoint settings unchanged")
            return
        if event.state != WorkerState.SUCCESS:
            return
        self._server_update_in_progress = False
        self._set_endpoint_controls_disabled(False)
        result = event.worker.result
        if not isinstance(result, EndpointSaveResult):
            return
        self.config = result.config
        self.query_one("#endpoint-message", Static).update(result.endpoint_message)
        self.query_one("#server-status", Static).update(
            _status_text(
                result.status.message,
                result.status.url,
                result.server_reason,
            )
        )

    def _set_endpoint_controls_disabled(self, disabled: bool) -> None:
        self.query_one("#endpoint-url", Input).disabled = disabled
        self.query_one("#endpoint-key", Input).disabled = disabled
        self.query_one("#proxy-host", Input).disabled = disabled
        self.query_one("#proxy-port", Input).disabled = disabled
        self.query_one("#save-endpoint", Button).disabled = disabled

    def _validate_definition_mutation(self, mutation: DefinitionMutation) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_definitions_dir = Path(temp_dir) / "definitions"
            shutil.copytree(self.definitions_dir, temp_definitions_dir)
            candidate_path = temp_definitions_dir / mutation.path.relative_to(
                self.definitions_dir
            )
            if mutation.action == "delete":
                candidate_path.unlink()
            else:
                candidate_path.write_text(mutation.content or "", encoding="utf-8")
            try:
                load_registry(
                    temp_definitions_dir,
                    upstream=UpstreamSettings(
                        base_url=self.config.upstream_base_url or "http://unused",
                        api_key=self.config.upstream_api_key.strip() or None,
                        instruction_role=self.config.instruction_role.strip() or None,
                        tool_calling=not self.config.disable_tool_calling,
                    ),
                )
            except ValueError as exc:
                rendered = str(exc).replace(
                    str(temp_definitions_dir), str(self.definitions_dir)
                )
                raise ValueError(rendered) from exc

    def _capture_definition_rollback(
        self, mutation: DefinitionMutation
    ) -> DefinitionRollback:
        if mutation.path.exists():
            return DefinitionRollback(
                restore_content=mutation.path.read_text(encoding="utf-8")
            )
        return DefinitionRollback(delete_path=True)

    def _restore_definition_rollback(
        self, path: Path, rollback: DefinitionRollback
    ) -> None:
        if rollback.delete_path:
            if path.exists():
                path.unlink()
            return
        if rollback.restore_content is None:
            return
        atomic_write_text(path, rollback.restore_content)


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
            f"You are the {definition_id} in an AI software firm.\n\n"
            "## Responsibilities\n"
            "Describe what this role owns.\n\n"
            "## Working Style\n"
            "Describe how this role should think and collaborate.\n"
        )
    return (
        "---\n"
        f"id: {definition_id}\n"
        f"name: {definition_id.replace('-', ' ').title()}\n"
        "shape: staged\n"
        "steps:\n"
        "  - coder\n"
        "  - reviewer\n"
        "---\n\n"
        "## Purpose\n"
        "Describe what this workroom template is for.\n\n"
        "## Use When\n"
        "Describe when the lead developer should reach for it.\n\n"
        "## Notes\n"
        "Workroom templates are tactics for the orchestrator, not rigid scripts.\n"
    )


def _status_text(message: str, url: str | None, reason: str) -> str:
    if url:
        return f"{message} at {url} | {reason}"
    return f"{message} | {reason}"


def _mutation_message(mutation: DefinitionMutation) -> str:
    action_map = {
        "create": "Created",
        "save": "Saved",
        "delete": "Deleted",
    }
    return f"{action_map[mutation.action]} {mutation.path.name}"
