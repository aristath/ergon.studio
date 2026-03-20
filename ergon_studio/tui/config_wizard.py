"""Interactive configuration wizard for ergon.studio."""

from __future__ import annotations

import asyncio
import json
import urllib.error
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Static

from ergon_studio.provider_health import probe_endpoint_models
from ergon_studio.runtime import RuntimeContext


class ProviderEditorScreen(ModalScreen[dict[str, Any] | None]):
    """Add or edit a single provider."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    ProviderEditorScreen {
        align: center middle;
    }

    #provider-editor-container {
        width: 70;
        max-height: 80%;
        border: round $accent;
        background: transparent;
        padding: 1 2;
    }

    #provider-editor-container Label {
        margin: 1 0 0 0;
    }

    #provider-editor-container Input {
        margin: 0 0 0 0;
    }

    #model-list {
        height: auto;
        max-height: 12;
        margin: 0 0 0 0;
        display: none;
    }

    #model-list.visible {
        display: block;
    }

    #manual-model-input {
        display: none;
    }

    #manual-model-input.visible {
        display: block;
    }

    #provider-error {
        color: $error;
        height: auto;
        margin: 1 0 0 0;
    }

    #provider-status {
        color: $text-muted;
        height: auto;
        margin: 1 0 0 0;
    }

    .button-row {
        height: auto;
        margin: 1 0 0 0;
    }

    .button-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(
        self,
        *,
        name: str = "",
        base_url: str = "",
        api_key: str = "",
        model: str = "",
    ) -> None:
        super().__init__()
        self._initial_name = name
        self._initial_url = base_url
        self._initial_key = api_key
        self._initial_model = model
        self._selected_model: str | None = model or None
        self._selected_context_length: int | None = None
        self._fetched_models: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="provider-editor-container"):
            yield Static("[b]Provider Configuration[/b]")

            yield Label("Provider name")
            yield Input(
                value=self._initial_name,
                placeholder="e.g. local, cloud, openrouter",
                id="name-input",
            )

            yield Label("Endpoint URL")
            yield Input(
                value=self._initial_url,
                placeholder="e.g. http://localhost:11434/v1",
                id="endpoint-input",
            )

            yield Label("API Key (optional)")
            yield Input(
                value=self._initial_key,
                placeholder="Leave empty for local endpoints",
                password=True,
                id="api-key-input",
            )

            with Horizontal(classes="button-row"):
                yield Button("Fetch Models", id="fetch-btn", variant="primary")
                yield Button("Manual Entry", id="manual-btn")

            yield Static("", id="provider-status")
            yield Static("", id="provider-error")

            yield OptionList(id="model-list")

            yield Label("Or type model name manually")
            yield Input(
                value=self._initial_model,
                placeholder="e.g. qwen3:8b",
                id="manual-model-input",
            )

            with Horizontal(classes="button-row"):
                yield Button("Save", id="save-btn", variant="success")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        if self._initial_model:
            manual = self.query_one("#manual-model-input", Input)
            manual.add_class("visible")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fetch-btn":
            await self._fetch_models()
        elif event.button.id == "manual-btn":
            self.query_one("#manual-model-input", Input).add_class("visible")
            self.query_one("#manual-model-input", Input).focus()
        elif event.button.id == "save-btn":
            self._save()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        prompt = str(event.option.prompt)
        # Extract model name (before the context length annotation)
        self._selected_model = prompt.split("  ")[0].strip()
        # Find context_length from fetched models
        for m in self._fetched_models:
            if m["id"] == self._selected_model and "context_length" in m:
                self._selected_context_length = m["context_length"]
                break
        self.query_one("#provider-status", Static).update(
            f"[green]Selected:[/green] {self._selected_model}"
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def _fetch_models(self) -> None:
        url = self.query_one("#endpoint-input", Input).value.strip()
        api_key = self.query_one("#api-key-input", Input).value.strip() or None
        error_widget = self.query_one("#provider-error", Static)
        status_widget = self.query_one("#provider-status", Static)
        model_list = self.query_one("#model-list", OptionList)

        if not url:
            error_widget.update("[red]Please enter an endpoint URL[/red]")
            return

        error_widget.update("")
        status_widget.update("[dim]Fetching models...[/dim]")

        try:
            models = await asyncio.to_thread(probe_endpoint_models, url, api_key)
        except urllib.error.URLError as exc:
            error_widget.update(f"[red]Could not reach endpoint: {exc.reason}[/red]")
            status_widget.update("")
            return
        except Exception as exc:
            error_widget.update(f"[red]Error: {exc}[/red]")
            status_widget.update("")
            return

        if not models:
            error_widget.update("[yellow]No models found. Use manual entry.[/yellow]")
            status_widget.update("")
            self.query_one("#manual-model-input", Input).add_class("visible")
            return

        self._fetched_models = models
        model_list.clear_options()
        for m in models:
            label = m["id"]
            if "context_length" in m:
                ctx_k = m["context_length"] // 1024
                label += f"  [dim]({ctx_k}k ctx)[/dim]"
            model_list.add_option(label)
        model_list.add_class("visible")
        status_widget.update(f"[green]Found {len(models)} models. Click to select.[/green]")

    def _save(self) -> None:
        name = self.query_one("#name-input", Input).value.strip()
        url = self.query_one("#endpoint-input", Input).value.strip()
        api_key = self.query_one("#api-key-input", Input).value.strip()
        manual_model = self.query_one("#manual-model-input", Input).value.strip()
        error_widget = self.query_one("#provider-error", Static)

        model = manual_model or self._selected_model

        if not name:
            error_widget.update("[red]Provider name is required[/red]")
            return
        if not url:
            error_widget.update("[red]Endpoint URL is required[/red]")
            return
        if not model:
            error_widget.update("[red]Please select or enter a model[/red]")
            return

        result: dict[str, Any] = {
            "name": name,
            "type": "openai_chat",
            "model": model,
            "base_url": url,
        }
        if api_key:
            result["api_key"] = api_key
        if self._selected_context_length is not None:
            result["context_length"] = self._selected_context_length

        self.dismiss(result)


class RoleAssignmentScreen(ModalScreen[dict[str, str] | None]):
    """Assign providers to agent roles."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    RoleAssignmentScreen {
        align: center middle;
    }

    #role-assignment-container {
        width: 60;
        max-height: 80%;
        border: round $accent;
        background: transparent;
        padding: 1 2;
    }

    .role-row {
        height: 3;
        margin: 0;
    }

    .role-row Label {
        width: 16;
        padding: 1 0 0 0;
    }

    .role-row Select {
        width: 1fr;
    }

    .button-row {
        height: auto;
        margin: 1 0 0 0;
    }

    .button-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(
        self,
        roles: list[str],
        provider_names: list[str],
        current_assignments: dict[str, str],
    ) -> None:
        super().__init__()
        self._roles = roles
        self._provider_names = provider_names
        self._current = current_assignments

    def compose(self) -> ComposeResult:
        options = [(name, name) for name in self._provider_names]
        default = self._provider_names[0] if self._provider_names else Select.BLANK

        with VerticalScroll(id="role-assignment-container"):
            yield Static("[b]Role Assignments[/b]\nAssign a provider to each agent role.")

            for role in self._roles:
                current = self._current.get(role)
                value = current if current in self._provider_names else default
                with Horizontal(classes="role-row"):
                    yield Label(role)
                    yield Select(
                        options,
                        value=value,
                        allow_blank=False,
                        id=f"role-select-{role}",
                    )

            with Horizontal(classes="button-row"):
                yield Button("Save", id="save-btn", variant="success")
                yield Button("Cancel", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            assignments: dict[str, str] = {}
            for role in self._roles:
                select = self.query_one(f"#role-select-{role}", Select)
                if select.value and select.value != Select.BLANK:
                    assignments[role] = str(select.value)
            self.dismiss(assignments)
        elif event.button.id == "cancel-btn":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfigWizardScreen(ModalScreen[None]):
    """Main configuration wizard showing providers and role assignments."""

    BINDINGS = [("escape", "dismiss_wizard", "Close")]

    CSS = """
    ConfigWizardScreen {
        align: center middle;
    }

    #config-wizard-container {
        width: 70;
        max-height: 80%;
        border: round $accent;
        background: transparent;
        padding: 1 2;
    }

    #provider-list {
        height: auto;
        max-height: 12;
        margin: 1 0;
    }

    .button-row {
        height: auto;
        margin: 1 0 0 0;
    }

    .button-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        super().__init__()
        self.runtime = runtime

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="config-wizard-container"):
            yield Static("[b]Configuration[/b]")

            yield Label("Providers")
            yield OptionList(id="provider-list")

            with Horizontal(classes="button-row"):
                yield Button("Add Provider", id="add-btn", variant="primary")
                yield Button("Edit Assignments", id="assign-btn")
                yield Button("Done", id="done-btn", variant="success")

    def on_mount(self) -> None:
        self._refresh_provider_list()

    def _refresh_provider_list(self) -> None:
        provider_list = self.query_one("#provider-list", OptionList)
        provider_list.clear_options()
        config = self._load_config()
        providers = config.get("providers", {})
        if not providers:
            provider_list.add_option("(no providers configured)")
        else:
            for name, details in providers.items():
                if isinstance(details, dict):
                    model = details.get("model", "?")
                    url = details.get("base_url", "?")
                    provider_list.add_option(f"{name}  {url}  [{model}]")
                else:
                    provider_list.add_option(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-btn":
            self.app.push_screen(
                ProviderEditorScreen(),
                callback=self._on_provider_saved,
            )
        elif event.button.id == "assign-btn":
            self._open_role_assignments()
        elif event.button.id == "done-btn":
            self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Edit an existing provider when clicked."""
        config = self._load_config()
        providers = config.get("providers", {})
        provider_names = list(providers.keys())
        idx = event.option_index
        if idx < len(provider_names):
            name = provider_names[idx]
            details = providers[name]
            if isinstance(details, dict):
                self.app.push_screen(
                    ProviderEditorScreen(
                        name=name,
                        base_url=str(details.get("base_url", "")),
                        api_key=str(details.get("api_key", "")),
                        model=str(details.get("model", "")),
                    ),
                    callback=self._on_provider_saved,
                )

    def _on_provider_saved(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        config = self._load_config()
        name = result.pop("name")
        config.setdefault("providers", {})[name] = result

        # Auto-assign if this is the first provider and no assignments exist
        assignments = config.get("role_assignments", {})
        if not assignments:
            roles = self.runtime.list_agent_ids()
            config["role_assignments"] = {role: name for role in roles}

        self._save_config(config)
        self._refresh_provider_list()

    def _open_role_assignments(self) -> None:
        config = self._load_config()
        providers = config.get("providers", {})
        provider_names = sorted(providers.keys())
        if not provider_names:
            return
        roles = self.runtime.list_agent_ids()
        current = config.get("role_assignments", {})
        self.app.push_screen(
            RoleAssignmentScreen(roles, provider_names, current),
            callback=self._on_assignments_saved,
        )

    def _on_assignments_saved(self, result: dict[str, str] | None) -> None:
        if result is None:
            return
        config = self._load_config()
        config["role_assignments"] = result
        self._save_config(config)

    def _load_config(self) -> dict[str, Any]:
        return json.loads(self.runtime.read_global_config_text())

    def _save_config(self, config: dict[str, Any]) -> None:
        self.runtime.save_global_config_text(
            text=json.dumps(config, indent=2, sort_keys=True),
        )

    def action_dismiss_wizard(self) -> None:
        self.dismiss(None)
