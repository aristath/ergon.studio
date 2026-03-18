from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from ergon_studio.runtime import RuntimeContext


class Panel(Static):
    def __init__(self, title: str, body: str, *, panel_id: str, classes: str | None = None) -> None:
        super().__init__(self._render_panel(title, body), id=panel_id, classes=classes)

    @staticmethod
    def _render_panel(title: str, body: str) -> str:
        return f"[b]{title}[/b]\n{body}"


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
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        super().__init__()
        self.runtime = runtime

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="left-sidebar"):
                yield Panel("Tasks", "Task tree will appear here.", panel_id="tasks", classes="panel")
                yield Panel("Threads", "Thread list will appear here.", panel_id="threads", classes="panel")
                yield Panel("Activity", "Workflow activity will appear here.", panel_id="activity", classes="panel")
            with Vertical(id="center-column"):
                yield Panel(
                    "Main Chat",
                    (
                        f"Project UUID: {self.runtime.paths.project_uuid}\n"
                        f"Workspace: {self.runtime.paths.project_root}"
                    ),
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
        yield Footer()
