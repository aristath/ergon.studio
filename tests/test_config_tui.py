from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from textual.widgets import Input, Static, TabbedContent

from ergon_studio.app_config import ProxyAppConfig
from ergon_studio.proxy.config_tui import DefinitionEditor, ProxyConfigApp
from ergon_studio.server_control import ProxyServerStatus
from ergon_studio.workspace import ensure_workspace


class ConfigTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_tui_shows_tabs_and_server_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            controller = _FakeController()
            app = ProxyConfigApp(
                app_dir=workspace.app_dir,
                definitions_dir=workspace.definitions_dir,
                initial_config=ProxyAppConfig(
                    upstream_base_url="http://localhost:8080/v1"
                ),
                server_controller=controller,
            )

            async with app.run_test() as pilot:
                await pilot.pause()
                tabs = app.query_one(TabbedContent)
                self.assertEqual(tabs.active, "endpoint-tab")
                status = app.query_one("#server-status", Static)
                self.assertIn("server running", str(status.render()))
                self.assertEqual(len(controller.start_calls), 1)

    async def test_tui_can_save_endpoint_and_add_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            controller = _FakeController()
            app = ProxyConfigApp(
                app_dir=workspace.app_dir,
                definitions_dir=workspace.definitions_dir,
                initial_config=ProxyAppConfig(),
                server_controller=controller,
            )

            async with app.run_test() as pilot:
                app.query_one("#endpoint-url", Input).value = (
                    "http://localhost:8080/v1"
                )
                app.query_one("#endpoint-key", Input).value = "secret"
                await pilot.click("#save-endpoint")
                await pilot.pause()

                tabs = app.query_one(TabbedContent)
                tabs.active = "agents-tab"
                await pilot.pause()
                app.query_one("#agent-new-name", Input).value = "tester"
                app.query(DefinitionEditor).first()._add_definition()
                await pilot.pause()

            self.assertEqual(len(controller.start_calls), 3)
            self.assertEqual(
                (workspace.agents_dir / "tester.md").exists(),
                True,
            )
            saved_config = Path(temp_dir) / "config.json"
            self.assertIn("secret", saved_config.read_text(encoding="utf-8"))


class _FakeController:
    def __init__(self) -> None:
        self.start_calls: list[tuple[ProxyAppConfig, Path]] = []

    def start(
        self,
        *,
        config: ProxyAppConfig,
        definitions_dir: Path,
    ) -> ProxyServerStatus:
        self.start_calls.append((config, definitions_dir))
        return ProxyServerStatus(
            running=True,
            message="server running",
            url="http://127.0.0.1:4000/v1",
        )

    def stop(self) -> None:
        return None
