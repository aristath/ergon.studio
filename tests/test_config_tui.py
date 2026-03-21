from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, Static, TabbedContent, TextArea

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
                app.query_one("#proxy-host", Input).value = "0.0.0.0"
                app.query_one("#proxy-port", Input).value = "4310"
                app._save_endpoint()
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
            self.assertIn('"host": "0.0.0.0"', saved_config.read_text(encoding="utf-8"))
            self.assertIn('"port": 4310', saved_config.read_text(encoding="utf-8"))
            self.assertEqual(controller.start_calls[1][0].host, "0.0.0.0")
            self.assertEqual(controller.start_calls[1][0].port, 4310)

    async def test_tui_rejects_invalid_proxy_port(self) -> None:
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
                app.query_one("#proxy-port", Input).value = "not-a-port"
                app._save_endpoint()
                await pilot.pause()

                self.assertEqual(app.config.port, 4000)

            self.assertEqual(len(controller.start_calls), 1)

    async def test_tui_does_not_persist_endpoint_changes_when_restart_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            controller = _FakeController(fail_on_start=1)
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
                app.query_one("#endpoint-url", Input).value = (
                    "http://localhost:9999/v1"
                )
                app.query_one("#proxy-port", Input).value = "4310"
                app._save_endpoint()
                await pilot.pause()

                self.assertEqual(app.config.upstream_base_url, "http://localhost:8080/v1")
                self.assertEqual(app.config.port, 4000)
                self.assertIn(
                    "Could not apply endpoint settings",
                    str(app.query_one("#endpoint-message", Static).render()),
                )
                self.assertIn(
                    "server running at http://127.0.0.1:4000/v1",
                    str(app.query_one("#server-status", Static).render()),
                )

            saved_config = Path(temp_dir) / "config.json"
            if saved_config.exists():
                saved_text = saved_config.read_text(encoding="utf-8")
                self.assertNotIn("localhost:9999", saved_text)
            self.assertEqual(len(controller.start_calls), 2)

    async def test_tui_rolls_back_endpoint_when_config_save_fails(self) -> None:
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
                app.query_one("#endpoint-url", Input).value = (
                    "http://localhost:9999/v1"
                )
                with patch(
                    "ergon_studio.proxy.config_tui.save_app_config",
                    side_effect=OSError("disk full"),
                ):
                    app._save_endpoint()
                    await pilot.pause()

                self.assertEqual(app.config.upstream_base_url, "http://localhost:8080/v1")
                self.assertIn(
                    "Could not persist endpoint settings",
                    str(app.query_one("#endpoint-message", Static).render()),
                )
                self.assertIn(
                    "server running at http://127.0.0.1:4000/v1",
                    str(app.query_one("#server-status", Static).render()),
                )

            self.assertEqual(len(controller.start_calls), 3)

    async def test_navigation_does_not_discard_unsaved_definition_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            app = ProxyConfigApp(
                app_dir=workspace.app_dir,
                definitions_dir=workspace.definitions_dir,
                initial_config=ProxyAppConfig(
                    upstream_base_url="http://localhost:8080/v1"
                ),
                server_controller=_FakeController(),
            )

            async with app.run_test() as pilot:
                tabs = app.query_one(TabbedContent)
                tabs.active = "agents-tab"
                await pilot.pause()

                editor_widget = app.query(DefinitionEditor).first()
                original_path = editor_widget._selected_path
                self.assertIsNotNone(original_path)
                text_area = app.query_one("#agent-editor", TextArea)
                text_area.load_text("dirty")
                editor_widget._handle_list_navigation(
                    workspace.agents_dir / "orchestrator.md"
                )
                await pilot.pause()

                self.assertEqual(
                    editor_widget._selected_path,
                    original_path,
                )
                self.assertEqual(text_area.text, "dirty")

    async def test_invalid_definition_edits_do_not_restart_or_replace_files(
        self,
    ) -> None:
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

            original = (workspace.agents_dir / "architect.md").read_text(
                encoding="utf-8"
            )

            async with app.run_test() as pilot:
                tabs = app.query_one(TabbedContent)
                tabs.active = "agents-tab"
                await pilot.pause()

                editor_widget = app.query(DefinitionEditor).first()
                editor_widget._handle_list_navigation(
                    workspace.agents_dir / "architect.md"
                )
                text_area = app.query_one("#agent-editor", TextArea)
                text_area.load_text("---\nrole: architect\n")
                editor_widget._save_definition()
                await pilot.pause()

                editor_widget._revert_definition()
                await pilot.pause()
                editor_widget._delete_definition()
                await pilot.pause()

            self.assertEqual(len(controller.start_calls), 1)
            self.assertEqual(
                (workspace.agents_dir / "architect.md").read_text(encoding="utf-8"),
                original,
            )

    async def test_definition_save_failure_does_not_restart_server(self) -> None:
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
                tabs = app.query_one(TabbedContent)
                tabs.active = "agents-tab"
                await pilot.pause()

                editor_widget = app.query(DefinitionEditor).first()
                editor_widget._handle_list_navigation(
                    workspace.agents_dir / "architect.md"
                )
                app.query_one("#agent-editor", TextArea).load_text(
                    "---\n"
                    "id: architect\n"
                    "role: architect\n"
                    "---\n\n"
                    "## Identity\n"
                    "Architect.\n"
                )
                with patch(
                    "ergon_studio.proxy.config_tui.atomic_write_text",
                    side_effect=OSError("disk full"),
                ):
                    editor_widget._save_definition()
                    await pilot.pause()

                self.assertIn(
                    "could not persist architect.md",
                    str(app.query_one("#agent-message", Static).render()),
                )

            self.assertEqual(len(controller.start_calls), 1)


class _FakeController:
    def __init__(self, *, fail_on_start: int | None = None) -> None:
        self.start_calls: list[tuple[ProxyAppConfig, Path]] = []
        self.fail_on_start = fail_on_start
        self._status = ProxyServerStatus(
            running=False,
            message="server stopped",
            url=None,
        )

    @property
    def status(self) -> ProxyServerStatus:
        return self._status

    def start(
        self,
        *,
        config: ProxyAppConfig,
        definitions_dir: Path,
    ) -> ProxyServerStatus:
        self.start_calls.append((config, definitions_dir))
        if (
            self.fail_on_start is not None
            and len(self.start_calls) > self.fail_on_start
        ):
            raise OSError("address already in use")
        self._status = ProxyServerStatus(
            running=True,
            message="server running",
            url="http://127.0.0.1:4000/v1",
        )
        return self._status

    def stop(self) -> None:
        self._status = ProxyServerStatus(
            running=False,
            message="server stopped",
            url=None,
        )
        return None
