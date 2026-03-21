from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ergon_studio.app_config import ProxyAppConfig
from ergon_studio.server_control import ProxyServerController
from ergon_studio.workspace import ensure_workspace


class ServerControlTests(unittest.TestCase):
    def test_start_returns_message_when_upstream_url_missing(self) -> None:
        controller = ProxyServerController()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            status = controller.start(
                config=ProxyAppConfig(),
                definitions_dir=workspace.definitions_dir,
            )

        self.assertFalse(status.running)
        self.assertIn("set the upstream URL", status.message)

    def test_start_starts_proxy_thread_and_reports_url(self) -> None:
        controller = ProxyServerController()
        fake_handle = type("Handle", (), {"port": 4310, "close": lambda self: None})()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            with patch(
                "ergon_studio.server_control.start_proxy_server_in_thread",
                return_value=fake_handle,
            ):
                status = controller.start(
                    config=ProxyAppConfig(
                        upstream_base_url="http://localhost:8080/v1",
                        host="0.0.0.0",
                    ),
                    definitions_dir=workspace.definitions_dir,
                )

        self.assertTrue(status.running)
        self.assertEqual(status.url, "http://0.0.0.0:4310/v1")

    def test_start_keeps_existing_server_running_when_new_registry_is_invalid(
        self,
    ) -> None:
        controller = ProxyServerController()

        class _Handle:
            port = 4310

            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        fake_handle = _Handle()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            with patch(
                "ergon_studio.server_control.start_proxy_server_in_thread",
                return_value=fake_handle,
            ):
                controller.start(
                    config=ProxyAppConfig(
                        upstream_base_url="http://localhost:8080/v1"
                    ),
                    definitions_dir=workspace.definitions_dir,
                )

                (workspace.agents_dir / "coder.md").unlink()

                with self.assertRaisesRegex(ValueError, "unknown agents: coder"):
                    controller.start(
                        config=ProxyAppConfig(
                            upstream_base_url="http://localhost:8080/v1"
                        ),
                        definitions_dir=workspace.definitions_dir,
                    )

        self.assertTrue(controller.status.running)
        self.assertFalse(fake_handle.closed)
