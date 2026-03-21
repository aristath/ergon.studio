from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from ergon_studio.app_config import ProxyAppConfig
from ergon_studio.server_control import ProxyServerController
from ergon_studio.workspace import ensure_workspace


@dataclass(frozen=True)
class _PreparedRuntime:
    host: str
    port: int
    registry: object
    core: object


class ServerControlTests(unittest.TestCase):
    def test_start_rejects_missing_upstream_url(self) -> None:
        controller = ProxyServerController()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "missing upstream base URL"):
                controller.start(
                    config=ProxyAppConfig(),
                    definitions_dir=workspace.definitions_dir,
                )

    def test_start_rejects_invalid_upstream_url(self) -> None:
        controller = ProxyServerController()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            with self.assertRaisesRegex(
                ValueError, "upstream base URL must be a valid http\\(s\\) URL"
            ):
                controller.start(
                    config=ProxyAppConfig(upstream_base_url="localhost:8080/v1"),
                    definitions_dir=workspace.definitions_dir,
                )

    def test_start_starts_proxy_thread_and_reports_url(self) -> None:
        controller = ProxyServerController()
        fake_handle = type("Handle", (), {"port": 4310, "close": lambda self: None})()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            prepared = _PreparedRuntime(
                host="0.0.0.0",
                port=4000,
                registry=object(),
                core=object(),
            )
            with (
                patch(
                    "ergon_studio.server_control.prepare_proxy_runtime",
                    return_value=prepared,
                ),
                patch(
                    "ergon_studio.server_control.start_proxy_server_in_thread",
                    return_value=fake_handle,
                ),
            ):
                status = controller.start(
                    config=ProxyAppConfig(
                        upstream_base_url="http://localhost:8080/v1",
                        host="0.0.0.0",
                    ),
                    definitions_dir=workspace.definitions_dir,
                )

        self.assertTrue(status.running)
        self.assertEqual(status.url, "http://127.0.0.1:4310/v1")

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
            prepared = _PreparedRuntime(
                host="127.0.0.1",
                port=4000,
                registry=object(),
                core=object(),
            )
            with (
                patch(
                    "ergon_studio.server_control.prepare_proxy_runtime",
                    side_effect=[prepared, ValueError("unknown agents: coder")],
                ),
                patch(
                    "ergon_studio.server_control.start_proxy_server_in_thread",
                    return_value=fake_handle,
                ),
            ):
                controller.start(
                    config=ProxyAppConfig(
                        upstream_base_url="http://localhost:8080/v1"
                    ),
                    definitions_dir=workspace.definitions_dir,
                )

                with self.assertRaisesRegex(ValueError, "unknown agents: coder"):
                    controller.start(
                        config=ProxyAppConfig(
                            upstream_base_url="http://localhost:8080/v1"
                        ),
                        definitions_dir=workspace.definitions_dir,
                    )

        self.assertTrue(controller.status.running)
        self.assertFalse(fake_handle.closed)

    def test_start_rolls_back_same_bind_restart_failures(self) -> None:
        controller = ProxyServerController()

        class _Handle:
            def __init__(self, port: int) -> None:
                self.port = port
                self.closed = False

            def close(self) -> None:
                self.closed = True

        first_handle = _Handle(4310)
        rollback_handle = _Handle(4310)

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            prepared = _PreparedRuntime(
                host="127.0.0.1",
                port=4000,
                registry=object(),
                core=object(),
            )
            with (
                patch(
                    "ergon_studio.server_control.prepare_proxy_runtime",
                    return_value=prepared,
                ),
                patch(
                    "ergon_studio.server_control.start_proxy_server_in_thread",
                    side_effect=[
                        first_handle,
                        OSError("address already in use"),
                        rollback_handle,
                    ],
                ),
            ):
                controller.start(
                    config=ProxyAppConfig(
                        upstream_base_url="http://localhost:8080/v1",
                    ),
                    definitions_dir=workspace.definitions_dir,
                )

                with self.assertRaisesRegex(OSError, "address already in use"):
                    controller.start(
                        config=ProxyAppConfig(
                            upstream_base_url="http://localhost:8080/v1",
                        ),
                        definitions_dir=workspace.definitions_dir,
                    )

        self.assertTrue(controller.status.running)
        self.assertFalse(rollback_handle.closed)

    def test_start_rejects_unreachable_upstream(self) -> None:
        controller = ProxyServerController()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = ensure_workspace(Path(temp_dir))
            with patch(
                "ergon_studio.server_control.prepare_proxy_runtime",
                side_effect=ValueError("upstream endpoint is not reachable: refused"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "upstream endpoint is not reachable: refused"
                ):
                    controller.start(
                        config=ProxyAppConfig(
                            upstream_base_url="http://localhost:8080/v1"
                        ),
                        definitions_dir=workspace.definitions_dir,
                    )
