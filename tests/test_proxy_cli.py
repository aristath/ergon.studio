from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from ergon_studio.debug_log import default_debug_log_path
from ergon_studio.proxy_cli import main


@dataclass(frozen=True)
class _PreparedRuntime:
    host: str
    port: int
    core: object


class ProxyCliTests(unittest.TestCase):
    def test_proxy_cli_starts_proxy_server_in_headless_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = _PreparedRuntime(
                host="0.0.0.0",
                port=4242,
                core=object(),
            )
            with (
                patch(
                    "ergon_studio.proxy_cli.prepare_proxy_runtime",
                    return_value=prepared,
                ) as prepare_proxy_runtime,
                patch("ergon_studio.proxy_cli.serve_proxy") as serve_proxy,
            ):
                exit_code = main(
                    [
                        "--serve",
                        "--app-dir",
                        temp_dir,
                        "--definitions-dir",
                        str(Path(temp_dir) / "definitions"),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "4242",
                        "--upstream-base-url",
                        "http://localhost:8080/v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            prepare_proxy_runtime.assert_called_once()
            serve_proxy.assert_called_once()
            _, kwargs = serve_proxy.call_args
            self.assertEqual(kwargs["host"], "0.0.0.0")
            self.assertEqual(kwargs["port"], 4242)

    def test_proxy_cli_requires_upstream_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                main(["--serve", "--app-dir", temp_dir])

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("missing upstream base URL", stderr.getvalue())

    def test_proxy_cli_does_not_bootstrap_workspace_in_headless_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = _PreparedRuntime(
                host="127.0.0.1",
                port=4000,
                core=object(),
            )
            with (
                patch("ergon_studio.proxy_cli.ensure_workspace") as ensure_workspace,
                patch(
                    "ergon_studio.proxy_cli.prepare_proxy_runtime",
                    return_value=prepared,
                ) as prepare_proxy_runtime,
                patch("ergon_studio.proxy_cli.import_module") as import_module,
                patch("ergon_studio.proxy_cli.serve_proxy") as serve_proxy,
            ):
                exit_code = main(
                    [
                        "--serve",
                        "--app-dir",
                        temp_dir,
                        "--definitions-dir",
                        str(Path(temp_dir) / "definitions"),
                        "--upstream-base-url",
                        "http://localhost:8080/v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            ensure_workspace.assert_not_called()
            prepare_proxy_runtime.assert_called_once()
            import_module.assert_not_called()
            serve_proxy.assert_called_once()

    def test_proxy_cli_launches_tui_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("ergon_studio.proxy_cli._run_config_tui") as run_config_tui:
                exit_code = main(["--app-dir", temp_dir])

            self.assertEqual(exit_code, run_config_tui.return_value)
            run_config_tui.assert_called_once()

    def test_proxy_cli_reports_missing_textual_for_tui_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = io.StringIO()
            with (
                redirect_stderr(stderr),
                patch(
                    "ergon_studio.proxy_cli.import_module",
                    side_effect=ModuleNotFoundError(name="textual"),
                ),
                self.assertRaises(SystemExit) as exc,
            ):
                main(["--app-dir", temp_dir])

        self.assertEqual(exc.exception.code, 2)
        self.assertIn(
            "the configuration TUI requires the 'textual' dependency",
            stderr.getvalue(),
        )

    def test_proxy_cli_rejects_unreachable_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stderr = io.StringIO()
            with (
                redirect_stderr(stderr),
                patch(
                    "ergon_studio.proxy_cli.prepare_proxy_runtime",
                    side_effect=ValueError(
                        "upstream endpoint is not reachable: refused"
                    ),
                ),
                self.assertRaises(SystemExit) as exc,
            ):
                main(
                    [
                        "--serve",
                        "--app-dir",
                        temp_dir,
                        "--definitions-dir",
                        str(Path(temp_dir) / "definitions"),
                        "--upstream-base-url",
                        "http://localhost:8080/v1",
                    ]
                )

        self.assertEqual(exc.exception.code, 2)
        self.assertIn("upstream endpoint is not reachable", stderr.getvalue())

    def test_proxy_cli_enables_debug_log_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = _PreparedRuntime(
                host="127.0.0.1",
                port=4000,
                core=object(),
            )
            with (
                patch(
                    "ergon_studio.proxy_cli.prepare_proxy_runtime",
                    return_value=prepared,
                ),
                patch("ergon_studio.proxy_cli.serve_proxy"),
                patch(
                    "ergon_studio.proxy_cli.configure_debug_logging"
                ) as configure_debug_logging,
            ):
                exit_code = main(
                    [
                        "--serve",
                        "--log",
                        "--app-dir",
                        temp_dir,
                        "--definitions-dir",
                        str(Path(temp_dir) / "definitions"),
                        "--upstream-base-url",
                        "http://localhost:8080/v1",
                    ]
                )

        self.assertEqual(exit_code, 0)
        configure_debug_logging.assert_called_once_with(default_debug_log_path())
