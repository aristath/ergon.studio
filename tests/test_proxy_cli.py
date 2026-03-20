from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ergon_studio.proxy_cli import main


class ProxyCliTests(unittest.TestCase):
    def test_proxy_cli_starts_proxy_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()

            with patch("ergon_studio.proxy_cli.serve_proxy") as serve_proxy:
                exit_code = main(
                    [
                        "--home-dir",
                        str(home_dir),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "4242",
                    ]
                )

            self.assertEqual(exit_code, 0)
            serve_proxy.assert_called_once()
            _, kwargs = serve_proxy.call_args
            self.assertEqual(kwargs["host"], "0.0.0.0")
            self.assertEqual(kwargs["port"], 4242)

    def test_proxy_cli_check_fails_fast_when_orchestrator_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            stdout = io.StringIO()

            with patch("ergon_studio.proxy_cli.serve_proxy") as serve_proxy:
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "--home-dir",
                            str(home_dir),
                            "--check",
                        ]
                    )

            self.assertEqual(exit_code, 1)
            self.assertIn("ok=false", stdout.getvalue())
            serve_proxy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
