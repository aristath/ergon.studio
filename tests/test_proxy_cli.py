from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ergon_studio.proxy_cli import main


class ProxyCliTests(unittest.TestCase):
    def test_proxy_cli_starts_proxy_server(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions_dir = Path(temp_dir) / "definitions"
            (definitions_dir / "agents").mkdir(parents=True)
            (definitions_dir / "workflows").mkdir(parents=True)
            (definitions_dir / "agents" / "orchestrator.md").write_text(
                (
                    "---\n"
                    "id: orchestrator\n"
                    "role: orchestrator\n"
                    "---\n"
                    "## Identity\n"
                    "Lead engineer.\n"
                ),
                encoding="utf-8",
            )

            with patch("ergon_studio.proxy_cli.serve_proxy") as serve_proxy:
                exit_code = main(
                    [
                        "--definitions-dir",
                        str(definitions_dir),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "4242",
                        "--upstream-base-url",
                        "http://localhost:8080/v1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            serve_proxy.assert_called_once()
            _, kwargs = serve_proxy.call_args
            self.assertEqual(kwargs["host"], "0.0.0.0")
            self.assertEqual(kwargs["port"], 4242)

    def test_proxy_cli_requires_upstream_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            definitions_dir = Path(temp_dir) / "definitions"
            definitions_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "missing upstream base URL"):
                main(["--definitions-dir", str(definitions_dir)])

    def test_proxy_cli_requires_definitions_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing definitions directory"):
            main(["--upstream-base-url", "http://localhost:8080/v1"])
