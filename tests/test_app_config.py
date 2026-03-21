from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ergon_studio.app_config import (
    ProxyAppConfig,
    config_path,
    default_app_dir,
    definitions_dir,
    load_app_config,
    save_app_config,
    validate_proxy_host,
    validate_proxy_port,
)


class AppConfigTests(unittest.TestCase):
    def test_default_app_dir_uses_xdg_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": temp_dir}):
                self.assertEqual(default_app_dir(), Path(temp_dir) / "ergon")

    def test_load_app_config_returns_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            self.assertEqual(load_app_config(path), ProxyAppConfig())

    def test_save_and_load_app_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            config = ProxyAppConfig(
                upstream_base_url="http://localhost:8080/v1",
                upstream_api_key="secret",
                host="0.0.0.0",
                port=4242,
                instruction_role="developer",
                disable_tool_calling=True,
            )

            save_app_config(path, config)

            self.assertEqual(load_app_config(path), config)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["port"], 4242)

    def test_helper_paths_are_derived_from_app_dir(self) -> None:
        app_dir = Path("/tmp/example")
        self.assertEqual(config_path(app_dir), app_dir / "config.json")
        self.assertEqual(definitions_dir(app_dir), app_dir / "definitions")

    def test_load_app_config_rejects_boolean_port_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text('{"port": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "config numeric values must be integers"
            ):
                load_app_config(path)

    def test_load_app_config_rejects_non_boolean_disable_tool_calling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text('{"disable_tool_calling": "false"}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "config boolean values must be bools"
            ):
                load_app_config(path)

    def test_validate_proxy_host_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "proxy host must be non-empty"):
            validate_proxy_host("   ")

    def test_validate_proxy_port_rejects_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "proxy port must be between 1 and 65535"
        ):
            validate_proxy_port(0)
