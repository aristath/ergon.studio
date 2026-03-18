from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.config import load_or_create_global_config, save_global_config, save_global_config_text


class GlobalConfigTests(unittest.TestCase):
    def test_load_or_create_global_config_creates_default_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".ergon.studio" / "config.json"

            config = load_or_create_global_config(config_path)

            self.assertEqual(
                config,
                {
                    "providers": {},
                    "role_assignments": {},
                    "approvals": {},
                    "ui": {},
                },
            )
            self.assertTrue(config_path.exists())

    def test_save_global_config_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".ergon.studio" / "config.json"
            expected = {
                "providers": {
                    "llama-server": {
                        "base_url": "http://localhost:8080/v1",
                        "model": "qwen",
                    }
                },
                "role_assignments": {"orchestrator": "llama-server"},
                "approvals": {"default": "ask"},
                "ui": {"theme": "default"},
            }

            save_global_config(config_path, expected)

            self.assertEqual(load_or_create_global_config(config_path), expected)

    def test_save_global_config_text_validates_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".ergon.studio" / "config.json"
            original = {
                "providers": {},
                "role_assignments": {},
                "approvals": {},
                "ui": {},
            }
            save_global_config(config_path, original)

            with self.assertRaisesRegex(ValueError, "JSON object"):
                save_global_config_text(config_path, "[]")

            self.assertEqual(load_or_create_global_config(config_path), original)
