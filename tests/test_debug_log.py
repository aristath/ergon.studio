from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from ergon_studio.debug_log import (
    _to_jsonable,
    configure_debug_logging,
    disable_debug_logging,
    log_event,
)


class DebugLogTests(unittest.TestCase):
    def tearDown(self) -> None:
        disable_debug_logging()

    def test_log_event_writes_json_line_to_configured_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ergon.log"
            configure_debug_logging(log_path)

            log_event(
                "channel_message",
                session_id="session_1",
                payload={"author": "coder", "content": "Done"},
            )

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertGreaterEqual(len(lines), 2)
        self.assertIn('"event": "logging_enabled"', lines[0])
        self.assertIn('"event": "channel_message"', lines[-1])
        self.assertIn('"author": "coder"', lines[-1])

    def test_to_jsonable_truncates_deep_nesting(self) -> None:
        @dataclass
        class _Node:
            child: object

        value: object = "leaf"
        for _ in range(12):
            value = _Node(child=value)

        result = _to_jsonable(value)

        depth = 0
        node = result
        while isinstance(node, dict) and "child" in node:
            node = node["child"]
            depth += 1
        self.assertLessEqual(depth, 8)
        self.assertIsInstance(node, str)

    def test_to_jsonable_truncates_long_lists(self) -> None:
        result = _to_jsonable(list(range(30)))
        self.assertLessEqual(len(result), 21)
        self.assertIsInstance(result[-1], str)
        self.assertIn("more", result[-1])

    def test_to_jsonable_truncates_long_strings(self) -> None:
        result = _to_jsonable("x" * 600)
        self.assertLessEqual(len(result), 520)
        self.assertIn("truncated", result)
