from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.debug_log import (
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
