from __future__ import annotations

import subprocess
import sys
import unittest


class CliImportTests(unittest.TestCase):
    def test_cli_import_does_not_pull_in_native_app_modules(self) -> None:
        script = """
import json
import sys
import ergon_studio.cli
tracked = [
    'ergon_studio.runtime',
    'ergon_studio.session_store',
    'ergon_studio.tui.app',
    'textual',
]
print(json.dumps({name: (name in sys.modules) for name in tracked}))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = __import__("json").loads(result.stdout)

        self.assertFalse(loaded["ergon_studio.runtime"])
        self.assertFalse(loaded["ergon_studio.session_store"])
        self.assertFalse(loaded["ergon_studio.tui.app"])
        self.assertFalse(loaded["textual"])


if __name__ == "__main__":
    unittest.main()
