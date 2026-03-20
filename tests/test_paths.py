from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.paths import GlobalStudioPaths


class GlobalStudioPathsTests(unittest.TestCase):
    def test_paths_follow_the_proxy_home_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            paths = GlobalStudioPaths(home_dir=home_dir)

            self.assertEqual(paths.config_path, home_dir / ".ergon.studio" / "config.json")
            self.assertEqual(paths.agents_dir, home_dir / ".ergon.studio" / "agents")
            self.assertEqual(paths.workflows_dir, home_dir / ".ergon.studio" / "workflows")
