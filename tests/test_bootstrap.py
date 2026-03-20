from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_definition_home


class BootstrapDefinitionHomeTests(unittest.TestCase):
    def test_bootstrap_definition_home_creates_global_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()

            paths = bootstrap_definition_home(home_dir)

            self.assertTrue(paths.agents_dir.exists())
            self.assertTrue(paths.workflows_dir.exists())
            self.assertTrue((paths.agents_dir / "orchestrator.md").exists())
            self.assertTrue((paths.workflows_dir / "standard-build.md").exists())

    def test_bootstrap_definition_home_does_not_overwrite_existing_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            agents_dir = home_dir / ".ergon.studio" / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            orchestrator_path = agents_dir / "orchestrator.md"
            orchestrator_path.write_text("custom orchestrator\n", encoding="utf-8")

            bootstrap_definition_home(home_dir)

            self.assertEqual(orchestrator_path.read_text(encoding="utf-8"), "custom orchestrator\n")
