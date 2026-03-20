from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.config import save_global_config
from ergon_studio.paths import GlobalStudioPaths
from ergon_studio.registry import load_registry
from ergon_studio.seed import seed_default_definitions


class RegistryTests(unittest.TestCase):
    def test_load_registry_returns_config_and_definition_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            paths = GlobalStudioPaths(home_dir=home_dir)
            paths.ensure_layout()
            save_global_config(
                paths.config_path,
                {
                    "providers": {"default": {"base_url": "http://localhost:8080/v1"}},
                    "role_assignments": {"orchestrator": "default"},
                    "approvals": {"default": "ask"},
                    "ui": {"theme": "default"},
                },
            )
            seed_default_definitions(paths.agents_dir, paths.workflows_dir)

            registry = load_registry(paths)

            self.assertIn("orchestrator", registry.agent_definitions)
            self.assertIn("standard-build", registry.workflow_definitions)
            self.assertIn("research-then-decide", registry.workflow_definitions)
            self.assertIn("debate", registry.workflow_definitions)
            self.assertEqual(registry.config["role_assignments"]["orchestrator"], "default")
