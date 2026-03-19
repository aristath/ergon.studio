from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from ergon_studio.config import save_global_config
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import load_registry
from ergon_studio.seed import seed_default_definitions


class RegistryTests(unittest.TestCase):
    def test_load_registry_returns_config_and_definition_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            home_dir = base / "home"
            project_root = base / "repo"
            home_dir.mkdir()
            project_root.mkdir()
            paths = StudioPaths(
                home_dir=home_dir,
                project_root=project_root,
                project_uuid=UUID("12345678-1234-5678-1234-567812345678"),
            )
            paths.ensure_global_layout()
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
            self.assertEqual(
                registry.config["role_assignments"]["orchestrator"],
                "default",
            )
            single_agent = registry.workflow_definitions["single-agent-execution"]
            self.assertEqual(single_agent.metadata["max_repair_cycles"], 3)
            self.assertEqual(
                single_agent.metadata["repair_step_groups"],
                [["tester"], ["fixer"], ["tester"], ["reviewer"]],
            )
