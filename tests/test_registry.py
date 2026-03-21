from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.registry import load_registry
from ergon_studio.upstream import UpstreamSettings


class RegistryTests(unittest.TestCase):
    def test_load_registry_returns_upstream_and_definition_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            agents_dir = root_dir / "agents"
            workflows_dir = root_dir / "workflows"
            agents_dir.mkdir(parents=True)
            workflows_dir.mkdir(parents=True)
            (agents_dir / "orchestrator.md").write_text(
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
            (workflows_dir / "standard-build.md").write_text(
                (
                    "---\n"
                    "id: standard-build\n"
                    "orchestration: sequential\n"
                    "steps:\n"
                    "  - architect\n"
                    "  - coder\n"
                    "---\n"
                    "## Purpose\n"
                    "Build.\n"
                ),
                encoding="utf-8",
            )
            (workflows_dir / "research-then-decide.md").write_text(
                (
                    "---\n"
                    "id: research-then-decide\n"
                    "orchestration: sequential\n"
                    "steps:\n"
                    "  - researcher\n"
                    "---\n"
                    "## Purpose\n"
                    "Research.\n"
                ),
                encoding="utf-8",
            )
            (workflows_dir / "debate.md").write_text(
                (
                    "---\n"
                    "id: debate\n"
                    "orchestration: group_chat\n"
                    "step_groups:\n"
                    "  - [architect, reviewer]\n"
                    "---\n"
                    "## Purpose\n"
                    "Debate.\n"
                ),
                encoding="utf-8",
            )

            registry = load_registry(
                root_dir,
                upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
            )

            self.assertIn("orchestrator", registry.agent_definitions)
            self.assertIn("standard-build", registry.workflow_definitions)
            self.assertIn("research-then-decide", registry.workflow_definitions)
            self.assertIn("debate", registry.workflow_definitions)
            self.assertEqual(registry.upstream.base_url, "http://localhost:8080/v1")

    def test_load_registry_requires_agents_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            root_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "missing agents directory"):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )

    def test_load_registry_requires_orchestrator_definition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            (root_dir / "agents").mkdir(parents=True)
            (root_dir / "workflows").mkdir(parents=True)
            with self.assertRaisesRegex(
                ValueError, "missing required agent definition"
            ):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )
