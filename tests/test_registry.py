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
            channels_dir = root_dir / "channels"
            agents_dir.mkdir(parents=True)
            channels_dir.mkdir(parents=True)
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
            (agents_dir / "architect.md").write_text(
                (
                    "---\n"
                    "id: architect\n"
                    "role: architect\n"
                    "---\n"
                    "## Identity\n"
                    "Architect.\n"
                ),
                encoding="utf-8",
            )
            (agents_dir / "reviewer.md").write_text(
                (
                    "---\n"
                    "id: reviewer\n"
                    "role: reviewer\n"
                    "---\n"
                    "## Identity\n"
                    "Reviewer.\n"
                ),
                encoding="utf-8",
            )
            (agents_dir / "researcher.md").write_text(
                (
                    "---\n"
                    "id: researcher\n"
                    "role: researcher\n"
                    "---\n"
                    "## Identity\n"
                    "Researcher.\n"
                ),
                encoding="utf-8",
            )
            (agents_dir / "coder.md").write_text(
                (
                    "---\n"
                    "id: coder\n"
                    "role: coder\n"
                    "---\n"
                    "## Identity\n"
                    "Coder.\n"
                ),
                encoding="utf-8",
            )
            (channels_dir / "standard-build.md").write_text(
                (
                    "---\n"
                    "id: standard-build\n"
                    "participants:\n"
                    "  - architect\n"
                    "  - coder\n"
                    "---\n"
                    "## Purpose\n"
                    "Build.\n"
                ),
                encoding="utf-8",
            )
            (channels_dir / "research-then-decide.md").write_text(
                (
                    "---\n"
                    "id: research-then-decide\n"
                    "participants:\n"
                    "  - researcher\n"
                    "---\n"
                    "## Purpose\n"
                    "Research.\n"
                ),
                encoding="utf-8",
            )
            (channels_dir / "debate.md").write_text(
                (
                    "---\n"
                    "id: debate\n"
                    "participants:\n"
                    "  - architect\n"
                    "  - reviewer\n"
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
            self.assertIn("standard-build", registry.channel_presets)
            self.assertIn("research-then-decide", registry.channel_presets)
            self.assertIn("debate", registry.channel_presets)
            self.assertEqual(
                registry.channel_presets["standard-build"],
                ("architect", "coder"),
            )
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
            (root_dir / "channels").mkdir(parents=True)
            with self.assertRaisesRegex(
                ValueError, "missing required agent definition"
            ):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )

    def test_load_registry_rejects_channel_presets_with_unknown_agents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            agents_dir = root_dir / "agents"
            channels_dir = root_dir / "channels"
            agents_dir.mkdir(parents=True)
            channels_dir.mkdir(parents=True)
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
            (channels_dir / "standard-build.md").write_text(
                (
                    "---\n"
                    "id: standard-build\n"
                    "participants:\n"
                    "  - coder\n"
                    "---\n"
                    "## Purpose\n"
                    "Build.\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "channel preset 'standard-build' references unknown agents: coder",
            ):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )

    def test_load_registry_requires_channel_preset_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            agents_dir = root_dir / "agents"
            channels_dir = root_dir / "channels"
            agents_dir.mkdir(parents=True)
            channels_dir.mkdir(parents=True)
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
            (channels_dir / "broken.md").write_text(
                "---\nid: broken\n---\n## Purpose\nBroken.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must declare `participants`"):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )

    def test_load_registry_rejects_non_list_channel_preset_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            agents_dir = root_dir / "agents"
            channels_dir = root_dir / "channels"
            agents_dir.mkdir(parents=True)
            channels_dir.mkdir(parents=True)
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
            (channels_dir / "broken.md").write_text(
                "---\nid: broken\nparticipants: coder\n---\n## Purpose\nBroken.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "participants must be a list"):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )

    def test_load_registry_rejects_empty_channel_preset_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir) / "definitions"
            agents_dir = root_dir / "agents"
            channels_dir = root_dir / "channels"
            agents_dir.mkdir(parents=True)
            channels_dir.mkdir(parents=True)
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
            (channels_dir / "broken.md").write_text(
                "---\nid: broken\nparticipants:\n  - \"\"\n---\n## Purpose\nBroken.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "participants must be non-empty strings"
            ):
                load_registry(
                    root_dir,
                    upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
                )
