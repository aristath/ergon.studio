from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.registry import load_registry
from ergon_studio.upstream import UpstreamSettings


class AgentFactoryTests(unittest.TestCase):
    def test_build_agent_creates_openai_chat_agent_from_registry(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = _registry_with_agent(Path(temp_dir), metadata_extra={"temperature": 0.2, "max_tokens": 1200, "tools": ["read_file"]})

            def read_file(path: str) -> str:
                return path

            agent = build_agent(registry, "orchestrator", tool_registry={"read_file": read_file}, model_id_override="qwen2.5-coder")

            self.assertEqual(agent.id, "orchestrator")
            self.assertEqual(agent.name, "Orchestrator")
            self.assertEqual(agent.client.model_id, "qwen2.5-coder")
            self.assertEqual(agent.default_options["temperature"], 0.2)
            self.assertEqual(agent.default_options["max_tokens"], 1200)
            self.assertIn("## Identity", agent.default_options["instructions"])
            self.assertEqual(len(agent.default_options["tools"]), 1)

    def test_build_agent_honors_request_model_override(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = _registry_with_agent(Path(temp_dir))

            agent = build_agent(registry, "orchestrator", model_id_override="gpt-oss-20b")

            self.assertEqual(agent.client.model_id, "gpt-oss-20b")

    def test_build_agent_requires_request_model_override(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = _registry_with_agent(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "proxy requests must supply a model"):
                build_agent(registry, "orchestrator")

    def test_build_agent_ignores_local_tool_metadata_without_registry(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = _registry_with_agent(Path(temp_dir), metadata_extra={"tools": ["read_file"]})

            agent = build_agent(registry, "orchestrator", model_id_override="qwen2.5-coder")

            self.assertEqual(agent.id, "orchestrator")
            self.assertEqual(agent.default_options["tools"], [])

    def test_build_agent_requires_defined_tools_when_registry_is_used(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = _registry_with_agent(Path(temp_dir), metadata_extra={"tools": ["read_file"]})

            with self.assertRaisesRegex(ValueError, "unknown tool"):
                build_agent(registry, "orchestrator", tool_registry={}, model_id_override="qwen2.5-coder")


def _registry_with_agent(home_dir: Path, *, metadata_extra: dict[str, object] | None = None):
    root_dir = home_dir / "definitions"
    agents_dir = root_dir / "agents"
    workflows_dir = root_dir / "workflows"
    agents_dir.mkdir(parents=True)
    workflows_dir.mkdir(parents=True)
    metadata = {
        "id": "orchestrator",
        "name": "Orchestrator",
        "role": "orchestrator",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    frontmatter_lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            frontmatter_lines.append(f"{key}:")
            for item in value:
                frontmatter_lines.append(f"  - {item}")
        else:
            frontmatter_lines.append(f"{key}: {value}")
    frontmatter_lines.extend(["---", "## Identity", "Lead engineer for the AI firm.", ""])
    (agents_dir / "orchestrator.md").write_text("\n".join(frontmatter_lines), encoding="utf-8")
    return load_registry(
        root_dir,
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
    )
