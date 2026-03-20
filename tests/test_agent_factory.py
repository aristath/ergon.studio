from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.config import save_global_config
from ergon_studio.paths import GlobalStudioPaths
from ergon_studio.registry import load_registry


class AgentFactoryTests(unittest.TestCase):
    def test_build_agent_creates_openai_chat_agent_from_registry(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            paths = GlobalStudioPaths(home_dir=home_dir)
            paths.ensure_layout()
            save_global_config(
                paths.config_path,
                {
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "orchestrator.md").write_text(
                """---
id: orchestrator
name: Orchestrator
role: orchestrator
temperature: 0.2
max_tokens: 1200
tools:
  - read_file
---
## Identity
Lead engineer for the AI firm.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            def read_file(path: str) -> str:
                return path

            agent = build_agent(registry, "orchestrator", tool_registry={"read_file": read_file})

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
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            paths = GlobalStudioPaths(home_dir=home_dir)
            paths.ensure_layout()
            save_global_config(
                paths.config_path,
                {
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "orchestrator.md").write_text(
                """---
id: orchestrator
name: Orchestrator
role: orchestrator
---
## Identity
Lead engineer.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            agent = build_agent(registry, "orchestrator", model_id_override="gpt-oss-20b")

            self.assertEqual(agent.client.model_id, "gpt-oss-20b")

    def test_build_agent_ignores_local_tool_metadata_without_registry(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            paths = GlobalStudioPaths(home_dir=home_dir)
            paths.ensure_layout()
            save_global_config(
                paths.config_path,
                {
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "orchestrator.md").write_text(
                """---
id: orchestrator
name: Orchestrator
role: orchestrator
tools:
  - read_file
---
## Identity
Lead engineer.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            agent = build_agent(registry, "orchestrator")

            self.assertEqual(agent.id, "orchestrator")
            self.assertEqual(agent.default_options["tools"], [])

    def test_build_agent_requires_defined_tools_when_registry_is_used(self) -> None:
        from ergon_studio.agent_factory import build_agent

        with tempfile.TemporaryDirectory() as temp_dir:
            home_dir = Path(temp_dir) / "home"
            home_dir.mkdir()
            paths = GlobalStudioPaths(home_dir=home_dir)
            paths.ensure_layout()
            save_global_config(
                paths.config_path,
                {
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "orchestrator.md").write_text(
                """---
id: orchestrator
name: Orchestrator
role: orchestrator
tools:
  - read_file
---
## Identity
Lead engineer.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            with self.assertRaisesRegex(ValueError, "unknown tool"):
                build_agent(registry, "orchestrator", tool_registry={})
