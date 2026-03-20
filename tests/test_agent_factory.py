from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from ergon_studio.config import save_global_config
from ergon_studio.paths import StudioPaths
from ergon_studio.registry import load_registry


class AgentFactoryTests(unittest.TestCase):
    def test_build_agent_creates_openai_chat_agent_from_registry(self) -> None:
        from ergon_studio.agent_factory import build_agent

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
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {"default": "ask"},
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

## Responsibilities
Plan and delegate.

## Rules
Avoid keyword-triggered behavior.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            def read_file(path: str) -> str:
                return path

            agent = build_agent(registry, "orchestrator", tool_registry={"read_file": read_file})

            self.assertEqual(agent.id, "orchestrator")
            self.assertEqual(agent.name, "Orchestrator")
            self.assertEqual(agent.description, "orchestrator")
            self.assertEqual(agent.client.model_id, "qwen2.5-coder")
            self.assertEqual(agent.default_options["temperature"], 0.2)
            self.assertEqual(agent.default_options["max_tokens"], 1200)
            self.assertIn("## Identity", agent.default_options["instructions"])
            self.assertEqual(len(agent.default_options["tools"]), 1)

    def test_build_agent_requires_defined_tools(self) -> None:
        from ergon_studio.agent_factory import build_agent

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

    def test_build_agent_can_ignore_missing_tools_for_proxy_use(self) -> None:
        from ergon_studio.agent_factory import build_agent

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

            agent = build_agent(
                registry,
                "orchestrator",
                tool_registry={},
                ignore_missing_tools=True,
                include_mcp_servers=False,
            )

            self.assertEqual(agent.id, "orchestrator")
            self.assertEqual(agent.default_options["tools"], [])

    def test_build_agent_can_resolve_seeded_researcher_tools(self) -> None:
        from ergon_studio.agent_factory import build_agent
        from ergon_studio.artifact_store import ArtifactStore
        from ergon_studio.conversation_store import ConversationStore
        from ergon_studio.event_store import EventStore
        from ergon_studio.memory_store import MemoryStore
        from ergon_studio.whiteboard_store import WhiteboardStore

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
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"researcher": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "researcher.md").write_text(
                """---
id: researcher
name: Researcher
role: researcher
tools:
  - search_files
  - web_lookup
---
## Identity
Research specialist.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            def search_files(pattern: str, path: str = ".") -> list[dict[str, int | str]]:
                return [{"path": path, "line_number": 1, "line": pattern}]

            def web_lookup(query: str, limit: int = 5) -> list[dict[str, str]]:
                return [{"title": query, "url": "https://example.com", "snippet": str(limit)}]

            agent = build_agent(
                registry,
                "researcher",
                tool_registry={
                    "search_files": search_files,
                    "web_lookup": web_lookup,
                },
                conversation_store=ConversationStore(paths),
                memory_store=MemoryStore(paths),
                artifact_store=ArtifactStore(paths),
                whiteboard_store=WhiteboardStore(paths),
                event_store=EventStore(paths),
            )

            self.assertEqual(agent.id, "researcher")
            self.assertEqual(len(agent.default_options["tools"]), 2)
            self.assertEqual(len(agent.context_providers), 5)

    def test_build_agent_can_attach_mcp_servers_from_definition(self) -> None:
        from ergon_studio.agent_factory import build_agent

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
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                            "temperature": 0.3,
                            "capabilities": {
                                "tool_calling": True,
                                "structured_output": True,
                            },
                        }
                    },
                    "role_assignments": {"researcher": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            (paths.agents_dir / "researcher.md").write_text(
                """---
id: researcher
name: Researcher
role: researcher
mcp_servers:
  - name: docs
    transport: stdio
    command: npx
    args:
      - docs-server
---
## Identity
Research specialist.
""",
                encoding="utf-8",
            )
            registry = load_registry(paths)

            agent = build_agent(registry, "researcher", tool_registry={})

            self.assertEqual(agent.default_options["temperature"], 0.3)
            self.assertEqual(agent.default_options["tools"], [])
            self.assertEqual(len(agent.mcp_tools), 1)
            self.assertEqual(agent.mcp_tools[0].name, "docs")
