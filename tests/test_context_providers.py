from __future__ import annotations

import unittest

from ergon_studio.context_providers import AgentProfileContextProvider
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class _FakeContext:
    def __init__(self) -> None:
        self.instructions: list[tuple[str, str]] = []

    def extend_instructions(self, source_id: str, text: str) -> None:
        self.instructions.append((source_id, text))


class AgentProfileContextProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrator_profile_includes_specialists_and_workflows(
        self,
    ) -> None:
        registry = RuntimeRegistry(
            upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
            agent_definitions={
                "orchestrator": DefinitionDocument(
                    id="orchestrator",
                    path=None,
                    metadata={
                        "id": "orchestrator",
                        "role": "orchestrator",
                        "tools": ["read_file"],
                    },
                    body="",
                    sections={},
                ),
                "coder": DefinitionDocument(
                    id="coder",
                    path=None,
                    metadata={"id": "coder", "role": "coder"},
                    body="",
                    sections={},
                ),
            },
            workflow_definitions={
                "standard-build": DefinitionDocument(
                    id="standard-build",
                    path=None,
                    metadata={"id": "standard-build", "orchestration": "sequential"},
                    body="",
                    sections={},
                )
            },
        )
        provider = AgentProfileContextProvider(
            registry.agent_definitions["orchestrator"],
            registry=registry,
        )
        context = _FakeContext()

        await provider.before_run(agent=None, session=None, context=context, state=None)

        self.assertEqual(len(context.instructions), 1)
        payload = context.instructions[0][1]
        self.assertIn("Agent profile: orchestrator", payload)
        self.assertIn("Available specialists: coder(coder)", payload)
        self.assertIn("Available workflows: standard-build(sequential)", payload)
