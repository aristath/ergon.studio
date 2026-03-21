from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import (
    _StreamAccumulator,
    build_agent_messages,
    build_runtime_agent,
    compose_instructions,
)
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    encode_continuation_tool_call,
)
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class AgentRunnerTests(unittest.TestCase):
    def test_build_runtime_agent_reads_definition_metadata(self) -> None:
        registry = _registry()

        agent = build_runtime_agent(registry, "orchestrator")

        self.assertEqual(agent.id, "orchestrator")
        self.assertEqual(agent.name, "Orchestrator")
        self.assertEqual(agent.role, "orchestrator")
        self.assertEqual(agent.temperature, 0.7)
        self.assertEqual(agent.max_tokens, 1200)
        self.assertIn("## Identity", agent.instructions)
        self.assertIn("Agent profile: orchestrator", agent.instructions)

    def test_compose_instructions_includes_orchestrator_profile_context(
        self,
    ) -> None:
        registry = _registry()

        instructions = compose_instructions(
            registry.agent_definitions["orchestrator"],
            registry=registry,
        )

        self.assertIn("Available specialists: coder(coder)", instructions)
        self.assertIn(
            "Available workroom presets: best-of-n(coder, coder, reviewer)",
            instructions,
        )

    def test_build_agent_messages_rebuilds_tool_history(self) -> None:
        registry = _registry()
        agent = build_runtime_agent(registry, "orchestrator")
        encoded_tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(actor="orchestrator"),
        )
        pending = PendingContinuation(
            state=ContinuationState(actor="orchestrator"),
            assistant_message=ProxyInputMessage(
                role="assistant",
                content="",
                tool_calls=(encoded_tool_call,),
            ),
            tool_results=(
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=encoded_tool_call.id,
                ),
            ),
        )

        messages = build_agent_messages(
            registry=registry,
            agent=agent,
            prompt="Use the result.",
            pending_continuation=pending,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "assistant", "tool", "user"],
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"],
            "read_file",
        )
        self.assertEqual(messages[2]["tool_call_id"], encoded_tool_call.id)

    def test_build_agent_messages_synthesizes_missing_assistant_tool_call(
        self,
    ) -> None:
        registry = _registry()
        agent = build_runtime_agent(registry, "orchestrator")
        encoded_tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(actor="orchestrator"),
        )
        pending = PendingContinuation(
            state=ContinuationState(actor="orchestrator"),
            assistant_message=None,
            tool_results=(
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=encoded_tool_call.id,
                ),
            ),
        )

        messages = build_agent_messages(
            registry=registry,
            agent=agent,
            prompt="Use the result.",
            pending_continuation=pending,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "assistant", "tool", "user"],
        )
        self.assertEqual(messages[1]["tool_calls"][0]["id"], encoded_tool_call.id)
        self.assertEqual(messages[2]["tool_call_id"], "call_1")

    def test_stream_accumulator_rebuilds_incremental_tool_call_arguments(
        self,
    ) -> None:
        accumulator = _StreamAccumulator()
        deltas = [
            SimpleNamespace(
                index=0,
                id="call_1",
                function=SimpleNamespace(name="read_file", arguments='{"path":"'),
            ),
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='main.py"}'),
            ),
        ]

        accumulator.append_tool_deltas(deltas)
        tool_calls = accumulator.tool_calls()

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "read_file")
        self.assertEqual(tool_calls[0].arguments_json, '{"path":"main.py"}')


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={
                    "id": "orchestrator",
                    "name": "Orchestrator",
                    "role": "orchestrator",
                    "temperature": 0.7,
                    "max_tokens": 1200,
                    "tools": ["read_file"],
                },
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
            "coder": DefinitionDocument(
                id="coder",
                path=Path("coder.md"),
                metadata={"id": "coder", "role": "coder"},
                body="## Identity\nCoder.",
                sections={"Identity": "Coder."},
            ),
            "reviewer": DefinitionDocument(
                id="reviewer",
                path=Path("reviewer.md"),
                metadata={"id": "reviewer", "role": "reviewer"},
                body="## Identity\nReviewer.",
                sections={"Identity": "Reviewer."},
            ),
        },
        workroom_definitions={
            "best-of-n": DefinitionDocument(
                id="best-of-n",
                path=Path("best-of-n.md"),
                metadata={
                    "id": "best-of-n",
                    "participants": ["coder", "coder", "reviewer"],
                },
                body="## Purpose\nCompare options.",
                sections={"Purpose": "Compare options."},
            ),
        },
    )
