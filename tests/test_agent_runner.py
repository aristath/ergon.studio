from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import (
    _StreamAccumulator,
    build_agent_messages,
    compose_instructions,
)
from ergon_studio.proxy.continuation import (
    PendingToolContext,
    encode_continuation_tool_call,
)
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class AgentRunnerTests(unittest.TestCase):
    def test_compose_instructions_includes_orchestrator_profile_context(self) -> None:
        registry = _registry()

        instructions = compose_instructions(
            registry.agent_definitions["orchestrator"],
            registry=registry,
        )

        self.assertIn("Available specialists: coder(coder)", instructions)
        self.assertIn(
            "Available channel presets: best-of-n(coder, coder, reviewer)",
            instructions,
        )

    def test_build_agent_messages_rebuilds_pending_tool_history(self) -> None:
        registry = _registry()
        instructions = compose_instructions(
            registry.agent_definitions["orchestrator"],
            registry=registry,
        )
        encoded_tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            pending_id="pending_1",
        )
        pending = PendingToolContext(
            pending_id="pending_1",
            session_id="session_1",
            actor="orchestrator",
            active_channel_id=None,
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json='{"path":"main.py"}',
                ),
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
            instructions=instructions,
            prompt="Use the result.",
            prompt_role="user",
            conversation_messages=(),
            pending_continuation=pending,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[3]["tool_call_id"], "call_1")

    def test_build_agent_messages_replaces_encoded_pending_tail(self) -> None:
        registry = _registry()
        instructions = compose_instructions(
            registry.agent_definitions["orchestrator"],
            registry=registry,
        )
        encoded_tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            pending_id="pending_1",
        )
        pending = PendingToolContext(
            pending_id="pending_1",
            session_id="session_1",
            actor="orchestrator",
            active_channel_id=None,
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json='{"path":"main.py"}',
                ),
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
            instructions=instructions,
            prompt="Use the result.",
            prompt_role="system",
            conversation_messages=(
                ProxyInputMessage(role="user", content="Inspect it"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=(encoded_tool_call,),
                ),
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=encoded_tool_call.id,
                ),
            ),
            pending_continuation=pending,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user", "assistant", "tool"],
        )
        self.assertEqual(messages[3]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[4]["tool_call_id"], "call_1")

    def test_build_agent_messages_supports_system_framed_channel_calls(self) -> None:
        registry = _registry()
        instructions = compose_instructions(
            registry.agent_definitions["coder"],
            registry=registry,
        )

        messages = build_agent_messages(
            registry=registry,
            instructions=instructions,
            prompt="You are in channel debate.",
            prompt_role="system",
            conversation_messages=(
                ProxyInputMessage(
                    role="user",
                    name="orchestrator",
                    content="Please inspect models.py.",
                ),
                ProxyInputMessage(
                    role="assistant",
                    name="coder",
                    content="I checked the file and found the dataclasses.",
                ),
            ),
            pending_continuation=None,
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user", "assistant"],
        )
        self.assertEqual(messages[2]["name"], "orchestrator")
        self.assertEqual(messages[3]["name"], "coder")

    def test_strip_pending_does_not_pop_assistant_with_mixed_pending_ids(self) -> None:
        from ergon_studio.proxy.agent_runner import _strip_pending_messages
        from ergon_studio.proxy.continuation import (
            PendingToolContext,
            encode_continuation_tool_call,
        )
        from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall

        call_1 = ProxyToolCall(id="call_1", name="read_file", arguments_json="{}")
        call_2 = ProxyToolCall(id="call_2", name="write_file", arguments_json="{}")
        encoded_1 = encode_continuation_tool_call(call_1, pending_id="pending_1")
        encoded_2 = encode_continuation_tool_call(call_2, pending_id="pending_2")

        assistant = ProxyInputMessage(
            role="assistant",
            content="",
            tool_calls=(encoded_1, encoded_2),
        )
        tool_result_1 = ProxyInputMessage(
            role="tool",
            content="result",
            tool_call_id=encoded_1.id,
        )
        pending = PendingToolContext(
            pending_id="pending_1",
            session_id="session_1",
            actor="orchestrator",
            active_channel_id=None,
            tool_calls=(call_1,),
            tool_results=(tool_result_1,),
        )
        conversation = (
            ProxyInputMessage(role="user", content="Do it"),
            assistant,
            tool_result_1,
        )

        result = _strip_pending_messages(conversation, pending_continuation=pending)

        # Tool result for pending_1 is stripped
        # But assistant with mixed pending IDs is preserved (not popped)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1].role, "assistant")
        self.assertEqual(len(result[-1].tool_calls), 2)

    def test_stream_accumulator_rebuilds_incremental_tool_call_arguments(self) -> None:
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
        channel_presets={
            "best-of-n": ("coder", "coder", "reviewer"),
        },
    )
