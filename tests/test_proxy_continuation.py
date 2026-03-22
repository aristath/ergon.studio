from __future__ import annotations

import unittest

from ergon_studio.proxy.continuation import (
    continuation_result_map,
    continuation_tool_calls,
    decode_pending_id_from_tool_call_id,
    encode_continuation_tool_call,
    latest_pending_continuation,
    original_tool_call_id,
    pending_actors,
    pending_for_actor,
)
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall
from ergon_studio.proxy.pending_store import PendingSeed, PendingStore


class ProxyContinuationTests(unittest.TestCase):
    def test_encode_wraps_pending_id_and_original_call_id(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            pending_id="pending_123",
        )

        self.assertEqual(decode_pending_id_from_tool_call_id(encoded.id), "pending_123")
        self.assertEqual(original_tool_call_id(encoded.id), "call_1")

    def test_latest_pending_continuation_restores_server_side_pending_state(
        self,
    ) -> None:
        store = PendingStore()
        record = store.create(
            seed=PendingSeed(
                session_id="session_1",
                actor="coder",
                active_channel_id="channel-1",
            ),
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json='{"path":"main.py"}',
                ),
            ),
        )
        wrapped = encode_continuation_tool_call(
            record.tool_calls[0],
            pending_id=record.pending_id,
        )
        messages = (
            ProxyInputMessage(role="user", content="Implement it"),
            ProxyInputMessage(role="assistant", content="", tool_calls=(wrapped,)),
            ProxyInputMessage(
                role="tool",
                content="file contents",
                tool_call_id=wrapped.id,
            ),
        )

        pending = latest_pending_continuation(messages, pending_store=store)

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.session_id, "session_1")
        self.assertEqual(pending_actors(pending), ("coder",))
        coder = pending_for_actor(pending, "coder")
        assert coder is not None
        self.assertEqual(coder.active_channel_id, "channel-1")
        self.assertEqual(
            continuation_tool_calls(coder),
            (
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json='{"path":"main.py"}',
                ),
            ),
        )
        self.assertEqual(continuation_result_map(coder), {"call_1": "file contents"})
        self.assertIsNone(store.get(record.pending_id))

    def test_latest_pending_continuation_groups_multiple_pending_actors(
        self,
    ) -> None:
        store = PendingStore()
        architect = store.create(
            seed=PendingSeed(
                session_id="session_2",
                actor="architect",
                active_channel_id="channel-2",
            ),
            tool_calls=(
                ProxyToolCall(id="call_a", name="read_file", arguments_json="{}"),
            ),
        )
        coder = store.create(
            seed=PendingSeed(
                session_id="session_2",
                actor="coder",
                active_channel_id="channel-2",
            ),
            tool_calls=(
                ProxyToolCall(id="call_b", name="glob", arguments_json="{}"),
            ),
        )
        wrapped_architect = encode_continuation_tool_call(
            architect.tool_calls[0],
            pending_id=architect.pending_id,
        )
        wrapped_coder = encode_continuation_tool_call(
            coder.tool_calls[0],
            pending_id=coder.pending_id,
        )
        messages = (
            ProxyInputMessage(role="user", content="Debate it"),
            ProxyInputMessage(
                role="assistant",
                content="",
                tool_calls=(wrapped_architect, wrapped_coder),
            ),
            ProxyInputMessage(
                role="tool",
                content="architect context",
                tool_call_id=wrapped_architect.id,
            ),
            ProxyInputMessage(
                role="tool",
                content="coder context",
                tool_call_id=wrapped_coder.id,
            ),
        )

        pending = latest_pending_continuation(messages, pending_store=store)

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.session_id, "session_2")
        self.assertEqual(set(pending_actors(pending)), {"architect", "coder"})

    def test_latest_pending_continuation_rejects_mixed_sessions(self) -> None:
        store = PendingStore()
        first = store.create(
            seed=PendingSeed(session_id="session_a", actor="coder"),
            tool_calls=(
                ProxyToolCall(
                    id="call_a",
                    name="read_file",
                    arguments_json="{}",
                ),
            ),
        )
        second = store.create(
            seed=PendingSeed(session_id="session_b", actor="reviewer"),
            tool_calls=(
                ProxyToolCall(
                    id="call_b",
                    name="glob",
                    arguments_json="{}",
                ),
            ),
        )
        wrapped_first = encode_continuation_tool_call(
            first.tool_calls[0],
            pending_id=first.pending_id,
        )
        wrapped_second = encode_continuation_tool_call(
            second.tool_calls[0],
            pending_id=second.pending_id,
        )
        messages = (
            ProxyInputMessage(
                role="assistant",
                content="",
                tool_calls=(wrapped_first, wrapped_second),
            ),
            ProxyInputMessage(role="tool", content="a", tool_call_id=wrapped_first.id),
            ProxyInputMessage(role="tool", content="b", tool_call_id=wrapped_second.id),
        )

        self.assertIsNone(latest_pending_continuation(messages, pending_store=store))
