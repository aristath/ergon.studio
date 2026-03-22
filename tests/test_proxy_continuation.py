from __future__ import annotations

import unittest

from ergon_studio.proxy.channels import ChannelMessage, ChannelSnapshot
from ergon_studio.proxy.continuation import (
    ContinuationState,
    decode_continuation_from_tool_call_id,
    decode_original_tool_call,
    encode_continuation_tool_call,
    latest_pending_continuation,
    original_tool_call_id,
    pending_actors,
)
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall


class ProxyContinuationTests(unittest.TestCase):
    def test_encode_and_decode_round_trip(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(
                actor="architect",
                active_channel_id="channel-1",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-1",
                        name="design",
                        participants=("architect",),
                    ),
                ),
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.actor, "architect")
        self.assertEqual(decoded.active_channel_id, "channel-1")
        self.assertEqual(
            decoded.channels,
            (
                ChannelSnapshot(
                    channel_id="channel-1",
                    name="design",
                    participants=("architect",),
                ),
            ),
        )
        self.assertEqual(decoded.worklog, ())

    def test_encode_and_decode_round_trip_with_context(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_2",
                name="write_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(
                actor="coder[1]",
                active_channel_id="channel-2",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-2",
                        name="best-of-n",
                        participants=("coder", "coder", "reviewer"),
                        transcript=(
                            ChannelMessage("architect", "Use main.py"),
                            ChannelMessage("coder[2]", "Option B"),
                        ),
                    ),
                ),
                worklog=("reviewer: keep it small",),
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(
            decoded.channels,
            (
                ChannelSnapshot(
                    channel_id="channel-2",
                    name="best-of-n",
                    participants=("coder", "coder", "reviewer"),
                    transcript=(
                        ChannelMessage("architect", "Use main.py"),
                        ChannelMessage("coder[2]", "Option B"),
                    ),
                ),
            ),
        )
        self.assertEqual(decoded.active_channel_id, "channel-2")
        self.assertEqual(decoded.worklog, ("reviewer: keep it small",))
        self.assertEqual(decoded.actor, "coder[1]")

    def test_encode_caps_transcript_and_worklog_payloads(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(id="call_3", name="read_file", arguments_json="{}"),
            state=ContinuationState(
                actor="coder",
                active_channel_id="channel-3",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-3",
                        name="ad hoc",
                        participants=("coder",),
                        transcript=tuple(
                            ChannelMessage("coder", f"line {index}")
                            for index in range(20)
                        ),
                    ),
                ),
                worklog=tuple(f"note {index}" for index in range(20)),
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(
            decoded.channels,
            (
                ChannelSnapshot(
                    channel_id="channel-3",
                    name="ad hoc",
                    participants=("coder",),
                    transcript=tuple(
                        ChannelMessage("coder", f"line {index}")
                        for index in range(8, 20)
                    ),
                ),
            ),
        )
        self.assertEqual(
            decoded.worklog,
            tuple(f"note {index}" for index in range(8, 20)),
        )

    def test_latest_pending_continuation_requires_tool_loop_tail(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(
                actor="coder",
                active_channel_id="channel-1",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-1",
                        name="ad hoc",
                        participants=("coder",),
                    ),
                ),
            ),
        )
        messages = (
            ProxyInputMessage(role="user", content="Build it"),
            ProxyInputMessage(role="assistant", content="", tool_calls=(tool_call,)),
            ProxyInputMessage(
                role="tool", content="file contents", tool_call_id=tool_call.id
            ),
            ProxyInputMessage(role="assistant", content="Done with that."),
            ProxyInputMessage(role="user", content="Now do something else"),
        )

        self.assertIsNone(latest_pending_continuation(messages))

    def test_latest_pending_continuation_returns_matching_assistant_and_results(
        self,
    ) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(
                actor="coder",
                active_channel_id="channel-1",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-1",
                        name="ad hoc",
                        participants=("coder",),
                    ),
                ),
            ),
        )
        messages = (
            ProxyInputMessage(role="user", content="Build it"),
            ProxyInputMessage(role="assistant", content="", tool_calls=(tool_call,)),
            ProxyInputMessage(
                role="tool", content="file contents", tool_call_id=tool_call.id
            ),
        )

        pending = latest_pending_continuation(messages)

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.state.active_channel_id, "channel-1")
        self.assertEqual(pending.assistant_message.tool_calls[0].id, tool_call.id)
        self.assertEqual(pending.tool_results[0].content, "file contents")

    def test_latest_pending_continuation_allows_multiple_channel_actors(self) -> None:
        architect_tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(
                actor="architect",
                active_channel_id="channel-1",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-1",
                        name="debate",
                        participants=("architect", "coder"),
                    ),
                ),
            ),
        )
        coder_tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_2", name="glob", arguments_json="{}"),
            state=ContinuationState(
                actor="coder",
                active_channel_id="channel-1",
                channels=(
                    ChannelSnapshot(
                        channel_id="channel-1",
                        name="debate",
                        participants=("architect", "coder"),
                    ),
                ),
            ),
        )
        messages = (
            ProxyInputMessage(role="user", content="Debate it"),
            ProxyInputMessage(
                role="assistant",
                content="",
                tool_calls=(architect_tool_call, coder_tool_call),
            ),
            ProxyInputMessage(
                role="tool",
                content="architect context",
                tool_call_id=architect_tool_call.id,
            ),
            ProxyInputMessage(
                role="tool",
                content="coder context",
                tool_call_id=coder_tool_call.id,
            ),
        )

        pending = latest_pending_continuation(messages)

        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.state.active_channel_id, "channel-1")
        self.assertEqual(pending_actors(pending), ("architect", "coder"))

    def test_original_tool_call_id_extracts_wrapped_id(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_123", name="read_file", arguments_json="{}"),
            state=ContinuationState(actor="orchestrator"),
        )

        self.assertEqual(original_tool_call_id(tool_call.id), "call_123")

    def test_decode_original_tool_call_restores_wrapped_tool_metadata(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_123",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(actor="orchestrator"),
        )

        original = decode_original_tool_call(tool_call.id)

        self.assertIsNotNone(original)
        assert original is not None
        self.assertEqual(original.id, tool_call.id)
        self.assertEqual(original.name, "read_file")
        self.assertEqual(original.arguments_json, '{"path":"main.py"}')
