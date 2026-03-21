from __future__ import annotations

import unittest

from ergon_studio.proxy.continuation import (
    ContinuationState,
    decode_continuation_from_tool_call_id,
    decode_original_tool_call,
    encode_continuation_tool_call,
    latest_continuation,
    latest_pending_continuation,
    original_tool_call_id,
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
                mode="workroom",
                agent_id="architect",
                workroom_id="standard-build",
                progress_index=2,
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.mode, "workroom")
        self.assertEqual(decoded.agent_id, "architect")
        self.assertEqual(decoded.workroom_id, "standard-build")
        self.assertEqual(decoded.workroom_participants, ())
        self.assertEqual(decoded.last_stage_outputs, ())
        self.assertFalse(decoded.last_stage_parallel_attempts)
        self.assertEqual(decoded.progress_index, 2)
        self.assertEqual(decoded.member_index, None)
        self.assertEqual(decoded.message, None)
        self.assertEqual(decoded.goal, None)
        self.assertEqual(decoded.current_brief, None)
        self.assertEqual(decoded.workroom_outputs, ())

    def test_encode_and_decode_round_trip_with_context(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_2",
                name="write_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(
                mode="workroom",
                agent_id="coder",
                workroom_id="standard-build",
                workroom_participants=("coder", "coder", "coder", "reviewer"),
                workroom_request="Polish the selected candidate.",
                last_stage_outputs=("coder[1]: Idea A", "coder[2]: Idea B"),
                last_stage_parallel_attempts=True,
                progress_index=1,
                member_index=0,
                message="Implement A",
                goal="Build calculator",
                current_brief="Updating main.py",
                workroom_outputs=("architect: use main.py",),
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.message, "Implement A")
        self.assertEqual(decoded.goal, "Build calculator")
        self.assertEqual(decoded.current_brief, "Updating main.py")
        self.assertEqual(
            decoded.workroom_participants,
            ("coder", "coder", "coder", "reviewer"),
        )
        self.assertEqual(decoded.workroom_request, "Polish the selected candidate.")
        self.assertEqual(
            decoded.last_stage_outputs,
            ("coder[1]: Idea A", "coder[2]: Idea B"),
        )
        self.assertTrue(decoded.last_stage_parallel_attempts)
        self.assertEqual(decoded.workroom_outputs, ("architect: use main.py",))
        self.assertEqual(decoded.member_index, 0)

    def test_latest_continuation_uses_latest_tool_message(self) -> None:
        first_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(mode="delegate", agent_id="coder"),
        )
        second_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_2", name="run_command", arguments_json="{}"),
            state=ContinuationState(mode="orchestrator", agent_id="orchestrator"),
        )
        messages = (
            ProxyInputMessage(role="user", content="Build it"),
            ProxyInputMessage(role="assistant", content="", tool_calls=(first_call,)),
            ProxyInputMessage(
                role="tool",
                content="first output",
                tool_call_id=first_call.id,
            ),
            ProxyInputMessage(role="assistant", content="", tool_calls=(second_call,)),
            ProxyInputMessage(
                role="tool",
                content="second output",
                tool_call_id=second_call.id,
            ),
        )

        decoded = latest_continuation(messages)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.mode, "orchestrator")
        self.assertEqual(decoded.agent_id, "orchestrator")

    def test_latest_pending_continuation_requires_tool_loop_tail(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(mode="delegate", agent_id="coder"),
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
        self.assertIsNone(latest_continuation(messages))

    def test_latest_pending_continuation_returns_matching_assistant_and_results(
        self,
    ) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
            state=ContinuationState(mode="delegate", agent_id="coder"),
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
        self.assertEqual(pending.state.mode, "delegate")
        self.assertEqual(pending.assistant_message.tool_calls[0].id, tool_call.id)
        self.assertEqual(pending.tool_results[0].content, "file contents")

    def test_original_tool_call_id_extracts_wrapped_id(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(id="call_123", name="read_file", arguments_json="{}"),
            state=ContinuationState(mode="orchestrator", agent_id="orchestrator"),
        )

        self.assertEqual(original_tool_call_id(tool_call.id), "call_123")

    def test_decode_original_tool_call_restores_wrapped_tool_metadata(self) -> None:
        tool_call = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_123",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(mode="orchestrator", agent_id="orchestrator"),
        )

        original = decode_original_tool_call(tool_call.id)

        self.assertIsNotNone(original)
        assert original is not None
        self.assertEqual(original.id, tool_call.id)
        self.assertEqual(original.name, "read_file")
        self.assertEqual(original.arguments_json, '{"path":"main.py"}')
