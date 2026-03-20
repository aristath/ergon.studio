from __future__ import annotations

import unittest

from ergon_studio.proxy.continuation import ContinuationState, decode_continuation_from_tool_call_id, encode_continuation_tool_call, latest_continuation
from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall


class ProxyContinuationTests(unittest.TestCase):
    def test_encode_and_decode_round_trip(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json="{\"path\":\"main.py\"}",
            ),
            state=ContinuationState(
                mode="workflow",
                agent_id="architect",
                workflow_id="standard-build",
                step_index=2,
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.mode, "workflow")
        self.assertEqual(decoded.agent_id, "architect")
        self.assertEqual(decoded.workflow_id, "standard-build")
        self.assertEqual(decoded.step_index, 2)
        self.assertEqual(decoded.request_text, None)
        self.assertEqual(decoded.goal, None)
        self.assertEqual(decoded.current_brief, None)
        self.assertEqual(decoded.workflow_outputs, ())

    def test_encode_and_decode_round_trip_with_context(self) -> None:
        encoded = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_2",
                name="write_file",
                arguments_json="{\"path\":\"main.py\"}",
            ),
            state=ContinuationState(
                mode="workflow",
                agent_id="coder",
                workflow_id="standard-build",
                step_index=1,
                request_text="Implement A",
                goal="Build calculator",
                current_brief="Updating main.py",
                workflow_outputs=("architect: use main.py",),
            ),
        )

        decoded = decode_continuation_from_tool_call_id(encoded.id)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.request_text, "Implement A")
        self.assertEqual(decoded.goal, "Build calculator")
        self.assertEqual(decoded.current_brief, "Updating main.py")
        self.assertEqual(decoded.workflow_outputs, ("architect: use main.py",))

    def test_latest_continuation_uses_latest_tool_message(self) -> None:
        messages = (
            ProxyInputMessage(role="user", content="Build it"),
            ProxyInputMessage(
                role="tool",
                content="first output",
                tool_call_id=encode_continuation_tool_call(
                    ProxyToolCall(id="call_1", name="read_file", arguments_json="{}"),
                    state=ContinuationState(mode="delegate", agent_id="coder"),
                ).id,
            ),
            ProxyInputMessage(
                role="tool",
                content="second output",
                tool_call_id=encode_continuation_tool_call(
                    ProxyToolCall(id="call_2", name="run_command", arguments_json="{}"),
                    state=ContinuationState(mode="act", agent_id="orchestrator"),
                ).id,
            ),
        )

        decoded = latest_continuation(messages)

        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.mode, "act")
        self.assertEqual(decoded.agent_id, "orchestrator")


if __name__ == "__main__":
    unittest.main()
