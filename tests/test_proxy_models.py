from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyFunctionTool, ProxyInputMessage, ProxyReasoningDeltaEvent, ProxyToolCall, ProxyToolCallEvent, ProxyTurnRequest


class ProxyModelsTests(unittest.TestCase):
    def test_turn_request_returns_latest_user_text(self) -> None:
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="system", content="rules"),
                ProxyInputMessage(role="user", content="first"),
                ProxyInputMessage(role="assistant", content="reply"),
                ProxyInputMessage(role="user", content="latest"),
            ),
        )

        self.assertEqual(request.latest_user_text(), "latest")

    def test_assistant_message_accepts_tool_calls(self) -> None:
        message = ProxyInputMessage(
            role="assistant",
            content="",
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                ),
            ),
        )

        self.assertEqual(message.tool_calls[0].name, "read_file")

    def test_non_assistant_message_rejects_tool_calls(self) -> None:
        with self.assertRaises(ValueError):
            ProxyInputMessage(
                role="user",
                content="hi",
                tool_calls=(ProxyToolCall(id="call_1", name="x", arguments_json="{}"),),
            )

    def test_tool_message_requires_tool_role_for_tool_call_id(self) -> None:
        with self.assertRaises(ValueError):
            ProxyInputMessage(
                role="assistant",
                content="result",
                tool_call_id="call_1",
            )

    def test_function_tool_requires_object_parameters(self) -> None:
        with self.assertRaises(TypeError):
            ProxyFunctionTool(
                name="read_file",
                description="Read a file",
                parameters="not-an-object",  # type: ignore[arg-type]
            )

    def test_finish_event_rejects_unknown_reason(self) -> None:
        with self.assertRaises(ValueError):
            ProxyFinishEvent("bad-reason")

    def test_output_events_validate_types(self) -> None:
        self.assertEqual(ProxyReasoningDeltaEvent("plan").delta, "plan")
        self.assertEqual(ProxyContentDeltaEvent("done").delta, "done")
        self.assertEqual(
            ProxyToolCallEvent(
                ProxyToolCall(id="call_1", name="read_file", arguments_json="{}")
            ).call.name,
            "read_file",
        )
