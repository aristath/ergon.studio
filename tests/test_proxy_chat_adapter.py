from __future__ import annotations

import json
import unittest

from ergon_studio.proxy.chat_adapter import build_chat_completion_response, encode_chat_stream_done, encode_chat_stream_event, encode_chat_stream_sse
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyReasoningDeltaEvent, ProxyToolCall, ProxyToolCallEvent


class ProxyChatAdapterTests(unittest.TestCase):
    def test_reasoning_delta_encodes_to_chat_chunk(self) -> None:
        payload = encode_chat_stream_event(
            ProxyReasoningDeltaEvent("Orchestrator: planning."),
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
        )

        self.assertEqual(payload["object"], "chat.completion.chunk")
        self.assertEqual(payload["choices"][0]["delta"]["reasoning_content"], "Orchestrator: planning.")
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_content_delta_encodes_to_chat_chunk(self) -> None:
        payload = encode_chat_stream_event(
            ProxyContentDeltaEvent("Done."),
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
        )

        self.assertEqual(payload["choices"][0]["delta"]["content"], "Done.")
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_tool_call_encodes_to_chat_chunk(self) -> None:
        payload = encode_chat_stream_event(
            ProxyToolCallEvent(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                )
            ),
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
        )

        tool_call = payload["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "call_1")
        self.assertEqual(tool_call["function"]["name"], "read_file")
        self.assertIsNone(payload["choices"][0]["finish_reason"])

    def test_finish_event_encodes_to_chat_chunk(self) -> None:
        payload = encode_chat_stream_event(
            ProxyFinishEvent("stop"),
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
        )

        self.assertEqual(payload["choices"][0]["delta"], {})
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_sse_encoding_wraps_json_payload(self) -> None:
        encoded = encode_chat_stream_sse(
            ProxyContentDeltaEvent("Done."),
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
        )

        self.assertTrue(encoded.startswith(b"data: "))
        payload = json.loads(encoded.decode("utf-8")[6:].strip())
        self.assertEqual(payload["choices"][0]["delta"]["content"], "Done.")

    def test_done_marker_is_openai_compatible(self) -> None:
        self.assertEqual(encode_chat_stream_done(), b"data: [DONE]\n\n")

    def test_non_stream_response_includes_tool_calls(self) -> None:
        payload = build_chat_completion_response(
            completion_id="chatcmpl_1",
            model="ergon",
            created_at=123,
            content="",
            finish_reason="tool_calls",
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                ),
            ),
        )

        message = payload["choices"][0]["message"]
        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")


if __name__ == "__main__":
    unittest.main()
