from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyOutputItemRef, ProxyReasoningDeltaEvent, ProxyToolCall, ProxyToolCallEvent
from ergon_studio.proxy.responses_adapter import build_responses_response, encode_responses_stream_events


class ProxyResponsesAdapterTests(unittest.TestCase):
    def test_builds_non_stream_response_object(self) -> None:
        payload = build_responses_response(
            response_id="resp_1",
            model="ergon",
            created_at=123,
            content="Done.",
            reasoning="Planned carefully.",
        )

        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["output_text"], "Done.")
        self.assertEqual(payload["output"][0]["type"], "reasoning")
        self.assertEqual(payload["output"][1]["type"], "message")

    def test_encodes_reasoning_and_content_events(self) -> None:
        reasoning = encode_responses_stream_events(
            event=ProxyReasoningDeltaEvent("Plan."),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=1,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
        )
        content = encode_responses_stream_events(
            event=ProxyContentDeltaEvent("Done."),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=2,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
        )
        finished = encode_responses_stream_events(
            event=ProxyFinishEvent("stop"),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=3,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
        )

        self.assertEqual(reasoning[0]["type"], "response.reasoning_text.delta")
        self.assertEqual(content[0]["type"], "response.output_text.delta")
        self.assertEqual(finished[-1]["type"], "response.completed")

    def test_builds_function_call_items(self) -> None:
        payload = build_responses_response(
            response_id="resp_1",
            model="ergon",
            created_at=123,
            content="",
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                ),
            ),
        )

        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["call_id"], "call_1")
        self.assertEqual(len(payload["output"]), 1)

    def test_build_responses_response_preserves_recorded_output_order(self) -> None:
        payload = build_responses_response(
            response_id="resp_1",
            model="ergon",
            created_at=123,
            content="Draft first.",
            tool_calls=(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                ),
            ),
            output_items=(
                ProxyOutputItemRef(kind="content"),
                ProxyOutputItemRef(kind="tool_call", call_id="call_1"),
            ),
        )

        self.assertEqual([item["type"] for item in payload["output"]], ["message", "function_call"])

    def test_encodes_tool_call_stream_events(self) -> None:
        payload = encode_responses_stream_events(
            event=ProxyToolCallEvent(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                )
            ),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=3,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
        )

        self.assertEqual(payload[0]["type"], "response.output_item.added")
        self.assertEqual(payload[0]["item"]["type"], "function_call")
        self.assertEqual(payload[0]["output_index"], 0)

    def test_finish_event_can_skip_output_done_when_no_content_was_streamed(self) -> None:
        payload = encode_responses_stream_events(
            event=ProxyFinishEvent("tool_calls"),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=3,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
            include_output_done=False,
        )

        self.assertEqual([item["type"] for item in payload], ["response.completed"])

    def test_response_stream_output_indexes_can_be_offset(self) -> None:
        reasoning = encode_responses_stream_events(
            event=ProxyReasoningDeltaEvent("Plan."),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=1,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
            reasoning_output_index=0,
            message_output_index=2,
        )
        tool = encode_responses_stream_events(
            event=ProxyToolCallEvent(
                ProxyToolCall(
                    id="call_1",
                    name="read_file",
                    arguments_json="{\"path\":\"main.py\"}",
                ),
                index=0,
            ),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=2,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
            tool_output_index=1,
        )
        content = encode_responses_stream_events(
            event=ProxyContentDeltaEvent("Done."),
            response_id="resp_1",
            model="ergon",
            created_at=123,
            sequence_number=3,
            reasoning_item_id="rs_1",
            message_item_id="msg_1",
            message_output_index=2,
        )

        self.assertEqual(reasoning[0]["output_index"], 0)
        self.assertEqual(tool[0]["output_index"], 1)
        self.assertEqual(content[0]["output_index"], 2)


if __name__ == "__main__":
    unittest.main()
