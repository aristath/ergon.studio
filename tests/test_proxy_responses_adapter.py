from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyReasoningDeltaEvent
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


if __name__ == "__main__":
    unittest.main()
