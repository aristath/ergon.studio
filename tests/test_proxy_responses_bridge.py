from __future__ import annotations

import unittest

from ergon_studio.proxy.responses_bridge import parse_responses_request


class ProxyResponsesBridgeTests(unittest.TestCase):
    def test_parses_string_input_as_single_user_message(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": "Build it",
                "stream": True,
            }
        )

        self.assertEqual(request.model, "ergon")
        self.assertTrue(request.stream)
        self.assertEqual(request.messages[0].role, "user")
        self.assertEqual(request.messages[0].content, "Build it")

    def test_parses_message_and_function_call_output_items(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Need a file read."}],
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "{\"contents\":\"print(1)\"}",
                    },
                ],
            }
        )

        self.assertEqual(request.messages[0].role, "assistant")
        self.assertEqual(request.messages[1].role, "tool")
        self.assertEqual(request.messages[1].tool_call_id, "call_1")


if __name__ == "__main__":
    unittest.main()
