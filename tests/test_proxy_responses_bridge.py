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

    def test_parses_top_level_instructions_as_leading_system_message(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "instructions": "Always explain tradeoffs.",
                "input": "Build it",
            }
        )

        self.assertEqual([message.role for message in request.messages], ["system", "user"])
        self.assertEqual(request.messages[0].content, "Always explain tradeoffs.")
        self.assertEqual(request.messages[1].content, "Build it")

    def test_allows_instruction_only_requests(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "instructions": "You are reviewing prior work.",
            }
        )

        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].role, "system")
        self.assertEqual(request.messages[0].content, "You are reviewing prior work.")

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

    def test_function_call_output_accepts_fallback_tool_call_id(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "   ",
                        "tool_call_id": "call_2",
                        "output": "ok",
                    },
                ],
            }
        )

        self.assertEqual(request.messages[0].tool_call_id, "call_2")

    def test_function_call_output_requires_non_empty_call_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            parse_responses_request(
                {
                    "model": "ergon",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "   ",
                            "tool_call_id": "",
                            "output": "ok",
                        },
                    ],
                }
            )

    def test_function_call_output_requires_identifier_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "must include call_id or tool_call_id"):
            parse_responses_request(
                {
                    "model": "ergon",
                    "input": [
                        {
                            "type": "function_call_output",
                            "output": "ok",
                        },
                    ],
                }
            )

    def test_parses_function_call_items_as_assistant_tool_history(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{\"path\":\"main.py\"}",
                    }
                ],
            }
        )

        self.assertEqual(request.messages[0].role, "assistant")
        self.assertEqual(request.messages[0].tool_calls[0].id, "call_1")
        self.assertEqual(request.messages[0].tool_calls[0].name, "read_file")

    def test_normalizes_developer_role_to_system(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": "Always explain tradeoffs.",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Build it",
                    },
                ],
            }
        )

        self.assertEqual([message.role for message in request.messages], ["system", "user"])
        self.assertEqual(request.messages[0].content, "Always explain tradeoffs.")

    def test_parses_specific_function_tool_choice(self) -> None:
        request = parse_responses_request(
            {
                "model": "ergon",
                "input": "Inspect main.py",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "read_file"},
                },
            }
        )

        self.assertEqual(request.tool_choice, {"type": "function", "function": {"name": "read_file"}})
