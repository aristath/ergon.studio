from __future__ import annotations

import unittest

from ergon_studio.proxy.chat_bridge import parse_chat_completion_request


class ProxyChatBridgeTests(unittest.TestCase):
    def test_parses_chat_completion_request_with_tool_history(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "ergon",
                "stream": True,
                "parallel_tool_calls": False,
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": [{"type": "text", "text": "Please fix it."}]},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":\"main.py\"}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "print('hello')",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "description": "Read a workspace file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                },
                                "required": ["path"],
                            },
                            "strict": True,
                        },
                    }
                ],
            }
        )

        self.assertEqual(request.model, "ergon")
        self.assertTrue(request.stream)
        self.assertFalse(request.parallel_tool_calls)
        self.assertEqual(request.latest_user_text(), "Please fix it.")
        self.assertEqual(request.messages[2].tool_calls[0].name, "read_file")
        self.assertEqual(request.messages[3].tool_call_id, "call_1")
        self.assertEqual(request.tools[0].name, "read_file")
        self.assertTrue(request.tools[0].strict)

    def test_rejects_non_function_tools(self) -> None:
        with self.assertRaises(ValueError):
            parse_chat_completion_request(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "web_search"}],
                }
            )

    def test_rejects_invalid_message_content(self) -> None:
        with self.assertRaises(ValueError):
            parse_chat_completion_request(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": {"text": "bad"}}],
                }
            )

    def test_accepts_empty_assistant_content_when_tool_calls_present(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "ergon",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run_command", "arguments": "{\"command\":\"pwd\"}"},
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(request.messages[0].content, "")
        self.assertEqual(request.messages[0].tool_calls[0].name, "run_command")

    def test_normalizes_developer_role_to_system(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "ergon",
                "messages": [
                    {"role": "developer", "content": "Always explain tradeoffs."},
                    {"role": "user", "content": "Build it"},
                ],
            }
        )

        self.assertEqual([message.role for message in request.messages], ["system", "user"])
        self.assertEqual(request.messages[0].content, "Always explain tradeoffs.")

    def test_normalizes_legacy_function_call_history(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "ergon",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"main.py\"}",
                        },
                    },
                    {
                        "role": "function",
                        "name": "read_file",
                        "content": "print('hello')",
                    },
                ],
            }
        )

        self.assertEqual(request.messages[0].role, "assistant")
        self.assertEqual(request.messages[0].tool_calls[0].id, "legacy_call_0")
        self.assertEqual(request.messages[0].tool_calls[0].name, "read_file")
        self.assertEqual(request.messages[1].role, "tool")
        self.assertEqual(request.messages[1].tool_call_id, "legacy_call_0")

    def test_parses_specific_function_tool_choice(self) -> None:
        request = parse_chat_completion_request(
            {
                "model": "ergon",
                "messages": [{"role": "user", "content": "Inspect main.py"}],
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

    def test_rejects_unknown_specific_function_tool_choice(self) -> None:
        with self.assertRaises(ValueError):
            parse_chat_completion_request(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Inspect main.py"}],
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
                        "function": {"name": "write_file"},
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
