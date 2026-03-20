from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyFunctionTool
from ergon_studio.proxy.tool_passthrough import build_declaration_tools, extract_tool_calls


class ProxyToolPassthroughTests(unittest.TestCase):
    def test_build_declaration_tools_preserves_names(self) -> None:
        tools = build_declaration_tools(
            (
                ProxyFunctionTool(
                    name="read_file",
                    description="Read a file",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                ),
            )
        )

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "read_file")

    def test_extract_tool_calls_reads_function_call_contents(self) -> None:
        response = type(
            "Response",
            (),
            {
                "messages": [
                    type(
                        "Message",
                        (),
                        {
                            "contents": [
                                type(
                                    "Content",
                                    (),
                                    {
                                        "type": "function_call",
                                        "call_id": "call_1",
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"main.py\"}",
                                    },
                                )()
                            ]
                        },
                    )()
                ]
            },
        )()

        tool_calls = extract_tool_calls(response)

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "read_file")
        self.assertEqual(tool_calls[0].arguments_json, "{\"path\":\"main.py\"}")
