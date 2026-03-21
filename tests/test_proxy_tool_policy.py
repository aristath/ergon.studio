from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyFunctionTool
from ergon_studio.proxy.tool_policy import validate_tool_choice


def _tool(name: str) -> ProxyFunctionTool:
    return ProxyFunctionTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object"},
    )


class ProxyToolPolicyTests(unittest.TestCase):
    def test_validate_tool_choice_accepts_required_function_selector(self) -> None:
        choice = validate_tool_choice(
            {
                "type": "function",
                "function": {"name": "read_file"},
            },
            tools=(_tool("read_file"),),
        )

        self.assertEqual(
            choice, {"type": "function", "function": {"name": "read_file"}}
        )

    def test_validate_tool_choice_rejects_unknown_function(self) -> None:
        with self.assertRaises(ValueError):
            validate_tool_choice(
                {
                    "type": "function",
                    "function": {"name": "write_file"},
                },
                tools=(_tool("read_file"),),
            )
