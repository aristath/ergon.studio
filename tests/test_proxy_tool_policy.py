from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyFunctionTool
from ergon_studio.proxy.tool_policy import (
    resolve_agent_tool_policy,
    validate_tool_choice,
)


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

    def test_resolve_agent_tool_policy_filters_to_required_function(self) -> None:
        tools, tool_choice, parallel_tool_calls = resolve_agent_tool_policy(
            tools=(_tool("read_file"), _tool("write_file")),
            tool_choice={"type": "function", "function": {"name": "write_file"}},
            parallel_tool_calls=False,
        )

        self.assertEqual(tuple(tool.name for tool in tools), ("write_file",))
        self.assertEqual(
            tool_choice,
            {"type": "function", "function": {"name": "write_file"}},
        )
        self.assertFalse(parallel_tool_calls)

    def test_resolve_agent_tool_policy_disables_tools_for_none(self) -> None:
        tools, tool_choice, parallel_tool_calls = resolve_agent_tool_policy(
            tools=(_tool("read_file"),),
            tool_choice="none",
            parallel_tool_calls=None,
        )

        self.assertEqual(tools, ())
        self.assertEqual(tool_choice, "none")
        self.assertIsNone(parallel_tool_calls)

    def test_resolve_agent_tool_policy_rejects_malformed_function_selector(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "function object"):
            resolve_agent_tool_policy(
                tools=(_tool("read_file"),),
                tool_choice={"type": "function"},
                parallel_tool_calls=None,
            )
