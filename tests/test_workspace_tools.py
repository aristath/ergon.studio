from __future__ import annotations

import unittest

from ergon_studio.proxy.models import ProxyToolCall
from ergon_studio.proxy.orchestrator_tools import MalformedToolCallError
from ergon_studio.proxy.workspace_tools import (
    WORKSPACE_TOOL_NAMES,
    ListFilesAction,
    ReadFileAction,
    WriteFileAction,
    build_workspace_tools,
    is_workspace_tool_name,
    parse_list_files_action,
    parse_read_file_action,
    parse_write_file_action,
)


class WorkspaceToolNamesTests(unittest.TestCase):
    def test_workspace_tool_names_set(self) -> None:
        self.assertEqual(
            WORKSPACE_TOOL_NAMES, frozenset({"read_file", "write_file", "list_files"})
        )

    def test_is_workspace_tool_name_true_for_each(self) -> None:
        for name in ("read_file", "write_file", "list_files"):
            with self.subTest(name=name):
                self.assertTrue(is_workspace_tool_name(name))

    def test_is_workspace_tool_name_false_for_others(self) -> None:
        self.assertFalse(is_workspace_tool_name("open_channel"))
        self.assertFalse(is_workspace_tool_name("message_channel"))
        self.assertFalse(is_workspace_tool_name(""))
        self.assertFalse(is_workspace_tool_name("close_channel"))


class ParseReadFileActionTests(unittest.TestCase):
    def test_parse_returns_read_file_action(self) -> None:
        call = ProxyToolCall(
            id="t1",
            name="read_file",
            arguments_json='{"path": "/home/user/foo.py"}',
        )
        action = parse_read_file_action(call)
        self.assertIsInstance(action, ReadFileAction)
        self.assertEqual(action.path, "/home/user/foo.py")

    def test_parse_strips_whitespace_from_path(self) -> None:
        call = ProxyToolCall(
            id="t1",
            name="read_file",
            arguments_json='{"path": "  /home/user/foo.py  "}',
        )
        action = parse_read_file_action(call)
        self.assertEqual(action.path, "/home/user/foo.py")

    def test_parse_raises_on_missing_path(self) -> None:
        call = ProxyToolCall(id="t1", name="read_file", arguments_json="{}")
        with self.assertRaises(ValueError):
            parse_read_file_action(call)

    def test_parse_raises_on_empty_path(self) -> None:
        call = ProxyToolCall(
            id="t1", name="read_file", arguments_json='{"path": "  "}'
        )
        with self.assertRaises(ValueError):
            parse_read_file_action(call)

    def test_parse_raises_malformed_on_bad_json(self) -> None:
        call = ProxyToolCall(id="t1", name="read_file", arguments_json="{bad json}")
        with self.assertRaises(MalformedToolCallError):
            parse_read_file_action(call)


class ParseWriteFileActionTests(unittest.TestCase):
    def test_parse_returns_write_file_action(self) -> None:
        call = ProxyToolCall(
            id="t2",
            name="write_file",
            arguments_json='{"path": "/p/f.py", "content": "hello"}',
        )
        action = parse_write_file_action(call)
        self.assertIsInstance(action, WriteFileAction)
        self.assertEqual(action.path, "/p/f.py")
        self.assertEqual(action.content, "hello")

    def test_parse_allows_empty_content(self) -> None:
        call = ProxyToolCall(
            id="t2",
            name="write_file",
            arguments_json='{"path": "/p/f.py", "content": ""}',
        )
        action = parse_write_file_action(call)
        self.assertEqual(action.content, "")

    def test_parse_raises_on_missing_path(self) -> None:
        call = ProxyToolCall(
            id="t2",
            name="write_file",
            arguments_json='{"content": "x"}',
        )
        with self.assertRaises(ValueError):
            parse_write_file_action(call)

    def test_parse_raises_on_missing_content_key(self) -> None:
        call = ProxyToolCall(
            id="t2",
            name="write_file",
            arguments_json='{"path": "/p/f.py"}',
        )
        with self.assertRaises(ValueError):
            parse_write_file_action(call)

    def test_parse_raises_malformed_on_bad_json(self) -> None:
        call = ProxyToolCall(id="t2", name="write_file", arguments_json="{bad}")
        with self.assertRaises(MalformedToolCallError):
            parse_write_file_action(call)


class ParseListFilesActionTests(unittest.TestCase):
    def test_parse_returns_list_files_action(self) -> None:
        call = ProxyToolCall(
            id="t3",
            name="list_files",
            arguments_json='{"directory": "/home/user/proj"}',
        )
        action = parse_list_files_action(call)
        self.assertIsInstance(action, ListFilesAction)
        self.assertEqual(action.directory, "/home/user/proj")

    def test_parse_raises_on_missing_directory(self) -> None:
        call = ProxyToolCall(id="t3", name="list_files", arguments_json="{}")
        with self.assertRaises(ValueError):
            parse_list_files_action(call)

    def test_parse_raises_on_empty_directory(self) -> None:
        call = ProxyToolCall(
            id="t3", name="list_files", arguments_json='{"directory": ""}'
        )
        with self.assertRaises(ValueError):
            parse_list_files_action(call)

    def test_parse_raises_malformed_on_bad_json(self) -> None:
        call = ProxyToolCall(id="t3", name="list_files", arguments_json="{bad}")
        with self.assertRaises(MalformedToolCallError):
            parse_list_files_action(call)


class BuildWorkspaceToolsTests(unittest.TestCase):
    def test_returns_three_tools(self) -> None:
        tools = build_workspace_tools()
        self.assertEqual(len(tools), 3)

    def test_tool_names_match_workspace_tool_names(self) -> None:
        tools = build_workspace_tools()
        names = {t.name for t in tools}
        self.assertEqual(names, WORKSPACE_TOOL_NAMES)

    def test_each_tool_has_non_empty_description(self) -> None:
        for tool in build_workspace_tools():
            with self.subTest(name=tool.name):
                self.assertIsInstance(tool.description, str)
                self.assertGreater(len(tool.description), 0)

    def test_each_tool_has_parameters_with_properties(self) -> None:
        for tool in build_workspace_tools():
            with self.subTest(name=tool.name):
                self.assertIn("properties", tool.parameters)
                self.assertIn("required", tool.parameters)
