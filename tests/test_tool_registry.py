from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.tool_registry import build_workspace_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def test_build_workspace_tool_registry_exposes_core_tool_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = build_workspace_tool_registry(Path(temp_dir))

            self.assertTrue(
                {
                    "read_file",
                    "write_file",
                    "patch_file",
                    "run_command",
                    "search_files",
                    "list_files",
                }.issubset(registry.keys())
            )

    def test_read_write_patch_and_search_tools_operate_within_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            registry = build_workspace_tool_registry(project_root)

            write_result = registry["write_file"].func(path="notes.txt", content="hello\nworld\n")
            self.assertEqual(write_result["path"], "notes.txt")

            read_result = registry["read_file"].func(path="notes.txt")
            self.assertEqual(read_result, "hello\nworld\n")

            patch_result = registry["patch_file"].func(
                path="notes.txt",
                old_text="world",
                new_text="team",
            )
            self.assertEqual(patch_result["replacements"], 1)
            self.assertEqual(registry["read_file"].func(path="notes.txt"), "hello\nteam\n")

            search_result = registry["search_files"].func(pattern="team")
            self.assertEqual(search_result[0]["path"], "notes.txt")
            self.assertEqual(search_result[0]["line_number"], 2)

    def test_run_command_executes_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            registry = build_workspace_tool_registry(project_root)

            result = registry["run_command"].func(command="pwd")

            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["cwd"], str(project_root))
            self.assertEqual(result["stdout"].strip(), str(project_root))

    def test_tools_reject_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            registry = build_workspace_tool_registry(project_root)

            with self.assertRaisesRegex(ValueError, "outside the project workspace"):
                registry["read_file"].func(path="../secret.txt")
