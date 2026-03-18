from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                    "web_lookup",
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
            self.assertEqual(result["status"], "completed")

    def test_run_command_can_delegate_to_a_custom_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            observed: list[tuple[str, int]] = []

            def handler(command: str, timeout: int) -> dict[str, int | str]:
                observed.append((command, timeout))
                return {
                    "command": command,
                    "cwd": str(project_root),
                    "exit_code": 0,
                    "stdout": "delegated",
                    "stderr": "",
                    "status": "completed",
                    "command_run_id": "command-run-1",
                }

            registry = build_workspace_tool_registry(
                project_root,
                run_command_handler=handler,
            )

            result = registry["run_command"].func(command="pwd", timeout=5)

            self.assertEqual(observed, [("pwd", 5)])
            self.assertEqual(result["command_run_id"], "command-run-1")

    def test_web_lookup_parses_search_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            registry = build_workspace_tool_registry(project_root)
            html = """
<html>
  <body>
    <a class="result__a" href="https://example.com/one">First Result</a>
    <div class="result__snippet">First snippet</div>
    <a class="result__a" href="https://example.com/two">Second Result</a>
    <div class="result__snippet">Second snippet</div>
  </body>
</html>
"""

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return html.encode("utf-8")

            with patch("ergon_studio.tool_registry.urlopen", return_value=FakeResponse()):
                results = registry["web_lookup"].func(query="ergon studio", limit=2)

            self.assertEqual(
                results,
                [
                    {
                        "title": "First Result",
                        "url": "https://example.com/one",
                        "snippet": "First snippet",
                    },
                    {
                        "title": "Second Result",
                        "url": "https://example.com/two",
                        "snippet": "Second snippet",
                    },
                ],
            )

    def test_tools_reject_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            registry = build_workspace_tool_registry(project_root)

            with self.assertRaisesRegex(ValueError, "outside the project workspace"):
                registry["read_file"].func(path="../secret.txt")
