from __future__ import annotations

import subprocess
from pathlib import Path

from agent_framework import FunctionTool, tool


def build_workspace_tool_registry(project_root: Path) -> dict[str, FunctionTool]:
    workspace_root = project_root.resolve()

    def resolve_path(path: str) -> Path:
        candidate = (workspace_root / path).resolve()
        if workspace_root not in (candidate, *candidate.parents):
            raise ValueError("path is outside the project workspace")
        return candidate

    @tool(name="read_file", approval_mode="never_require")
    def read_file(path: str) -> str:
        target = resolve_path(path)
        return target.read_text(encoding="utf-8")

    @tool(name="write_file", approval_mode="always_require")
    def write_file(path: str, content: str) -> dict[str, str]:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(workspace_root)), "status": "written"}

    @tool(name="patch_file", approval_mode="always_require")
    def patch_file(path: str, old_text: str, new_text: str) -> dict[str, int | str]:
        target = resolve_path(path)
        content = target.read_text(encoding="utf-8")
        replacements = content.count(old_text)
        if replacements == 0:
            raise ValueError("old_text was not found in the file")
        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "path": str(target.relative_to(workspace_root)),
            "replacements": 1,
        }

    @tool(name="list_files", approval_mode="never_require")
    def list_files(path: str = ".") -> list[str]:
        target = resolve_path(path)
        if target.is_file():
            return [str(target.relative_to(workspace_root))]
        return sorted(
            str(file_path.relative_to(workspace_root))
            for file_path in target.rglob("*")
            if file_path.is_file()
        )

    @tool(name="search_files", approval_mode="never_require")
    def search_files(pattern: str, path: str = ".") -> list[dict[str, int | str]]:
        target = resolve_path(path)
        files = [target] if target.is_file() else sorted(file_path for file_path in target.rglob("*") if file_path.is_file())
        matches: list[dict[str, int | str]] = []
        for file_path in files:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern in line:
                    matches.append(
                        {
                            "path": str(file_path.relative_to(workspace_root)),
                            "line_number": line_number,
                            "line": line,
                        }
                    )
        return matches

    @tool(name="run_command", approval_mode="always_require", kind="shell")
    def run_command(command: str, timeout: int = 60) -> dict[str, int | str]:
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(workspace_root),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    return {
        "read_file": read_file,
        "write_file": write_file,
        "patch_file": patch_file,
        "list_files": list_files,
        "search_files": search_files,
        "run_command": run_command,
    }
