from __future__ import annotations

import subprocess
import time
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

from agent_framework import FunctionTool, tool


def build_workspace_tool_registry(
    project_root: Path,
    *,
    run_command_handler: Callable[[str, int], dict[str, int | str]] | None = None,
    write_file_handler: Callable[[str, str], dict[str, str]] | None = None,
    patch_file_handler: Callable[[str, str, str], dict[str, int | str]] | None = None,
    list_agents_handler: Callable[[], list[dict[str, object]]] | None = None,
    describe_agent_handler: Callable[[str], dict[str, object]] | None = None,
    list_workflows_handler: Callable[[], list[dict[str, object]]] | None = None,
    describe_workflow_handler: Callable[[str], dict[str, object]] | None = None,
    delegate_to_agent_handler: Callable[[str, str, str | None], Awaitable[dict[str, object]]] | None = None,
    run_workflow_handler: Callable[[str, str], Awaitable[dict[str, object]]] | None = None,
) -> dict[str, FunctionTool]:
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
        if write_file_handler is not None:
            return write_file_handler(path, content)
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(workspace_root)), "status": "written"}

    @tool(name="patch_file", approval_mode="always_require")
    def patch_file(path: str, old_text: str, new_text: str) -> dict[str, int | str]:
        if patch_file_handler is not None:
            return patch_file_handler(path, old_text, new_text)
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

    @tool(name="web_lookup", approval_mode="always_require", kind="network")
    def web_lookup(query: str, limit: int = 5) -> list[dict[str, str]]:
        request = Request(
            f"https://duckduckgo.com/html/?q={quote(query)}",
            headers={"User-Agent": "ergon.studio/0.1"},
        )
        with urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
        return _parse_web_lookup_results(html, limit)

    @tool(name="run_command", approval_mode="always_require", kind="shell")
    def run_command(command: str, timeout: int = 60) -> dict[str, int | str]:
        if run_command_handler is not None:
            return run_command_handler(command, timeout)
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
            "status": "completed",
            "created_at": int(time.time()),
        }

    @tool(name="list_agents", approval_mode="never_require")
    def list_agents() -> list[dict[str, object]]:
        """Return the available specialist agents and their roles."""
        if list_agents_handler is None:
            return []
        return list_agents_handler()

    @tool(name="describe_agent", approval_mode="never_require")
    def describe_agent(agent_id: str) -> dict[str, object]:
        """Return the editable definition details for a specific agent."""
        if describe_agent_handler is None:
            raise ValueError("agent descriptions are unavailable")
        return describe_agent_handler(agent_id)

    @tool(name="list_workflows", approval_mode="never_require")
    def list_workflows() -> list[dict[str, object]]:
        """Return the workflow catalog available to the orchestrator."""
        if list_workflows_handler is None:
            return []
        return list_workflows_handler()

    @tool(name="describe_workflow", approval_mode="never_require")
    def describe_workflow(workflow_id: str) -> dict[str, object]:
        """Return the editable definition details for a specific workflow."""
        if describe_workflow_handler is None:
            raise ValueError("workflow descriptions are unavailable")
        return describe_workflow_handler(workflow_id)

    @tool(name="delegate_to_agent", approval_mode="never_require")
    async def delegate_to_agent(agent_id: str, request: str, title: str | None = None) -> dict[str, object]:
        """Open a side thread with a specialist, send the request, and return the result."""
        if delegate_to_agent_handler is None:
            raise ValueError("agent delegation is unavailable")
        return await delegate_to_agent_handler(agent_id, request, title)

    @tool(name="run_workflow", approval_mode="never_require")
    async def run_workflow(workflow_id: str, goal: str) -> dict[str, object]:
        """Execute a workflow end to end and return the orchestrator review."""
        if run_workflow_handler is None:
            raise ValueError("workflow execution is unavailable")
        return await run_workflow_handler(workflow_id, goal)

    return {
        "read_file": read_file,
        "write_file": write_file,
        "patch_file": patch_file,
        "list_files": list_files,
        "search_files": search_files,
        "web_lookup": web_lookup,
        "run_command": run_command,
        "list_agents": list_agents,
        "describe_agent": describe_agent,
        "list_workflows": list_workflows,
        "describe_workflow": describe_workflow,
        "delegate_to_agent": delegate_to_agent,
        "run_workflow": run_workflow,
    }


def _parse_web_lookup_results(html: str, limit: int) -> list[dict[str, str]]:
    titles = re.findall(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for index, (url, title_html) in enumerate(titles[:limit]):
        title = _strip_html(title_html)
        snippet = _strip_html(snippets[index]) if index < len(snippets) else ""
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )
    return results


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value)).strip()
