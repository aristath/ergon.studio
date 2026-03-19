from __future__ import annotations

import json
import re
import tempfile
import unittest
import urllib.request
from pathlib import Path

from ergon_studio.config import save_global_config
from ergon_studio.runtime import load_runtime


MODEL_ID = "qwen3-coder-next-q40"
ALL_ROLES = {
    "orchestrator": "local",
    "architect": "local",
    "coder": "local",
    "reviewer": "local",
    "fixer": "local",
    "researcher": "local",
    "tester": "local",
    "documenter": "local",
    "brainstormer": "local",
    "designer": "local",
}


def _model_available() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    models = payload.get("data", [])
    if not isinstance(models, list):
        return False
    return any(isinstance(model, dict) and model.get("id") == MODEL_ID for model in models)


def _configure_local_runtime(runtime) -> None:
    save_global_config(
        runtime.paths.config_path,
        {
            "providers": {
                "local": {
                    "type": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "api_key": "not-needed",
                    "model": MODEL_ID,
                    "capabilities": {
                        "tool_calling": True,
                        "structured_output": True,
                    },
                }
            },
            "role_assignments": dict(ALL_ROLES),
            "approvals": {"default": "auto"},
            "ui": {},
        },
    )
    runtime.reload_registry()


def _calculator_entrypoint(project_root: Path) -> tuple[Path, list[str]] | None:
    candidates: list[tuple[Path, list[str]]] = []
    seen_paths: set[Path] = set()

    def add_candidate(path: Path, commands: list[str]) -> None:
        if path in seen_paths or not path.exists():
            return
        seen_paths.add(path)
        candidates.append((path, commands))

    calc_path = project_root / "calc.py"
    add_candidate(calc_path, _cli_command_candidates("python3 calc.py"))
    calculator_path = project_root / "calculator.py"
    add_candidate(calculator_path, _cli_command_candidates("python3 calculator.py"))
    cli_path = project_root / "cli.py"
    add_candidate(cli_path, _cli_command_candidates("python3 cli.py"))

    for package_dir in sorted(project_root.iterdir()):
        if not package_dir.is_dir() or package_dir.name.startswith(".") or package_dir.name == "tests":
            continue
        package_main = package_dir / "main.py"
        if package_main.exists():
            module = f"{package_dir.name}.main"
            add_candidate(
                package_main,
                _cli_command_candidates(f"python3 -m {module}"),
            )

    for path in sorted(project_root.rglob("*.py")):
        if ".ergon.studio" in path.parts or "tests" in path.parts:
            continue
        relative = path.relative_to(project_root)
        if path.name.startswith("test_"):
            continue
        command_base = str(relative)
        add_candidate(
            path,
            _cli_command_candidates(f"python3 {command_base}"),
        )

    return candidates[0] if candidates else None


def _verification_commands(project_root: Path, entrypoint_commands: list[str]) -> list[tuple[str, bool]]:
    commands = [(command, True) for command in entrypoint_commands]
    if any(project_root.glob("test_*.py")) or any(project_root.glob("tests/test_*.py")):
        commands.extend(
            [
                ("python3 -m pytest -q", False),
                ("python3 -m unittest discover -s . -p 'test*.py'", False),
            ]
        )
    return commands


def _workspace_python_files(project_root: Path) -> list[Path]:
    return [
        path
        for path in project_root.rglob("*.py")
        if ".ergon.studio" not in path.parts
    ]


def _cli_command_candidates(invocation: str) -> list[str]:
    return [
        f"{invocation} 2 + 3",
        f"{invocation} add 2 3",
        f"{invocation} 2 add 3",
        f"{invocation} + 2 3",
        f"{invocation} 2 3 +",
        f"{invocation} 2 3 add",
        f"{invocation} --num1 2 --num2 3 --op add",
        f"{invocation} --a 2 --b 3 --op add",
        f"{invocation} '2 + 3'",
        f"printf '2\\n+\\n3\\n' | {invocation}",
    ]


@unittest.skipUnless(_model_available(), f"requires local llama-router model {MODEL_ID}")
class RealE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_real_coder_can_create_a_file_with_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            _configure_local_runtime(runtime)

            result = await runtime.delegate_to_agent(
                agent_id="coder",
                request=(
                    "Create a file named hello.txt in the repo root with exactly this content: "
                    "hello from coder\\n. Use the file tools. Do not only describe it."
                ),
                title="Direct write test",
                created_at=1,
                parent_thread_id=runtime.main_thread_id,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual((project_root / "hello.txt").read_text(encoding="utf-8").rstrip("\n"), "hello from coder")
            self.assertEqual(
                [(tool_call.tool_name, tool_call.status, tool_call.agent_id) for tool_call in runtime.list_tool_calls()],
                [("write_file", "completed", "coder")],
            )

    async def test_real_orchestrator_can_complete_a_vague_build_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            _configure_local_runtime(runtime)

            await runtime.send_user_message_to_orchestrator(
                body=(
                    "We are starting from scratch. Build a tiny Python CLI calculator app. "
                    "First decide the approach, then implement it in this repo."
                ),
                created_at=1,
            )

            messages = runtime.list_main_messages()
            self.assertGreaterEqual(len(messages), 2)
            final_message = runtime.conversation_store.read_message_body(messages[-1]).strip()
            self.assertIn("workflow", final_message)
            self.assertTrue("ACCEPTED" in final_message or "Accepted" in final_message)

            workflow_runs = runtime.list_workflow_runs()
            self.assertEqual(len(workflow_runs), 1)
            self.assertEqual(workflow_runs[0].state, "completed")

            task_titles = [task.title for task in runtime.list_tasks()]
            self.assertGreaterEqual(len(task_titles), 1)

            entrypoint = _calculator_entrypoint(project_root)
            self.assertIsNotNone(entrypoint, "expected a runnable calculator entrypoint to be created")
            assert entrypoint is not None

            successful_result = None
            for index, (command, require_output_prefix) in enumerate(
                _verification_commands(project_root, entrypoint[1]),
                start=1,
            ):
                command_result = runtime.run_workspace_command(
                    command,
                    created_at=200 + index,
                    thread_id=runtime.main_thread_id,
                    agent_id="tester",
                    require_approval=False,
                )
                if command_result["status"] != "completed":
                    continue
                stdout = str(command_result["stdout"]).strip()
                if command_result["exit_code"] != 0:
                    continue
                if require_output_prefix and re.search(r"\b5(?:\\.0+)?\b", stdout) is None:
                    continue
                if not require_output_prefix or re.search(r"\b5(?:\\.0+)?\b", stdout) is not None:
                    successful_result = command_result
                    break
            self.assertIsNotNone(successful_result, "expected one calculator invocation to succeed and print a result starting with 5")

    async def test_real_orchestrator_can_discuss_then_build_from_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            _configure_local_runtime(runtime)

            await runtime.send_user_message_to_orchestrator(
                body="We need a tiny Python CLI calculator. Talk me through the approach first before building anything.",
                created_at=1,
            )

            first_pass_messages = runtime.list_main_messages()
            self.assertGreaterEqual(len(first_pass_messages), 2)
            self.assertEqual(runtime.list_workflow_runs(), [])
            self.assertIsNone(_calculator_entrypoint(project_root))

            await runtime.send_user_message_to_orchestrator(
                body="That sounds fine. Please proceed and build it.",
                created_at=100,
            )

            workflow_runs = runtime.list_workflow_runs()
            self.assertEqual(len(workflow_runs), 1)
            self.assertEqual(workflow_runs[0].state, "completed")

            final_message = runtime.conversation_store.read_message_body(runtime.list_main_messages()[-1]).strip()
            self.assertTrue("workflow" in final_message or "built" in final_message.lower())
            self.assertTrue(_calculator_entrypoint(project_root) is not None or bool(_workspace_python_files(project_root)))
