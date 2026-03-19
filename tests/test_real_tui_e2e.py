from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from textual.widgets import Input

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


def _calculator_entrypoints(project_root: Path) -> list[tuple[Path, list[str]]]:
    candidates: list[tuple[Path, list[str]]] = []
    seen_paths: set[Path] = set()

    def add_candidate(path: Path, commands: list[str]) -> None:
        if path in seen_paths or not path.exists():
            return
        seen_paths.add(path)
        candidates.append((path, commands))

    add_candidate(project_root / "calc.py", _cli_command_candidates("python3 calc.py"))
    add_candidate(project_root / "calculator.py", _cli_command_candidates("python3 calculator.py"))
    add_candidate(project_root / "cli.py", _cli_command_candidates("python3 cli.py"))

    for package_dir in sorted(project_root.iterdir()):
        if not package_dir.is_dir() or package_dir.name.startswith(".") or package_dir.name == "tests":
            continue
        package_main = package_dir / "main.py"
        if package_main.exists():
            add_candidate(package_main, _cli_command_candidates(f"python3 -m {package_dir.name}.main"))

    for path in sorted(project_root.rglob("*.py")):
        if ".ergon.studio" in path.parts or "tests" in path.parts or path.name.startswith("test_"):
            continue
        relative = path.relative_to(project_root)
        add_candidate(path, _cli_command_candidates(f"python3 {relative}"))
        module_name = ".".join(relative.with_suffix("").parts)
        if module_name:
            add_candidate(path, _cli_command_candidates(f"python3 -m {module_name}"))

    return candidates


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


@unittest.skipUnless(_model_available(), f"requires local llama-router model {MODEL_ID}")
class RealTuiE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_textual_app_can_build_end_to_end_from_main_chat(self) -> None:
        last_error: AssertionError | None = None
        for _attempt in range(2):
            try:
                await self._assert_textual_build_flow()
                return
            except AssertionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        self.fail("textual build flow did not run")

    async def _assert_textual_build_flow(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            _configure_local_runtime(runtime)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                composer = app.query_one("#composer-input", Input)
                composer.value = (
                    "We are starting from scratch. Build a tiny Python CLI calculator app. "
                    "First decide the approach, then implement it in this repo."
                )
                await app.on_input_submitted(SimpleNamespace(value=composer.value, input=composer))
                await self._wait_for(
                    lambda: self._workflow_finished(runtime),
                    pilot,
                    attempts=4200,
                )

                workflow_runs = runtime.list_workflow_runs()
                self.assertEqual(len(workflow_runs), 1)
                self.assertEqual(workflow_runs[0].state, "completed")

                entrypoints = _calculator_entrypoints(project_root)
                self.assertTrue(entrypoints, "expected a calculator implementation to be created")

                final_message = runtime.conversation_store.read_message_body(runtime.list_main_messages()[-1]).strip()
                self.assertIn("workflow", final_message.lower())
                self.assertIn("accepted", final_message.lower())
                self.assertIn("Changed files:", final_message)
                self.assertIn("Checks:", final_message)

                successful_result = None
                command_index = 0
                for _path, commands in entrypoints:
                    for command, require_output_prefix in _verification_commands(project_root, commands):
                        command_index += 1
                        command_result = runtime.run_workspace_command(
                            command,
                            created_at=10_000 + command_index,
                            thread_id=runtime.main_thread_id,
                            agent_id="tester",
                            require_approval=False,
                        )
                        if command_result["status"] != "completed":
                            continue
                        if command_result["exit_code"] != 0:
                            continue
                        if require_output_prefix and "5" not in str(command_result["stdout"]):
                            continue
                        successful_result = command_result
                        break
                    if successful_result is not None:
                        break
                self.assertIsNotNone(successful_result, "expected one calculator invocation to succeed")

    async def _wait_for(self, predicate, pilot, *, attempts: int = 200) -> None:
        for _ in range(attempts):
            if predicate():
                return
            await pilot.pause()
        self.fail("condition not reached before timeout")

    def _workflow_finished(self, runtime) -> bool:
        workflow_runs = runtime.list_workflow_runs()
        if not workflow_runs:
            return False
        state = workflow_runs[0].state
        if state in {"blocked", "failed"}:
            self.fail(f"workflow ended in unexpected state: {state}")
        return state == "completed"
