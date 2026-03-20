from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from ergon_studio.config import save_global_config


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


def real_model_tests_enabled() -> bool:
    return os.environ.get("ERGON_STUDIO_RUN_REAL_E2E", "").strip() == "1"


def model_available() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    models = payload.get("data", [])
    if not isinstance(models, list):
        return False
    return any(isinstance(model, dict) and model.get("id") == MODEL_ID for model in models)


def should_run_real_model_tests() -> bool:
    return real_model_tests_enabled() and model_available()


def configure_local_runtime(runtime) -> None:
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


def cli_command_candidates(invocation: str) -> list[str]:
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


def calculator_entrypoints(project_root: Path) -> list[tuple[Path, list[str]]]:
    candidates: list[tuple[Path, list[str]]] = []
    seen_paths: set[Path] = set()

    def add_candidate(path: Path, commands: list[str]) -> None:
        if path in seen_paths or not path.exists():
            return
        seen_paths.add(path)
        candidates.append((path, commands))

    add_candidate(project_root / "calc.py", cli_command_candidates("python3 calc.py"))
    add_candidate(project_root / "calculator.py", cli_command_candidates("python3 calculator.py"))
    add_candidate(project_root / "cli.py", cli_command_candidates("python3 cli.py"))

    for package_dir in sorted(project_root.iterdir()):
        if not package_dir.is_dir() or package_dir.name.startswith(".") or package_dir.name == "tests":
            continue
        package_main = package_dir / "main.py"
        if package_main.exists():
            add_candidate(package_main, cli_command_candidates(f"python3 -m {package_dir.name}.main"))

    for path in sorted(project_root.rglob("*.py")):
        if ".ergon.studio" in path.parts or "tests" in path.parts or path.name.startswith("test_"):
            continue
        relative = path.relative_to(project_root)
        add_candidate(path, cli_command_candidates(f"python3 {relative}"))
        module_name = ".".join(relative.with_suffix("").parts)
        if module_name:
            add_candidate(path, cli_command_candidates(f"python3 -m {module_name}"))

    return candidates


def verification_commands(project_root: Path, entrypoint_commands: list[str]) -> list[tuple[str, bool]]:
    commands = [(command, True) for command in entrypoint_commands]
    if any(project_root.glob("test_*.py")) or any(project_root.glob("tests/test_*.py")):
        commands.extend(
            [
                ("python3 -m pytest -q", False),
                ("python3 -m unittest discover -s . -p 'test*.py'", False),
            ]
        )
    return commands
