from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.config import save_global_config
from ergon_studio.evals import run_builtin_evals, summarize_results, write_eval_report
from ergon_studio.runtime import load_runtime


class EvalTests(unittest.TestCase):
    def test_run_builtin_evals_reports_results_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            save_global_config(
                runtime.paths.config_path,
                {
                    "providers": {
                        "local": {
                            "type": "openai_chat",
                            "base_url": "http://localhost:8080/v1",
                            "api_key": "not-needed",
                            "model": "qwen2.5-coder",
                        }
                    },
                    "role_assignments": {"orchestrator": "local"},
                    "approvals": {},
                    "ui": {},
                },
            )
            runtime.reload_registry()

            results = run_builtin_evals(runtime)
            report_path = write_eval_report(runtime, results, created_at=10)

            self.assertEqual(report_path.name, "eval-10.md")
            self.assertIn("passed=", summarize_results(results))
            self.assertTrue(report_path.exists())
            self.assertIn("workflow_compilation", report_path.read_text(encoding="utf-8"))
