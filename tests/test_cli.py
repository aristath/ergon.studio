from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ergon_studio.cli import main


class CliTests(unittest.TestCase):
    def test_bootstrap_command_initializes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "bootstrap",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((project_root / ".ergon.studio" / "project.json").exists())
            self.assertIn("project_uuid", stdout.getvalue())

    def test_eval_command_runs_builtin_evals_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "eval",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("passed=", output)
            self.assertIn("session_id=", output)
            self.assertIn("report=", output)
            self.assertIn("workflow_compilation", output)

    def test_sessions_new_and_list_commands_manage_project_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            new_stdout = io.StringIO()
            with redirect_stdout(new_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "new",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        "--title",
                        "Bugfix lane",
                    ]
                )

            self.assertEqual(exit_code, 0)
            new_output = new_stdout.getvalue()
            self.assertIn("session_id=session-", new_output)
            self.assertIn("title=Bugfix lane", new_output)

            list_stdout = io.StringIO()
            with redirect_stdout(list_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "list",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Bugfix lane", list_stdout.getvalue())

    def test_sessions_rename_and_archive_commands_update_session_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            new_stdout = io.StringIO()
            with redirect_stdout(new_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "new",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        "--title",
                        "Draft lane",
                    ]
                )

            self.assertEqual(exit_code, 0)
            created_lines = new_stdout.getvalue().splitlines()
            session_id = created_lines[0].split("=", 1)[1]

            rename_stdout = io.StringIO()
            with redirect_stdout(rename_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "rename",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        session_id,
                        "--title",
                        "Renamed lane",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("title=Renamed lane", rename_stdout.getvalue())

            archive_stdout = io.StringIO()
            with redirect_stdout(archive_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "archive",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        session_id,
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("archived=true", archive_stdout.getvalue())

            list_stdout = io.StringIO()
            with redirect_stdout(list_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "list",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("Renamed lane", list_stdout.getvalue())

            all_stdout = io.StringIO()
            with redirect_stdout(all_stdout):
                exit_code = main(
                    [
                        "sessions",
                        "list",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        "--all",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Renamed lane", all_stdout.getvalue())
            self.assertIn("archived", all_stdout.getvalue())
