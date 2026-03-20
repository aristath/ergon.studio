from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.cli import main
from ergon_studio.session_store import SessionStore


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

    def test_serve_command_bootstraps_registry_and_starts_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            with patch("ergon_studio.cli.serve_proxy") as serve_proxy:
                exit_code = main(
                    [
                        "serve",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "4242",
                    ]
                )

            self.assertEqual(exit_code, 0)
            serve_proxy.assert_called_once()
            _, kwargs = serve_proxy.call_args
            self.assertEqual(kwargs["host"], "0.0.0.0")
            self.assertEqual(kwargs["port"], 4242)

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

    def test_tui_pick_session_opens_picker_when_multiple_sessions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = SessionStore(paths)
            store.create_session(title="First lane", created_at=10)
            store.create_session(title="Second lane", created_at=20)

            captured: dict[str, object] = {}

            class FakeApp:
                def __init__(self, runtime, *, open_session_picker_on_mount=False, open_config_wizard_on_mount=False):
                    captured["runtime"] = runtime
                    captured["open_session_picker_on_mount"] = open_session_picker_on_mount
                    captured["open_config_wizard_on_mount"] = open_config_wizard_on_mount

                def run(self) -> None:
                    captured["ran"] = True

            with patch("ergon_studio.cli.ErgonStudioApp", FakeApp):
                exit_code = main(
                    [
                        "tui",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                        "--pick-session",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(captured["ran"])
            self.assertTrue(captured["open_session_picker_on_mount"])
            self.assertFalse(captured["open_config_wizard_on_mount"])

    def test_tui_opens_config_wizard_on_fresh_unconfigured_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            captured: dict[str, object] = {}

            class FakeApp:
                def __init__(self, runtime, *, open_session_picker_on_mount=False, open_config_wizard_on_mount=False):
                    captured["runtime"] = runtime
                    captured["open_session_picker_on_mount"] = open_session_picker_on_mount
                    captured["open_config_wizard_on_mount"] = open_config_wizard_on_mount

                def run(self) -> None:
                    captured["ran"] = True

            with patch("ergon_studio.cli.ErgonStudioApp", FakeApp):
                exit_code = main(
                    [
                        "tui",
                        "--project-root",
                        str(project_root),
                        "--home-dir",
                        str(home_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(captured["ran"])
            self.assertFalse(captured["open_session_picker_on_mount"])
            self.assertTrue(captured["open_config_wizard_on_mount"])
