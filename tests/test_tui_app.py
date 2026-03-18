from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.runtime import load_runtime


class TuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_renders_core_workspace_panels(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                self.assertEqual(app.title, "ergon.studio")
                self.assertIsNotNone(app.query_one("#main-chat"))
                self.assertIsNotNone(app.query_one("#tasks"))
                self.assertIsNotNone(app.query_one("#threads"))
                self.assertIsNotNone(app.query_one("#activity"))
                self.assertIsNotNone(app.query_one("#artifacts"))
                self.assertIsNotNone(app.query_one("#approvals"))
                self.assertIsNotNone(app.query_one("#memory"))
                self.assertIsNotNone(app.query_one("#settings"))
                self.assertIn("thread-main", app.query_one("#threads", Panel).body)

    async def test_app_renders_persisted_main_thread_messages(self) -> None:
        from ergon_studio.tui.app import ErgonStudioApp
        from ergon_studio.tui.app import Panel

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="Hello from the persisted main thread.",
                created_at=1_710_755_200,
            )
            app = ErgonStudioApp(runtime)

            async with app.run_test():
                main_chat = app.query_one("#main-chat", Panel)
                threads = app.query_one("#threads", Panel)
                self.assertIn("Hello from the persisted main thread.", main_chat.body)
                self.assertIn("thread-main", threads.body)
