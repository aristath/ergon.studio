from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ergon_studio.runtime import load_runtime
from ergon_studio.tui.widgets import ComposerTextArea
from tests.real_test_support import (
    calculator_entrypoints,
    configure_local_runtime,
    should_run_real_model_tests,
    verification_commands,
)


@unittest.skipUnless(
    should_run_real_model_tests(),
    "requires ERGON_STUDIO_RUN_REAL_E2E=1 and local qwen3-coder-next-q40 availability",
)
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
            configure_local_runtime(runtime)
            app = ErgonStudioApp(runtime)

            async with app.run_test() as pilot:
                composer = app.query_one("#composer-input", ComposerTextArea)
                composer.value = (
                    "We are starting from scratch. Build a tiny Python CLI calculator app. "
                    "First decide the approach, then implement it in this repo."
                )
                await app.on_composer_text_area_submitted(
                    SimpleNamespace(value=composer.value, text_area=composer)
                )
                await self._wait_for(
                    lambda: self._workflow_finished(runtime),
                    pilot,
                    attempts=4200,
                )

                workflow_runs = runtime.list_workflow_runs()
                self.assertEqual(len(workflow_runs), 1)
                self.assertEqual(workflow_runs[0].state, "completed")

                entrypoints = calculator_entrypoints(project_root)
                self.assertTrue(entrypoints, "expected a calculator implementation to be created")

                final_message = runtime.conversation_store.read_message_body(runtime.list_main_messages()[-1]).strip()
                self.assertIn("workflow", final_message.lower())
                self.assertIn("accepted", final_message.lower())
                self.assertIn("Changed files:", final_message)
                self.assertIn("Checks:", final_message)

                successful_result = None
                command_index = 0
                for _path, commands in entrypoints:
                    for command, require_output_prefix in verification_commands(project_root, commands):
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
