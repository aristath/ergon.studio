from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from ergon_studio.runtime import load_runtime
from tests.real_test_support import (
    calculator_entrypoints,
    configure_local_runtime,
    should_run_real_model_tests,
    verification_commands,
)


def _workspace_python_files(project_root: Path) -> list[Path]:
    return [
        path
        for path in project_root.rglob("*.py")
        if ".ergon.studio" not in path.parts
    ]


@unittest.skipUnless(
    should_run_real_model_tests(),
    "requires ERGON_STUDIO_RUN_REAL_E2E=1 and local qwen3-coder-next-q40 availability",
)
class RealE2ETests(unittest.IsolatedAsyncioTestCase):
    async def test_real_coder_can_create_a_file_with_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)

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
            tool_calls = [
                (tool_call.tool_name, tool_call.status, tool_call.agent_id)
                for tool_call in runtime.list_tool_calls()
            ]
            self.assertIn(("write_file", "completed", "coder"), tool_calls)
            self.assertFalse(
                any(status == "failed" for _, status, _ in tool_calls),
                msg=f"unexpected tool calls: {tool_calls}",
            )

    async def test_real_researcher_can_answer_from_retrieval_without_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "Dockerfile").write_text(
                "FROM python:3.12-slim\nRUN echo ready\n",
                encoding="utf-8",
            )

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)
            runtime.save_agent_definition_text(
                agent_id="researcher",
                text="""---
id: researcher
name: Researcher
role: researcher
temperature: 0.0
---
## Identity
You are the research specialist for the AI firm.

## Responsibilities
Answer questions from the available project context.

## Rules
Rely on the provided context. Do not invent files.

## Output Style
Be short and concrete.
""",
                created_at=1,
            )

            result = await runtime.delegate_to_agent(
                agent_id="researcher",
                request="Which file uses the python:3.12-slim base image? Reply with the filename only.",
                title="Retrieval-only lookup",
                created_at=2,
                parent_thread_id=runtime.main_thread_id,
            )

            self.assertEqual(result["status"], "completed")
            self.assertIn("Dockerfile", str(result["result"]))
            self.assertEqual(runtime.list_tool_calls(), [])

    async def test_real_sessions_share_workspace_but_not_chat_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(first)

            created = await first.delegate_to_agent(
                agent_id="coder",
                request=(
                    "Create a file named session_note.txt in the repo root with exactly this content: "
                    "shared workspace proof\\n. Use the file tools."
                ),
                title="Session one write",
                created_at=1,
                parent_thread_id=first.main_thread_id,
            )

            self.assertEqual(created["status"], "completed")
            self.assertTrue((project_root / "session_note.txt").exists())
            self.assertEqual(len(first.list_main_messages()), 0)

            second = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Parallel lane",
            )
            configure_local_runtime(second)

            self.assertNotEqual(second.main_session_id, first.main_session_id)
            self.assertEqual(second.list_main_messages(), [])

            second.save_agent_definition_text(
                agent_id="researcher",
                text="""---
id: researcher
name: Researcher
role: researcher
temperature: 0.0
---
## Identity
You are the research specialist for the AI firm.

## Responsibilities
Answer questions from the available project context.

## Rules
Rely on the provided context. Do not invent files.

## Output Style
Be short and concrete.
""",
                created_at=2,
            )

            result = await second.delegate_to_agent(
                agent_id="researcher",
                request="Which file contains the text 'shared workspace proof'? Reply with the filename only.",
                title="Cross-session retrieval lookup",
                created_at=3,
                parent_thread_id=second.main_thread_id,
            )

            self.assertEqual(result["status"], "completed")
            self.assertIn("session_note.txt", str(result["result"]))
            self.assertEqual(second.list_main_messages(), [])

    async def test_real_debate_workflow_uses_a_shared_workroom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)

            result = await runtime.run_workflow(
                workflow_id="debate",
                goal=(
                    "The repo is empty and we are designing a new Python CLI from scratch. "
                    "Debate whether we should prefer Typer or argparse as the default choice. "
                    "End with one clear recommendation and the top two reasons."
                ),
                created_at=1,
                parent_thread_id=runtime.main_thread_id,
            )

            self.assertIn(result["status"], {"completed", "blocked"})
            self.assertTrue(result["review_summary"])

            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)

            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            messages = runtime.list_thread_messages(workroom.id)
            senders = [message.sender for message in messages]
            self.assertEqual(senders[0], "workflow")
            self.assertEqual(senders[1:], ["architect", "brainstormer", "architect", "reviewer"])

    async def test_real_dynamic_open_ended_workflow_runs_with_magentic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)

            result = await runtime.run_workflow(
                workflow_id="dynamic-open-ended",
                goal="Create a minimal README.md for this new project with a title and one sentence description.",
                created_at=1,
                parent_thread_id=runtime.main_thread_id,
            )

            readme = project_root / "README.md"
            self.assertIn(result["status"], {"completed", "blocked"})
            if readme.exists():
                readme_text = readme.read_text(encoding="utf-8")
                self.assertIn("#", readme_text)

            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)

            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            senders = [message.sender for message in runtime.list_thread_messages(workroom.id)]
            self.assertEqual(senders[0], "workflow")
            self.assertTrue(any(sender != "workflow" for sender in senders))

    async def test_real_specialist_handoff_workflow_runs_in_a_shared_workroom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)

            result = await runtime.run_workflow(
                workflow_id="specialist-handoff",
                goal=(
                    "Discuss whether a new Python CLI should start with Typer or argparse. "
                    "End with one clear recommendation and two short reasons."
                ),
                created_at=1,
                parent_thread_id=runtime.main_thread_id,
            )

            self.assertIn(result["status"], {"completed", "blocked"})

            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)

            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            messages = runtime.list_thread_messages(workroom.id)
            self.assertGreaterEqual(len(messages), 2)
            senders = [message.sender for message in messages]
            self.assertEqual(senders[0], "workflow")
            self.assertTrue(any(sender != "workflow" for sender in senders))

    async def test_real_orchestrator_can_complete_a_vague_build_flow(self) -> None:
        last_error: AssertionError | None = None
        for _attempt in range(2):
            try:
                await self._assert_vague_build_flow()
                return
            except AssertionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        self.fail("vague build flow did not run")

    async def _assert_vague_build_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            configure_local_runtime(runtime)

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

            entrypoints = calculator_entrypoints(project_root)
            self.assertTrue(entrypoints, "expected a runnable calculator entrypoint to be created")

            successful_result = None
            command_index = 0
            for _path, commands in entrypoints:
                for command, require_output_prefix in verification_commands(project_root, commands):
                    command_index += 1
                    command_result = runtime.run_workspace_command(
                        command,
                        created_at=200 + command_index,
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
                if successful_result is not None:
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
            configure_local_runtime(runtime)

            await runtime.send_user_message_to_orchestrator(
                body="We need a tiny Python CLI calculator. Talk me through the approach first before building anything.",
                created_at=1,
            )

            first_pass_messages = runtime.list_main_messages()
            self.assertGreaterEqual(len(first_pass_messages), 2)
            self.assertEqual(runtime.list_workflow_runs(), [])
            self.assertEqual(calculator_entrypoints(project_root), [])

            await runtime.send_user_message_to_orchestrator(
                body="That sounds fine. Please proceed and build it.",
                created_at=100,
            )

            workflow_runs = runtime.list_workflow_runs()
            self.assertEqual(len(workflow_runs), 1)
            self.assertEqual(workflow_runs[0].state, "completed")

            final_message = runtime.conversation_store.read_message_body(runtime.list_main_messages()[-1]).strip()
            self.assertTrue("workflow" in final_message or "built" in final_message.lower())
            self.assertTrue(bool(calculator_entrypoints(project_root)) or bool(_workspace_python_files(project_root)))
