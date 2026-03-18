from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_framework import AgentSession

from ergon_studio.config import save_global_config


class RuntimeTests(unittest.TestCase):
    def test_load_runtime_combines_paths_registry_and_tools(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            self.assertEqual(runtime.paths.project_root, project_root)
            self.assertIn("orchestrator", runtime.registry.agent_definitions)
            self.assertIn("read_file", runtime.tool_registry)
            self.assertEqual(runtime.main_session_id, "session-main")
            self.assertEqual(runtime.main_thread_id, "thread-main")
            self.assertEqual(runtime.list_tasks(), [])
            self.assertEqual(runtime.list_workflow_runs(), [])
            self.assertEqual([thread.id for thread in runtime.list_threads()], ["thread-main"])
            self.assertEqual(runtime.list_main_messages(), [])
            self.assertEqual(runtime.list_events(), [])
            self.assertEqual(runtime.list_approvals(), [])
            self.assertEqual(runtime.list_memory_facts(), [])
            self.assertEqual(runtime.list_artifacts(), [])
            self.assertIsNotNone(runtime.agent_session_store)

    def test_runtime_can_build_orchestrator_when_provider_is_configured(self) -> None:
        from ergon_studio.runtime import load_runtime

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
            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            agent = runtime.build_agent("orchestrator")

            self.assertEqual(agent.name, "Orchestrator")
            self.assertEqual(agent.client.model_id, "qwen2.5-coder")
            self.assertEqual(runtime.agent_status_summary("orchestrator"), "ready via local (qwen2.5-coder)")

    def test_runtime_can_reload_registry_after_config_changes(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            self.assertEqual(runtime.agent_status_summary("orchestrator"), "not configured")

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

            self.assertEqual(runtime.agent_status_summary("orchestrator"), "ready via local (qwen2.5-coder)")

    def test_runtime_can_save_agent_definition_text_and_reload_registry(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            runtime.save_agent_definition_text(
                agent_id="orchestrator",
                text="""---
id: orchestrator
name: Orchestrator
role: orchestrator
temperature: 0.2
tools:
  - read_file
---
## Identity
Lead engineer for the AI firm.

## Output Style
Be extremely concise.
""",
                created_at=1_710_755_200,
            )

            self.assertEqual(
                runtime.registry.agent_definitions["orchestrator"].sections["Output Style"],
                "Be extremely concise.",
            )
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["definition_saved"],
            )

    def test_runtime_can_save_global_config_text_and_reload_registry(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.save_global_config_text(
                text="""{
  "providers": {
    "local": {
      "type": "openai_chat",
      "base_url": "http://localhost:8080/v1",
      "api_key": "not-needed",
      "model": "qwen2.5-coder"
    }
  },
  "role_assignments": {
    "orchestrator": "local"
  },
  "approvals": {},
  "ui": {}
}
""",
                created_at=1_710_755_200,
            )

            self.assertEqual(runtime.agent_status_summary("orchestrator"), "ready via local (qwen2.5-coder)")
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["config_saved"],
            )

    def test_runtime_can_save_workflow_definition_text_and_reload_registry(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            runtime.save_workflow_definition_text(
                workflow_id="standard-build",
                text="""---
id: standard-build
name: Standard Build
kind: workflow
orchestration: sequential
---
## Purpose
Ship standard implementation work.

## Exit Conditions
Return reviewed code and a clear summary.
""",
                created_at=1_710_755_200,
            )

            self.assertEqual(
                runtime.registry.workflow_definitions["standard-build"].sections["Exit Conditions"],
                "Return reviewed code and a clear summary.",
            )
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["definition_saved"],
            )

    def test_runtime_can_run_and_persist_workspace_commands(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            result = runtime.run_workspace_command(
                "pwd",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )

            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["status"], "completed")
            command_runs = runtime.list_command_runs()
            self.assertEqual(len(command_runs), 1)
            self.assertEqual(command_runs[0].thread_id, runtime.main_thread_id)
            self.assertEqual(command_runs[0].agent_id, "user")
            self.assertIn("# Command Run", runtime.read_command_output(command_runs[0].id))
            self.assertIn("command_run", [event.kind for event in runtime.list_events()])

    def test_runtime_can_append_and_read_main_thread_messages(self) -> None:
        from ergon_studio.runtime import load_runtime

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
                body="Hello from runtime.",
                created_at=1_710_755_200,
            )

            messages = runtime.list_main_messages()

            self.assertEqual([message.id for message in messages], ["message-1"])
            self.assertEqual(
                runtime.conversation_store.read_message_body(messages[0]),
                "Hello from runtime.\n",
            )
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["message_created"],
            )

    def test_runtime_can_create_and_list_tasks(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_task(
                task_id="task-1",
                title="Build real task panel",
                state="in_progress",
                created_at=1_710_755_200,
            )

            tasks = runtime.list_tasks()

            self.assertEqual([task.id for task in tasks], ["task-1"])
            self.assertEqual(tasks[0].title, "Build real task panel")
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["task_created"],
            )

    def test_runtime_can_create_and_list_additional_threads(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
                summary="Review thread",
            )

            threads = runtime.list_threads()

            self.assertEqual(
                [thread.id for thread in threads],
                ["thread-main", "thread-review-1"],
            )
            self.assertIsNone(threads[0].assigned_agent_id)
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["thread_created"],
            )

    def test_runtime_can_create_agent_thread(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            thread = runtime.create_agent_thread(
                agent_id="architect",
                created_at=1_710_755_200,
                parent_task_id="task-1",
            )

            self.assertEqual(thread.kind, "agent_direct")
            self.assertEqual(thread.assigned_agent_id, "architect")
            self.assertEqual(thread.parent_task_id, "task-1")
            self.assertEqual(runtime.get_thread(thread.id), thread)

    def test_runtime_can_start_workflow_run(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_200,
            )

            self.assertEqual(workflow_run.workflow_id, "standard-build")
            self.assertEqual(workflow_run.state, "running")
            self.assertEqual(workflow_run.current_step_index, 0)
            self.assertEqual(len(threads), 3)
            self.assertEqual([thread.assigned_agent_id for thread in threads], ["architect", "coder", "reviewer"])
            self.assertEqual(
                [run.id for run in runtime.list_workflow_runs()],
                [workflow_run.id],
            )
            self.assertIn("workflow_started", [event.kind for event in runtime.list_events()])

    def test_runtime_can_update_task_state(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            task = runtime.create_task(
                task_id="task-1",
                title="Track progress",
                state="planned",
                created_at=1_710_755_200,
            )

            updated = runtime.update_task_state(
                task_id=task.id,
                state="completed",
                updated_at=1_710_755_210,
            )

            self.assertEqual(updated.state, "completed")
            self.assertEqual(runtime.get_task(task.id), updated)
            self.assertIn("task_updated", [event.kind for event in runtime.list_events()])

    def test_runtime_can_read_latest_main_user_message_body(self) -> None:
        from ergon_studio.runtime import load_runtime

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
                body="First goal.",
                created_at=1_710_755_200,
            )
            runtime.append_message_to_main_thread(
                message_id="message-2",
                sender="orchestrator",
                kind="chat",
                body="Acknowledged.",
                created_at=1_710_755_201,
            )
            runtime.append_message_to_main_thread(
                message_id="message-3",
                sender="user",
                kind="chat",
                body="Latest goal.",
                created_at=1_710_755_202,
            )

            self.assertEqual(runtime.latest_main_user_message_body(), "Latest goal.")

    def test_runtime_can_append_messages_to_additional_threads(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_200,
            )
            runtime.append_message_to_thread(
                thread_id="thread-review-1",
                message_id="message-1",
                sender="reviewer",
                kind="review",
                body="Looks good overall.",
                created_at=1_710_755_201,
            )

            messages = runtime.list_thread_messages("thread-review-1")

            self.assertEqual([message.id for message in messages], ["message-1"])
            self.assertEqual(
                runtime.conversation_store.read_message_body(messages[0]),
                "Looks good overall.\n",
            )

    def test_runtime_can_request_and_list_approvals(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )

            approvals = runtime.list_approvals()

            self.assertEqual([approval.id for approval in approvals], ["approval-1"])
            self.assertEqual(approvals[0].status, "pending")
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["approval_requested"],
            )

    def test_runtime_can_approve_and_reject_approvals(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update README",
                created_at=1_710_755_200,
            )
            runtime.request_approval(
                approval_id="approval-2",
                requester="orchestrator",
                action="run_command",
                risk_class="high",
                reason="Install dependencies",
                created_at=1_710_755_201,
            )

            approved = runtime.resolve_approval(
                approval_id="approval-1",
                status="approved",
                created_at=1_710_755_202,
            )
            rejected = runtime.resolve_approval(
                approval_id="approval-2",
                status="rejected",
                created_at=1_710_755_203,
            )

            self.assertEqual(approved.status, "approved")
            self.assertEqual(rejected.status, "rejected")
            self.assertEqual(
                [approval.status for approval in runtime.list_approvals()],
                ["approved", "rejected"],
            )
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["approval_requested", "approval_requested", "approval_approved", "approval_rejected"],
            )

    def test_runtime_can_list_approvals_for_workflow_run(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_200,
            )
            runtime.request_approval(
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update workflow file",
                created_at=1_710_755_210,
                task_id=threads[0].parent_task_id,
                thread_id=threads[0].id,
            )
            runtime.request_approval(
                approval_id="approval-2",
                requester="orchestrator",
                action="run_command",
                risk_class="high",
                reason="Install dependencies",
                created_at=1_710_755_211,
            )

            approvals = runtime.list_pending_approvals_for_workflow_run(workflow_run.id)

            self.assertEqual([approval.id for approval in approvals], ["approval-1"])

    def test_runtime_can_add_and_list_memory_facts(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.add_memory_fact(
                fact_id="fact-1",
                scope="project",
                kind="decision",
                content="Use Textual for the TUI.",
                created_at=1_710_755_200,
            )

            facts = runtime.list_memory_facts()

            self.assertEqual([fact.id for fact in facts], ["fact-1"])
            self.assertEqual(facts[0].content, "Use Textual for the TUI.")
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["memory_fact_added"],
            )

    def test_runtime_can_create_and_list_artifacts(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            runtime.create_artifact(
                artifact_id="artifact-1",
                kind="design-note",
                title="Architecture Notes",
                content="Use Textual with a runtime-first architecture.",
                created_at=1_710_755_200,
            )

            artifacts = runtime.list_artifacts()

            self.assertEqual([artifact.id for artifact in artifacts], ["artifact-1"])
            self.assertEqual(artifacts[0].title, "Architecture Notes")
            self.assertEqual(
                runtime.read_artifact_body("artifact-1"),
                "Use Textual with a runtime-first architecture.\n",
            )


class RuntimeAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_can_send_user_message_and_persist_orchestrator_session(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text
                self.created_session_ids: list[str] = []
                self.seen_session_ids: list[str] = []

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                self.created_session_ids.append(session_id or "")
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                self.seen_session_ids.append(session.session_id if session is not None else "")
                return SimpleNamespace(text=self.response_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            first_agent = FakeAgent("I can take this on.")
            with patch.object(type(runtime), "build_agent", return_value=first_agent):
                await runtime.send_user_message_to_orchestrator(
                    body="Build the next slice.",
                    created_at=1_710_755_200,
                )

            session_path = runtime.paths.sessions_dir / "thread-main" / "orchestrator.json"
            self.assertTrue(session_path.exists())
            self.assertEqual(first_agent.created_session_ids, ["thread-main:orchestrator"])
            self.assertEqual(first_agent.seen_session_ids, ["thread-main:orchestrator"])
            self.assertEqual(
                [message.sender for message in runtime.list_main_messages()],
                ["user", "orchestrator"],
            )

            reloaded_runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            second_agent = FakeAgent("Continuing from the same thread.")
            with patch.object(type(reloaded_runtime), "build_agent", return_value=second_agent):
                await reloaded_runtime.send_user_message_to_orchestrator(
                    body="Keep going.",
                    created_at=1_710_755_210,
                )

            self.assertEqual(second_agent.created_session_ids, [])
            self.assertEqual(second_agent.seen_session_ids, ["thread-main:orchestrator"])
            self.assertEqual(
                [message.sender for message in reloaded_runtime.list_main_messages()],
                ["user", "orchestrator", "user", "orchestrator"],
            )

    async def test_runtime_logs_unavailable_orchestrator_without_crashing(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            user_message, orchestrator_message = await runtime.send_user_message_to_orchestrator(
                body="Ship it.",
                created_at=1_710_755_200,
            )

            self.assertEqual(user_message.sender, "user")
            self.assertIsNone(orchestrator_message)
            self.assertEqual(len(runtime.list_main_messages()), 1)
            self.assertEqual(
                [event.kind for event in runtime.list_events()],
                ["message_created", "agent_unavailable"],
            )

    async def test_runtime_can_send_message_to_agent_thread(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text
                self.created_session_ids: list[str] = []
                self.seen_session_ids: list[str] = []

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                self.created_session_ids.append(session_id or "")
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                self.seen_session_ids.append(session.session_id if session is not None else "")
                return SimpleNamespace(text=self.response_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            thread = runtime.create_agent_thread(agent_id="architect", created_at=1_710_755_200)
            agent = FakeAgent("Here is the architecture sketch.")

            with patch.object(type(runtime), "build_agent", return_value=agent):
                await runtime.send_message_to_agent_thread(
                    thread_id=thread.id,
                    body="Design the next component.",
                    created_at=1_710_755_201,
                )

            self.assertEqual(agent.created_session_ids, [f"{thread.id}:architect"])
            self.assertEqual(agent.seen_session_ids, [f"{thread.id}:architect"])
            self.assertEqual(
                [message.sender for message in runtime.list_thread_messages(thread.id)],
                ["orchestrator", "architect"],
            )

    async def test_runtime_blocks_workflow_when_agent_is_unavailable(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_200,
            )

            blocked_run, blocked_thread, reply = await runtime.advance_workflow_run(
                workflow_run_id=workflow_run.id,
                created_at=1_710_755_210,
            )

            self.assertIsNone(reply)
            self.assertEqual(blocked_thread.id, threads[0].id)
            self.assertEqual(blocked_run.state, "blocked")
            self.assertEqual(runtime.get_task(blocked_thread.parent_task_id).state, "blocked")
            self.assertIn("workflow_blocked", [event.kind for event in runtime.list_events()])

    async def test_runtime_can_advance_workflow_run_to_next_agent(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_210,
            )
            fake_agents = {
                "architect": FakeAgent("Architecture ready."),
                "coder": FakeAgent("Implementation ready."),
                "reviewer": FakeAgent("Review complete."),
            }

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ):
                advanced_run, first_thread, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )
                second_run, second_thread, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_230,
                )

            self.assertEqual(first_thread.id, threads[0].id)
            self.assertEqual(advanced_run.current_step_index, 1)
            self.assertEqual(runtime.get_task(first_thread.parent_task_id).state, "completed")
            self.assertEqual(second_thread.id, threads[1].id)
            self.assertEqual(second_run.current_step_index, 2)
            second_thread_messages = runtime.list_thread_messages(threads[1].id)
            self.assertIn("Architecture ready.", runtime.conversation_store.read_message_body(second_thread_messages[0]))
            self.assertEqual(
                [event.kind for event in runtime.list_events() if event.kind.startswith("workflow_")],
                ["workflow_started", "workflow_advanced", "workflow_advanced"],
            )

    async def test_runtime_completes_root_task_when_workflow_finishes(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            workflow_run, _ = runtime.start_workflow_run(
                workflow_id="single-agent-execution",
                created_at=1_710_755_210,
            )

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(f"{agent_id} done."),
            ):
                completed_run, _, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )

            self.assertEqual(completed_run.state, "completed")
            self.assertEqual(runtime.get_task(completed_run.root_task_id).state, "completed")
            self.assertIn(
                "Workflow complete: single-agent-execution",
                runtime.conversation_store.read_message_body(runtime.list_main_messages()[-1]),
            )
            artifacts = runtime.list_artifacts()
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].kind, "workflow-report")
            self.assertEqual(artifacts[0].task_id, completed_run.root_task_id)
            self.assertIn(
                "coder done.",
                artifacts[0].file_path.read_text(encoding="utf-8"),
            )

    async def test_runtime_can_list_artifacts_for_workflow_run(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            workflow_run, _ = runtime.start_workflow_run(
                workflow_id="single-agent-execution",
                created_at=1_710_755_210,
            )

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(f"{agent_id} done."),
            ):
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )

            runtime.create_artifact(
                artifact_id="artifact-unrelated",
                kind="design-note",
                title="Unrelated",
                content="Not part of the workflow run.",
                created_at=1_710_755_230,
            )

            related_artifacts = runtime.list_artifacts_for_workflow_run(workflow_run.id)

            self.assertEqual(len(related_artifacts), 1)
            self.assertEqual(related_artifacts[0].kind, "workflow-report")
            self.assertEqual(related_artifacts[0].task_id, workflow_run.root_task_id)

    async def test_runtime_can_list_events_for_workflow_run(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Ship the feature.",
                created_at=1_710_755_200,
            )
            workflow_run, _ = runtime.start_workflow_run(
                workflow_id="single-agent-execution",
                created_at=1_710_755_210,
            )
            runtime.append_event(
                kind="unrelated_event",
                summary="Unrelated activity",
                created_at=1_710_755_211,
            )

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(f"{agent_id} done."),
            ):
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )

            events = runtime.list_events_for_workflow_run(workflow_run.id)

            self.assertIn("workflow_started", [event.kind for event in events])
            self.assertIn("workflow_advanced", [event.kind for event in events])
            self.assertIn("workflow_completed", [event.kind for event in events])
            self.assertNotIn("unrelated_event", [event.kind for event in events])

    def test_runtime_can_list_threads_for_workflow_run(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_200,
            )
            runtime.create_thread(
                thread_id="thread-review-1",
                kind="review",
                created_at=1_710_755_210,
                summary="Unrelated review",
            )

            related_threads = runtime.list_threads_for_workflow_run(workflow_run.id)

            self.assertEqual([thread.id for thread in related_threads], [thread.id for thread in threads])

    async def test_runtime_can_request_workflow_fix_cycle(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            workflow_run, _ = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_210,
            )
            fake_agents = {
                "architect": FakeAgent("Architecture ready."),
                "coder": FakeAgent("Implementation ready."),
                "reviewer": FakeAgent("Needs fixes."),
                "fixer": FakeAgent("Fixes applied."),
            }

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ):
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_230,
                )
                completed_run, _, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_240,
                )
                repaired_run, repair_threads = runtime.request_workflow_fix_cycle(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_250,
                )

            self.assertEqual(completed_run.state, "completed")
            self.assertEqual(repaired_run.state, "repairing")
            self.assertEqual([thread.assigned_agent_id for thread in repair_threads], ["fixer", "reviewer"])
            self.assertEqual(repaired_run.current_step_index, 3)
            self.assertEqual(runtime.get_task(repaired_run.root_task_id).state, "in_progress")
            self.assertIn("workflow_fix_cycle_requested", [event.kind for event in runtime.list_events()])

    def test_runtime_can_describe_workflow_run_tree(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_200,
            )

            tree = runtime.describe_workflow_run(workflow_run.id)

            self.assertIsNotNone(tree)
            self.assertEqual(tree.workflow_run.id, workflow_run.id)
            self.assertEqual(tree.root_task.id, workflow_run.root_task_id)
            self.assertEqual(
                [step.task.title for step in tree.steps],
                [
                    "standard-build: architect",
                    "standard-build: coder",
                    "standard-build: reviewer",
                ],
            )
            self.assertEqual(
                [step.threads[0].id for step in tree.steps],
                [thread.id for thread in threads],
            )
            self.assertEqual(
                [step.threads[0].assigned_agent_id for step in tree.steps],
                ["architect", "coder", "reviewer"],
            )

    async def test_runtime_prefers_last_thread_for_workflow_run_selection(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

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
                body="Build the next feature.",
                created_at=1_710_755_200,
            )
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_210,
            )
            self.assertEqual(runtime.preferred_thread_id_for_workflow_run(workflow_run.id), threads[0].id)

            fake_agents = {
                "architect": FakeAgent("Architecture ready."),
                "coder": FakeAgent("Implementation ready."),
                "reviewer": FakeAgent("Review complete."),
            }
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ):
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_220,
                )
                updated_run, _, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_230,
                )

            self.assertEqual(updated_run.last_thread_id, threads[1].id)
            self.assertEqual(runtime.preferred_thread_id_for_workflow_run(workflow_run.id), threads[1].id)
