from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_framework import AgentSession, Message

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
            self.assertTrue(runtime.main_session_id.startswith("session-"))
            self.assertTrue(runtime.main_thread_id.startswith("thread-main-session-"))
            self.assertEqual(runtime.list_tasks(), [])
            self.assertEqual(runtime.list_workflow_runs(), [])
            self.assertEqual([thread.id for thread in runtime.list_threads()], [runtime.main_thread_id])
            self.assertEqual(runtime.list_main_messages(), [])
            self.assertEqual(runtime.list_events(), [])
            self.assertEqual(runtime.list_approvals(), [])
            self.assertEqual(runtime.list_memory_facts(), [])
            self.assertEqual(runtime.list_artifacts(), [])
            self.assertEqual(runtime.list_tool_calls(), [])
            self.assertIsNotNone(runtime.agent_session_store)

    def test_load_runtime_can_create_and_attach_to_multiple_sessions(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            second = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Parallel lane",
            )

            self.assertTrue(first.main_session_id.startswith("session-"))
            self.assertTrue(first.main_thread_id.startswith("thread-main-session-"))
            self.assertNotEqual(second.main_session_id, first.main_session_id)
            self.assertTrue(second.main_thread_id.startswith("thread-main-session-"))
            self.assertEqual(second.current_session().title, "Parallel lane")
            self.assertEqual(
                [session.id for session in second.list_sessions()],
                [second.main_session_id, first.main_session_id],
            )

    def test_runtime_isolates_messages_by_active_session(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            first.append_message_to_main_thread(
                message_id="message-main-1",
                sender="user",
                kind="chat",
                body="first session",
                created_at=1,
            )

            second = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Parallel lane",
            )
            second.append_message_to_main_thread(
                message_id="message-main-2",
                sender="user",
                kind="chat",
                body="second session",
                created_at=2,
            )

            reloaded_first = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                session_id=first.main_session_id,
            )

            self.assertEqual(
                [reloaded_first.conversation_store.read_message_body(message).strip() for message in reloaded_first.list_main_messages()],
                ["first session"],
            )
            self.assertEqual(
                [second.conversation_store.read_message_body(message).strip() for message in second.list_main_messages()],
                ["second session"],
            )

    def test_load_runtime_defaults_to_latest_session(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            second = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Latest lane",
            )

            resumed = load_runtime(project_root=project_root, home_dir=home_dir)

            self.assertEqual(resumed.main_session_id, second.main_session_id)
            self.assertEqual(resumed.current_session().title, "Latest lane")

    def test_load_runtime_can_reopen_specific_session_by_id(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            second = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Parallel lane",
            )

            reopened = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                session_id=first.main_session_id,
            )

            self.assertEqual(reopened.main_session_id, first.main_session_id)
            self.assertNotEqual(reopened.main_session_id, second.main_session_id)
            self.assertEqual(reopened.current_session().title, first.current_session().title)

    def test_load_runtime_creates_fresh_session_when_only_archived_sessions_exist(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            first = load_runtime(project_root=project_root, home_dir=home_dir)
            first.archive_session(
                session_id=first.main_session_id,
                created_at=10,
            )

            resumed = load_runtime(project_root=project_root, home_dir=home_dir)

            self.assertNotEqual(resumed.main_session_id, first.main_session_id)
            self.assertIsNone(resumed.current_session().archived_at)
            self.assertTrue(resumed.main_session_id.startswith("session-"))

    def test_appending_messages_updates_session_timestamp(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            created = runtime.current_session().created_at

            runtime.append_message_to_main_thread(
                message_id="message-1",
                sender="user",
                kind="chat",
                body="hello",
                created_at=created + 25,
            )

            self.assertEqual(runtime.current_session().updated_at, created + 25)
            resumed = load_runtime(project_root=project_root, home_dir=home_dir)
            self.assertEqual(resumed.main_session_id, runtime.main_session_id)

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
                            "capabilities": {
                                "tool_calling": True,
                                "structured_output": True,
                            },
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
            self.assertEqual(
                runtime.provider_capabilities("local"),
                {"tool_calling": True, "structured_output": True},
            )

    def test_runtime_builds_agents_with_retrieval_context_provider(self) -> None:
        from ergon_studio.context_providers import RetrievalContextProvider
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "index.php").write_text(
                "<?php\necho 'hello from php';\n",
                encoding="utf-8",
            )

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

            agent = runtime.build_agent("orchestrator")

            self.assertTrue(
                any(isinstance(provider, RetrievalContextProvider) for provider in agent.context_providers)
            )

    def test_starting_debate_workflow_creates_group_workroom(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="debate",
                created_at=1_710_755_200,
            )

            self.assertEqual(runtime.workflow_orchestration("debate"), "group_chat")
            self.assertEqual(len(threads), 1)
            self.assertEqual(threads[0].kind, "group_workroom")
            self.assertIn("architect + brainstormer + reviewer", threads[0].summary or "")
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)
            self.assertEqual(run_view.steps[0].threads[0].kind, "group_workroom")

    def test_starting_dynamic_open_ended_workflow_creates_group_workroom(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="dynamic-open-ended",
                created_at=1_710_755_200,
            )

            self.assertEqual(runtime.workflow_orchestration("dynamic-open-ended"), "magentic")
            self.assertEqual(len(threads), 1)
            self.assertEqual(threads[0].kind, "group_workroom")
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)
            self.assertEqual(run_view.steps[0].threads[0].kind, "group_workroom")

    def test_starting_specialist_handoff_workflow_creates_group_workroom(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="specialist-handoff",
                created_at=1_710_755_200,
            )

            self.assertEqual(runtime.workflow_orchestration("specialist-handoff"), "handoff")
            self.assertEqual(len(threads), 1)
            self.assertEqual(threads[0].kind, "group_workroom")
            run_view = runtime.describe_workflow_run(workflow_run.id)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)
            self.assertEqual(run_view.steps[0].threads[0].kind, "group_workroom")

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

    def test_runtime_caps_agent_thread_command_budget(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            thread = runtime.create_agent_thread(agent_id="reviewer", created_at=1_710_755_200)

            for index in range(3):
                result = runtime.run_workspace_command(
                    "pwd",
                    created_at=1_710_755_210 + index,
                    thread_id=thread.id,
                    agent_id="reviewer",
                    require_approval=False,
                )
                self.assertEqual(result["status"], "completed")

            capped = runtime.run_workspace_command(
                "pwd",
                created_at=1_710_755_220,
                thread_id=thread.id,
                agent_id="reviewer",
                require_approval=False,
            )

            self.assertEqual(capped["status"], "budget_exhausted")
            self.assertIn("Command budget exhausted", str(capped["stderr"]))
            self.assertIn("command_budget_exhausted", [event.kind for event in runtime.list_events()])
            command_run_count = len(runtime.list_command_runs())
            event_count = len(runtime.list_events())

            repeated = runtime.run_workspace_command(
                "pwd",
                created_at=1_710_755_221,
                thread_id=thread.id,
                agent_id="reviewer",
                require_approval=False,
            )

            self.assertEqual(repeated["status"], "budget_exhausted")
            self.assertEqual(len(runtime.list_command_runs()), command_run_count)
            self.assertEqual(len(runtime.list_events()), event_count)
            self.assertEqual(repeated["command_run_id"], capped["command_run_id"])

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

    def test_runtime_creates_and_saves_task_whiteboards(self) -> None:
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
                title="Build memory system",
                state="planned",
                created_at=10,
            )
            whiteboard = runtime.get_task_whiteboard("task-1")

            self.assertIsNotNone(whiteboard)
            self.assertEqual(whiteboard.sections["Goal"], "Build memory system")

            runtime.save_task_whiteboard_text(
                task_id="task-1",
                text="""---
task_id: task-1
title: Build memory system
updated_at: 20
---
## Goal
Build the framework-native memory system.

## Constraints
Use Agent Framework context providers.

## Plan
Add whiteboards, durable memory, and retrieval providers.

## Decisions
Use markdown whiteboards under ~/.ergon.studio.

## Open Questions

## Acceptance Criteria
Agents receive whiteboard context before each run.
""",
                created_at=20,
            )

            updated = runtime.get_task_whiteboard("task-1")
            self.assertIsNotNone(updated)
            self.assertEqual(updated.sections["Decisions"], "Use markdown whiteboards under ~/.ergon.studio.")
            self.assertIn("whiteboard_saved", [event.kind for event in runtime.list_events()])

    def test_runtime_add_memory_fact_supports_metadata(self) -> None:
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
                content="Use task whiteboards.",
                created_at=10,
                source="task-1",
                confidence=0.8,
                tags=("memory", "design"),
            )

            fact = runtime.list_memory_facts()[0]
            self.assertEqual(fact.source, "task-1")
            self.assertEqual(fact.tags, ("memory", "design"))

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
                [runtime.main_thread_id, "thread-review-1"],
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
            self.assertEqual(len(threads), 4)
            self.assertEqual([thread.assigned_agent_id for thread in threads], ["architect", "coder", "tester", "reviewer"])
            self.assertEqual(
                [run.id for run in runtime.list_workflow_runs()],
                [workflow_run.id],
            )
            self.assertIn("workflow_started", [event.kind for event in runtime.list_events()])
            whiteboard = runtime.get_task_whiteboard(workflow_run.root_task_id)
            self.assertIsNotNone(whiteboard)
            assert whiteboard is not None
            self.assertEqual(whiteboard.sections["Goal"], "Workflow: standard-build")
            self.assertEqual(
                whiteboard.sections["Plan"],
                "1. architect\n2. coder\n3. tester\n4. reviewer",
            )
            self.assertIn("minimal working result", whiteboard.sections["Acceptance Criteria"])

    def test_runtime_uses_workflow_definition_steps(self) -> None:
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
steps:
  - reviewer
  - fixer
---
## Purpose
Use a review-first loop.

## Exit Conditions
Return reviewed and repaired work.
""",
                created_at=1_710_755_200,
            )

            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="standard-build",
                created_at=1_710_755_201,
            )

            self.assertEqual(workflow_run.workflow_id, "standard-build")
            self.assertEqual([thread.assigned_agent_id for thread in threads], ["reviewer", "fixer"])
            self.assertEqual(
                [task.title for task in runtime.list_tasks() if task.parent_task_id == workflow_run.root_task_id],
                ["standard-build: reviewer", "standard-build: fixer"],
            )

    def test_runtime_can_expand_parallel_workflow_groups(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="best-of-n",
                created_at=1_710_755_200,
            )
            run_view = runtime.describe_workflow_run(workflow_run.id)

            self.assertEqual(len(threads), 4)
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 2)
            self.assertEqual(len(run_view.steps[0].threads), 3)
            self.assertEqual(len(run_view.steps[1].threads), 1)
            self.assertEqual(run_view.steps[0].task.title, "best-of-n: coder x3")

    def test_runtime_advances_parallel_workflow_groups_as_single_step(self) -> None:
        import asyncio

        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        responses = iter(
            [
                "Candidate A",
                "Candidate B",
                "Candidate C",
                "Review result",
            ]
        )

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
                body="Implement the feature.",
                created_at=1_710_755_190,
            )

            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: FakeAgent(next(responses)),
            ):
                workflow_run, _ = runtime.start_workflow_run(
                    workflow_id="best-of-n",
                    created_at=1_710_755_200,
                )

                advanced_run, thread, reply = asyncio.run(
                    runtime.advance_workflow_run(
                        workflow_run_id=workflow_run.id,
                        created_at=1_710_755_201,
                    )
                )

                self.assertEqual(advanced_run.current_step_index, 1)
                self.assertIsNotNone(thread)
                self.assertIsNotNone(reply)
                first_step_outputs = [
                    runtime.conversation_store.read_message_body(message)
                    for candidate_thread in runtime.describe_workflow_run(workflow_run.id).steps[0].threads
                    for message in runtime.list_thread_messages(candidate_thread.id)
                    if message.sender == "coder"
                ]
                self.assertEqual(len(first_step_outputs), 3)

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
            self.assertIsNone(runtime.read_approval_payload("approval-1"))

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

    def test_runtime_can_defer_and_execute_command_after_approval(self) -> None:
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
                    "providers": {},
                    "role_assignments": {},
                    "approvals": {"run_command": "ask"},
                    "ui": {},
                },
            )
            runtime.reload_registry()

            result = runtime.run_workspace_command(
                "pwd",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )

            self.assertEqual(result["status"], "awaiting_approval")
            self.assertEqual(runtime.list_command_runs(), [])
            approvals = runtime.list_approvals()
            self.assertEqual(len(approvals), 1)
            self.assertEqual(
                runtime.read_approval_payload(approvals[0].id),
                {
                    "agent_id": "user",
                    "command": "pwd",
                    "cwd": str(runtime.paths.project_root.resolve()),
                    "task_id": None,
                    "thread_id": runtime.main_thread_id,
                    "timeout": 60,
                },
            )

            runtime.resolve_approval(
                approval_id=approvals[0].id,
                status="approved",
                created_at=1_710_755_201,
            )

            self.assertEqual(len(runtime.list_command_runs()), 1)
            self.assertEqual(runtime.list_approvals()[0].status, "approved")
            self.assertIn("command_run", [event.kind for event in runtime.list_events()])

    def test_runtime_can_defer_and_execute_file_write_after_approval(self) -> None:
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
                    "providers": {},
                    "role_assignments": {},
                    "approvals": {"write_file": "ask"},
                    "ui": {},
                },
            )
            runtime.reload_registry()

            result = runtime.write_workspace_file(
                "notes.txt",
                "hello\n",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )

            self.assertEqual(result["status"], "awaiting_approval")
            self.assertFalse((project_root / "notes.txt").exists())

            approval = runtime.list_approvals()[0]
            self.assertEqual(
                runtime.read_approval_payload(approval.id),
                {
                    "agent_id": "user",
                    "content": "hello\n",
                    "path": "notes.txt",
                    "task_id": None,
                    "thread_id": runtime.main_thread_id,
                },
            )

            runtime.resolve_approval(
                approval_id=approval.id,
                status="approved",
                created_at=1_710_755_201,
            )

            self.assertEqual((project_root / "notes.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertIn("file_written", [event.kind for event in runtime.list_events()])

    def test_runtime_can_defer_and_execute_file_patch_after_approval(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            (project_root / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            save_global_config(
                runtime.paths.config_path,
                {
                    "providers": {},
                    "role_assignments": {},
                    "approvals": {"patch_file": "ask"},
                    "ui": {},
                },
            )
            runtime.reload_registry()

            result = runtime.patch_workspace_file(
                "notes.txt",
                "world",
                "team",
                created_at=1_710_755_200,
                thread_id=runtime.main_thread_id,
                agent_id="user",
            )

            self.assertEqual(result["status"], "awaiting_approval")
            self.assertEqual((project_root / "notes.txt").read_text(encoding="utf-8"), "hello\nworld\n")

            approval = runtime.list_approvals()[0]
            runtime.resolve_approval(
                approval_id=approval.id,
                status="approved",
                created_at=1_710_755_201,
            )

            self.assertEqual((project_root / "notes.txt").read_text(encoding="utf-8"), "hello\nteam\n")
            self.assertIn("file_patched", [event.kind for event in runtime.list_events()])

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

            session_path = (
                runtime.paths.session_agent_sessions_dir(runtime.main_session_id)
                / runtime.main_thread_id
                / "orchestrator.json"
            )
            self.assertTrue(session_path.exists())
            self.assertEqual(first_agent.created_session_ids, [f"{runtime.main_thread_id}:orchestrator"])
            self.assertEqual(first_agent.seen_session_ids, [f"{runtime.main_thread_id}:orchestrator"])
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
            self.assertEqual(second_agent.seen_session_ids, [f"{runtime.main_thread_id}:orchestrator"])
            self.assertEqual(
                [message.sender for message in reloaded_runtime.list_main_messages()],
                ["user", "orchestrator", "user", "orchestrator"],
            )

    async def test_runtime_auto_titles_default_session_from_first_user_turn(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text="I can take it from here.")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            original_title = runtime.current_session().title

            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                await runtime.send_user_message_to_orchestrator(
                    body="Build a calculator CLI with tests",
                    created_at=1_710_755_200,
                )

            self.assertNotEqual(runtime.current_session().title, original_title)
            self.assertEqual(runtime.current_session().title, "Build a calculator CLI with tests")
            self.assertIn("session_titled", [event.kind for event in runtime.list_events()])

    async def test_runtime_preserves_explicit_session_title_from_first_user_turn(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text="Working on it.")

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(
                project_root=project_root,
                home_dir=home_dir,
                create_session=True,
                session_title="Focus lane",
            )

            with patch.object(type(runtime), "build_agent", return_value=FakeAgent()):
                await runtime.send_user_message_to_orchestrator(
                    body="Build a calculator CLI with tests",
                    created_at=1_710_755_200,
                )

            self.assertEqual(runtime.current_session().title, "Focus lane")
            self.assertNotIn("session_titled", [event.kind for event in runtime.list_events()])

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
                ["message_created", "session_titled", "orchestrator_turn_planned", "agent_unavailable"],
            )

    async def test_runtime_rejects_non_delivery_workflow_for_implementation_turns(self) -> None:
        from ergon_studio.runtime import OrchestratorTurnDecision, load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            async def decide(_runtime, *, body: str, created_at: int):
                return OrchestratorTurnDecision(
                    mode="workflow",
                    reply="",
                    workflow_id="architecture-first",
                    goal=body,
                    deliverable_expected=False,
                )

            async def classify(_runtime, *, body: str, created_at: int) -> bool:
                return False

            async def guard(_runtime, *, body: str, workflow_id: str, created_at: int) -> bool:
                return False

            async def choose_workflow(
                _runtime,
                *,
                body: str,
                goal: str,
                current_workflow_id: str | None,
                created_at: int,
            ) -> str:
                return "dynamic-open-ended"

            async def run_workflow(
                _runtime,
                *,
                workflow_id: str,
                goal: str,
                created_at: int | None = None,
                parent_thread_id: str | None = None,
            ):
                return {
                    "status": "completed",
                    "workflow_run_id": "workflow-run-1",
                    "review_summary": "ACCEPTED: implemented",
                    "last_thread_id": "thread-1",
                }

            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "_classify_deliverable_intent", side_effect=classify, autospec=True),
                patch.object(type(runtime), "_allow_non_delivery_workflow", side_effect=guard, autospec=True),
                patch.object(type(runtime), "_select_delivery_workflow", side_effect=choose_workflow, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                _, reply = await runtime.send_user_message_to_orchestrator(
                    body="Build the feature from scratch. First decide the approach, then implement it here.",
                    created_at=10,
                )

            self.assertIsNotNone(reply)
            assert reply is not None
            self.assertIn("dynamic-open-ended", runtime.conversation_store.read_message_body(reply))
            self.assertIn(
                "orchestrator_delivery_workflow_selected",
                [event.kind for event in runtime.list_events()],
            )

    async def test_runtime_reselects_greenfield_single_agent_delivery_with_selector(self) -> None:
        from ergon_studio.runtime import OrchestratorTurnDecision, load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            async def decide(_runtime, *, body: str, created_at: int):
                return OrchestratorTurnDecision(
                    mode="workflow",
                    reply="",
                    workflow_id="single-agent-execution",
                    goal=body,
                    deliverable_expected=True,
                )

            async def choose_workflow(
                _runtime,
                *,
                body: str,
                goal: str,
                current_workflow_id: str | None,
                created_at: int,
            ) -> str:
                return "dynamic-open-ended"

            async def run_workflow(
                _runtime,
                *,
                workflow_id: str,
                goal: str,
                created_at: int | None = None,
                parent_thread_id: str | None = None,
            ):
                return {
                    "status": "completed",
                    "workflow_run_id": "workflow-run-1",
                    "review_summary": "ACCEPTED: implemented",
                    "last_thread_id": "thread-1",
                }

            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "_select_delivery_workflow", side_effect=choose_workflow, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                _, reply = await runtime.send_user_message_to_orchestrator(
                    body="Build a tiny new CLI app from scratch in this repo.",
                    created_at=10,
                )

            self.assertIsNotNone(reply)
            assert reply is not None
            self.assertIn("dynamic-open-ended", runtime.conversation_store.read_message_body(reply))
            self.assertIn(
                "orchestrator_delivery_workflow_selected",
                [event.kind for event in runtime.list_events()],
            )

    async def test_runtime_reselects_non_delivery_workflow_by_metadata(self) -> None:
        from ergon_studio.runtime import OrchestratorTurnDecision, load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            async def decide(_runtime, *, body: str, created_at: int):
                return OrchestratorTurnDecision(
                    mode="workflow",
                    reply="",
                    workflow_id="specialist-handoff",
                    goal=body,
                    deliverable_expected=True,
                )

            async def choose_workflow(
                _runtime,
                *,
                body: str,
                goal: str,
                current_workflow_id: str | None,
                created_at: int,
            ) -> str:
                self.assertEqual(current_workflow_id, "specialist-handoff")
                return "standard-build"

            async def run_workflow(
                _runtime,
                *,
                workflow_id: str,
                goal: str,
                created_at: int | None = None,
                parent_thread_id: str | None = None,
            ):
                return {
                    "status": "completed",
                    "workflow_run_id": "workflow-run-1",
                    "review_summary": "ACCEPTED: implemented",
                    "last_thread_id": "thread-1",
                }

            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "_select_delivery_workflow", side_effect=choose_workflow, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                _, reply = await runtime.send_user_message_to_orchestrator(
                    body="Build the feature now after the discussion.",
                    created_at=10,
                )

            self.assertIsNotNone(reply)
            assert reply is not None
            self.assertIn("standard-build", runtime.conversation_store.read_message_body(reply))
            self.assertIn(
                "orchestrator_delivery_workflow_selected",
                [event.kind for event in runtime.list_events()],
            )

    async def test_runtime_keeps_full_delivery_goal_for_workflow_runs(self) -> None:
        from ergon_studio.runtime import OrchestratorTurnDecision, load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            captured: dict[str, str] = {}

            async def decide(_runtime, *, body: str, created_at: int):
                return OrchestratorTurnDecision(
                    mode="workflow",
                    reply="",
                    workflow_id="standard-build",
                    goal="Design the architecture only before any implementation happens.",
                    deliverable_expected=True,
                )

            async def run_workflow(
                _runtime,
                *,
                workflow_id: str,
                goal: str,
                created_at: int | None = None,
                parent_thread_id: str | None = None,
            ):
                captured["workflow_id"] = workflow_id
                captured["goal"] = goal
                return {
                    "status": "completed",
                    "workflow_run_id": "workflow-run-1",
                    "review_summary": "ACCEPTED: implemented",
                    "last_thread_id": "thread-1",
                }

            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                await runtime.send_user_message_to_orchestrator(
                    body="Build a tiny calculator CLI from scratch and implement it in this repo.",
                    created_at=10,
                )

            self.assertEqual(captured["workflow_id"], "standard-build")
            self.assertIn("Build a tiny calculator CLI from scratch", captured["goal"])
            self.assertIn("implement it in this repo", captured["goal"])
            self.assertNotIn("Design the architecture only", captured["goal"])

    async def test_runtime_keeps_full_delivery_goal_when_reselecting_delivery_workflow(self) -> None:
        from ergon_studio.runtime import OrchestratorTurnDecision, load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            captured: dict[str, str] = {}

            async def decide(_runtime, *, body: str, created_at: int):
                return OrchestratorTurnDecision(
                    mode="workflow",
                    reply="",
                    workflow_id="architecture-first",
                    goal="Design the architecture only before any implementation happens.",
                    deliverable_expected=False,
                )

            async def classify(_runtime, *, body: str, created_at: int) -> bool:
                return False

            async def guard(_runtime, *, body: str, workflow_id: str, created_at: int) -> bool:
                return False

            async def choose_workflow(
                _runtime,
                *,
                body: str,
                goal: str,
                current_workflow_id: str | None,
                created_at: int,
            ) -> str:
                captured["selected_goal"] = goal
                return "dynamic-open-ended"

            async def run_workflow(
                _runtime,
                *,
                workflow_id: str,
                goal: str,
                created_at: int | None = None,
                parent_thread_id: str | None = None,
            ):
                captured["run_goal"] = goal
                return {
                    "status": "completed",
                    "workflow_run_id": "workflow-run-1",
                    "review_summary": "ACCEPTED: implemented",
                    "last_thread_id": "thread-1",
                }

            with (
                patch.object(type(runtime), "_decide_orchestrator_turn", side_effect=decide, autospec=True),
                patch.object(type(runtime), "_classify_deliverable_intent", side_effect=classify, autospec=True),
                patch.object(type(runtime), "_allow_non_delivery_workflow", side_effect=guard, autospec=True),
                patch.object(type(runtime), "_select_delivery_workflow", side_effect=choose_workflow, autospec=True),
                patch.object(type(runtime), "run_workflow", side_effect=run_workflow, autospec=True),
            ):
                await runtime.send_user_message_to_orchestrator(
                    body="Build a tiny calculator CLI from scratch and implement it in this repo.",
                    created_at=10,
                )

            self.assertIn("Build a tiny calculator CLI from scratch", captured["selected_goal"])
            self.assertIn("implement it in this repo", captured["selected_goal"])
            self.assertNotIn("Design the architecture only", captured["selected_goal"])
            self.assertEqual(captured["run_goal"], captured["selected_goal"])

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
                "tester": FakeAgent("Tests passed."),
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
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(
                {artifact.kind for artifact in artifacts},
                {"workflow-report", "workflow-graph"},
            )
            report_artifact = next(artifact for artifact in artifacts if artifact.kind == "workflow-report")
            self.assertEqual(report_artifact.task_id, completed_run.root_task_id)
            self.assertIn(
                "coder done.",
                report_artifact.file_path.read_text(encoding="utf-8"),
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

            self.assertEqual(len(related_artifacts), 2)
            self.assertEqual([artifact.kind for artifact in related_artifacts], ["workflow-report", "workflow-graph"])
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
                "tester": FakeAgent("Tests passed."),
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
                await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_240,
                )
                completed_run, _, _ = await runtime.advance_workflow_run(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_245,
                )
                repaired_run, repair_threads = runtime.request_workflow_fix_cycle(
                    workflow_run_id=workflow_run.id,
                    created_at=1_710_755_250,
                )

            self.assertEqual(completed_run.state, "completed")
            self.assertEqual(repaired_run.state, "repairing")
            self.assertEqual([thread.assigned_agent_id for thread in repair_threads], ["fixer", "reviewer"])
            self.assertEqual(repaired_run.current_step_index, 4)
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
                    "standard-build: tester",
                    "standard-build: reviewer",
                ],
            )
            self.assertEqual(
                [step.threads[0].id for step in tree.steps],
                [thread.id for thread in threads],
            )
            self.assertEqual(
                [step.threads[0].assigned_agent_id for step in tree.steps],
                ["architect", "coder", "tester", "reviewer"],
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

    async def test_runtime_can_delegate_to_agent_without_user_managing_threads(self) -> None:
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
            with patch.object(type(runtime), "build_agent", return_value=FakeAgent("Implementation complete.")):
                result = await runtime.delegate_to_agent(
                    agent_id="coder",
                    request="Implement the feature.",
                    title="Feature delivery",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["agent_id"], "coder")
            tasks = runtime.list_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].state, "completed")
            threads = [thread for thread in runtime.list_threads() if thread.id != runtime.main_thread_id]
            self.assertEqual(len(threads), 1)
            self.assertEqual(threads[0].assigned_agent_id, "coder")
            thread_messages = runtime.list_thread_messages(threads[0].id)
            self.assertEqual([message.sender for message in thread_messages], ["orchestrator", "coder"])
            self.assertIn("delegation_completed", [event.kind for event in runtime.list_events()])

    async def test_runtime_can_run_workflow_end_to_end_with_orchestrator_review(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, response_text: str) -> None:
                self.response_text = response_text

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                return SimpleNamespace(text=self.response_text)

        fake_agents = {
            "architect": FakeAgent("Architecture plan ready."),
            "coder": FakeAgent("Implementation finished."),
            "tester": FakeAgent("Verification passed."),
            "reviewer": FakeAgent("Review passed."),
            "orchestrator": FakeAgent('{"accepted": true, "summary": "This matches the goal and is ready to present."}'),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.state, "completed")
            self.assertEqual(workflow_run.current_step_index, 4)
            self.assertIsInstance(result["last_thread_id"], str)
            self.assertIn("ACCEPTED:", result["review_summary"])

            run_threads = runtime.list_threads_for_workflow_run(workflow_run.id)
            self.assertEqual(
                [thread.assigned_agent_id for thread in run_threads],
                ["architect", "coder", "tester", "reviewer", "orchestrator"],
            )
            artifacts = runtime.list_artifacts_for_workflow_run(workflow_run.id)
            self.assertTrue(any(artifact.kind == "workflow-report" for artifact in artifacts))
            report = next(artifact for artifact in artifacts if artifact.kind == "workflow-report")
            body = runtime.read_artifact_body(report.id)
            self.assertIn("## Orchestrator Review", body)
            self.assertIn("ACCEPTED: This matches the goal", body)

    async def test_runtime_runs_debate_workflow_in_a_shared_workroom(self) -> None:
        import ergon_studio.workflow_runtime as workflow_runtime
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        class DummyCtx:
            def __init__(self) -> None:
                self.output = None

            async def yield_output(self, value) -> None:
                self.output = value

            async def send_message(self, value, target_id=None) -> None:
                del target_id
                self.output = value

        class FakeBuiltGroupChat:
            def __init__(self, participants, *, selection_func=None, max_rounds=None) -> None:
                self.participants = participants
                self.selection_func = selection_func
                self.max_rounds = max_rounds or len(participants)

            async def run(self, goal, include_status_events=True):
                del include_status_events
                participant_map = {participant.id: participant for participant in self.participants}
                opening = workflow_runtime.GroupChatParticipantMessage(
                    messages=[Message(role="user", text=goal, author_name="workflow")]
                )
                for participant in self.participants:
                    await participant.sync_messages(opening, DummyCtx())

                responses = []
                for round_index in range(self.max_rounds):
                    if self.selection_func is None:
                        participant = self.participants[round_index]
                    else:
                        state = SimpleNamespace(
                            current_round=round_index,
                            participants=[participant.id for participant in self.participants],
                            conversation=list(responses),
                        )
                        participant_id = self.selection_func(state)
                        participant = participant_map[participant_id]
                    ctx = DummyCtx()
                    await participant.handle_request(workflow_runtime.GroupChatRequestMessage(), ctx)
                    response = ctx.output.message
                    responses.append(response)
                    broadcast = workflow_runtime.GroupChatParticipantMessage(messages=[response])
                    for peer in self.participants:
                        if peer is participant:
                            continue
                        await peer.sync_messages(broadcast, DummyCtx())
                return SimpleNamespace(get_final_state=lambda: responses)

        class FakeGroupChatBuilder:
            def __init__(self, *, participants, orchestrator_agent=None, selection_func=None, max_rounds=None) -> None:
                del orchestrator_agent
                self.participants = participants
                self.selection_func = selection_func
                self.max_rounds = max_rounds

            def build(self):
                return FakeBuiltGroupChat(
                    self.participants,
                    selection_func=self.selection_func,
                    max_rounds=self.max_rounds,
                )

        fake_agents = {
            "architect": FakeAgent(
                [
                    "Typer is strong for ergonomics and command structure.",
                    "Given the tradeoffs, I still prefer Typer because it keeps the CLI easier to extend cleanly.",
                ]
            ),
            "brainstormer": FakeAgent("Argparse is lighter, but Typer will keep the CLI easier to extend."),
            "reviewer": FakeAgent("Recommendation: choose Typer for speed of iteration and clearer commands."),
            "orchestrator": FakeAgent('{"accepted": true, "summary": "The debate produced a clear direction."}'),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime.GroupChatBuilder",
                FakeGroupChatBuilder,
            ):
                result = await runtime.run_workflow(
                    workflow_id="debate",
                    goal="Debate whether the CLI should use Typer or argparse.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            run_view = runtime.describe_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(run_view)
            assert run_view is not None
            self.assertEqual(len(run_view.steps), 1)
            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            thread_messages = runtime.list_thread_messages(workroom.id)
            senders = [message.sender for message in thread_messages]
            self.assertEqual(senders, ["workflow", "architect", "brainstormer", "architect", "reviewer"])
            transcript = runtime.conversation_store.read_message_body(thread_messages[1])
            self.assertIn("Typer", transcript)

    async def test_runtime_runs_magentic_workflow_in_a_shared_workroom(self) -> None:
        import ergon_studio.workflow_runtime as workflow_runtime
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                del messages, session
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        class FakeWorkflowResult(list):
            def __init__(self, events, outputs) -> None:
                super().__init__(events)
                self._outputs = outputs

            def get_outputs(self):
                return self._outputs

            def get_final_state(self):
                return "completed"

            def get_request_info_events(self):
                return []

        class DummyCtx:
            def __init__(self) -> None:
                self.output = None

            async def yield_output(self, value) -> None:
                self.output = value

            async def send_message(self, value, target_id=None) -> None:
                del target_id
                self.output = value

        class FakeBuiltMagenticWorkflow:
            def __init__(self, participants) -> None:
                self.participants = participants

            async def run(self, goal, include_status_events=True):
                del include_status_events
                participant_map = {participant.id: participant for participant in self.participants}
                opening = workflow_runtime.GroupChatParticipantMessage(
                    messages=[Message(role="user", text=goal, author_name="workflow")]
                )
                for participant in self.participants:
                    await participant.sync_messages(opening, DummyCtx())

                responses = []
                for agent_id in ("architect", "coder", "reviewer"):
                    participant = participant_map[agent_id]
                    ctx = DummyCtx()
                    await participant.handle_request(
                        workflow_runtime.GroupChatRequestMessage(additional_instruction=f"Continue as {agent_id}."),
                        ctx,
                    )
                    response = ctx.output.message
                    responses.append(response)
                    broadcast = workflow_runtime.GroupChatParticipantMessage(messages=[response])
                    for peer in self.participants:
                        if peer.id == agent_id:
                            continue
                        await peer.sync_messages(broadcast, DummyCtx())

                return FakeWorkflowResult(
                    events=[
                        SimpleNamespace(
                            type="magentic_orchestrator",
                            data=SimpleNamespace(
                                event_type="PLAN_CREATED",
                                content=Message(
                                    role="assistant",
                                    text="Plan: architect -> coder -> reviewer",
                                    author_name="magentic_manager",
                                ),
                            ),
                        )
                    ],
                    outputs=[[
                        Message(
                            role="assistant",
                            text="Dynamic workflow complete.",
                            author_name="magentic_manager",
                        )
                    ]],
                )

        class FakeMagenticBuilder:
            def __init__(self, *, participants, manager_agent=None, max_round_count=None, enable_plan_review=False) -> None:
                del manager_agent, max_round_count, enable_plan_review
                self.participants = participants

            def build(self):
                return FakeBuiltMagenticWorkflow(self.participants)

        fake_agents = {
            "architect": FakeAgent("We should break the task into a CLI module and a thin entrypoint."),
            "coder": FakeAgent("I implemented the first concrete slice."),
            "reviewer": FakeAgent("The adaptive run is coherent and ready for review."),
            "orchestrator": FakeAgent('{"accepted": true, "summary": "The adaptive workflow reached a concrete result."}'),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime.MagenticBuilder",
                FakeMagenticBuilder,
            ), patch(
                "ergon_studio.workflow_runtime._build_magentic_manager_agent",
                return_value=SimpleNamespace(),
            ):
                result = await runtime.run_workflow(
                    workflow_id="dynamic-open-ended",
                    goal="Build a small CLI in the repo using adaptive delegation.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            run_view = runtime.describe_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(run_view)
            assert run_view is not None
            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            thread_messages = runtime.list_thread_messages(workroom.id)
            senders = [message.sender for message in thread_messages]
            self.assertEqual(
                senders,
                ["workflow", "architect", "coder", "reviewer", "magentic_manager", "magentic_manager"],
            )

    async def test_runtime_runs_handoff_workflow_in_a_shared_workroom(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                del messages, session
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        class FakeWorkflowResult(list):
            def __init__(self, events, outputs) -> None:
                super().__init__(events)
                self._outputs = outputs

            def get_outputs(self):
                return self._outputs

            def get_final_state(self):
                return "completed"

            def get_request_info_events(self):
                return []

        class FakeBuiltHandoffWorkflow:
            async def run(self, goal, include_status_events=True):
                del goal, include_status_events
                return FakeWorkflowResult(
                    events=[
                        SimpleNamespace(
                            type="handoff_sent",
                            data=SimpleNamespace(source="architect", target="reviewer"),
                        )
                    ],
                    outputs=[[
                        Message(
                            role="assistant",
                            text="Typer keeps the CLI easier to extend.",
                            author_name="architect",
                        ),
                        Message(
                            role="assistant",
                            text="Recommendation: choose Typer for the initial CLI.",
                            author_name="reviewer",
                        ),
                    ]],
                )

        class FakeHandoffBuilder:
            def __init__(self, *, name=None, participants=None, description=None, checkpoint_storage=None, termination_condition=None) -> None:
                del name, participants, description, checkpoint_storage, termination_condition

            def with_start_agent(self, agent) -> "FakeHandoffBuilder":
                del agent
                return self

            def with_autonomous_mode(self, *, agents=None, prompts=None, turn_limits=None) -> "FakeHandoffBuilder":
                del agents, prompts, turn_limits
                return self

            def add_handoff(self, source, targets, *, description=None) -> "FakeHandoffBuilder":
                del source, targets, description
                return self

            def build(self):
                return FakeBuiltHandoffWorkflow()

        fake_agents = {
            "orchestrator": FakeAgent('{"accepted": true, "summary": "The handoff workflow reached a clear recommendation."}'),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime.HandoffBuilder",
                FakeHandoffBuilder,
            ), patch(
                "ergon_studio.workflow_runtime._build_handoff_agents",
                return_value={
                    "architect": SimpleNamespace(name="architect"),
                    "researcher": SimpleNamespace(name="researcher"),
                    "brainstormer": SimpleNamespace(name="brainstormer"),
                    "reviewer": SimpleNamespace(name="reviewer"),
                },
            ):
                result = await runtime.run_workflow(
                    workflow_id="specialist-handoff",
                    goal="Discuss whether Typer or argparse is the better default for a new CLI.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            run_view = runtime.describe_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(run_view)
            assert run_view is not None
            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            thread_messages = runtime.list_thread_messages(workroom.id)
            senders = [message.sender for message in thread_messages]
            self.assertEqual(senders, ["workflow", "architect", "reviewer"])

    async def test_runtime_auto_repairs_with_workflow_metadata(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "architect": FakeAgent("Architecture ready."),
            "coder": FakeAgent("Implementation ready."),
            "tester": FakeAgent(["Tests failed on the first pass.", "Tests passed after the fix."]),
            "reviewer": FakeAgent(["Initial review complete.", "Follow-up review complete."]),
            "fixer": FakeAgent("Applied a focused fix."),
            "orchestrator": FakeAgent(
                [
                    '{"accepted": false, "summary": "The implementation still fails verification.", "findings": ["The tester reported a failing verification step."], "requires_replan": false, "replan_summary": ""}',
                    '{"action": "repair", "summary": "Run one focused repair cycle before escalating.", "tool_mode": "default"}',
                    '{"accepted": true, "summary": "The repair resolved the issue and the workflow can be accepted.", "findings": [], "requires_replan": false, "replan_summary": ""}',
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.current_step_index, 7)
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[-3:], ["fixer", "tester", "reviewer"])
            event_kinds = [event.kind for event in runtime.list_events_for_workflow_run(workflow_run.id)]
            self.assertIn("workflow_auto_repair_started", event_kinds)
            self.assertIn("workflow_repair_cycle_requested", event_kinds)
            report = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-report"
            )
            body = runtime.read_artifact_body(report.id)
            self.assertIn("## Findings", body)
            self.assertIn("Automatic repair cycles: 1", body)

    async def test_runtime_auto_replans_when_review_requires_it(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "architect": FakeAgent(["Initial plan was too narrow.", "Replanned around a cleaner CLI shape."]),
            "coder": FakeAgent(["Initial implementation ready.", "Replanned implementation ready."]),
            "tester": FakeAgent(["Initial verification exposed a structural gap.", "Replanned verification passed."]),
            "reviewer": FakeAgent(["Initial review complete.", "Replanned review complete."]),
            "fixer": FakeAgent("Not used."),
            "orchestrator": FakeAgent(
                [
                    '{"accepted": false, "summary": "The approach is structurally off.", "findings": ["The current structure does not match the goal cleanly."], "requires_replan": true, "replan_summary": "Replan around a simpler architecture before continuing."}',
                    '{"action": "replan", "summary": "Restart from architecture before continuing.", "tool_mode": "default"}',
                    '{"accepted": true, "summary": "The replanned approach now fits the goal.", "findings": [], "requires_replan": false, "replan_summary": ""}',
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.current_step_index, 8)
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[-4:], ["architect", "coder", "tester", "reviewer"])
            event_kinds = [event.kind for event in runtime.list_events_for_workflow_run(workflow_run.id)]
            self.assertIn("workflow_auto_replan_started", event_kinds)
            self.assertIn("workflow_replan_cycle_requested", event_kinds)
            report = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-report"
            )
            body = runtime.read_artifact_body(report.id)
            self.assertIn("Automatic replanning cycles: 1", body)

    async def test_runtime_clarifies_with_relevant_agent_before_replanning(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "architect": FakeAgent("Architecture ready."),
            "coder": FakeAgent("Implementation ready."),
            "tester": FakeAgent(
                [
                    "I ran the CLI and it worked, but I did not include the concrete command in the first summary.",
                    "Concrete evidence: `python3 calculator.py 5 + 3` returned `8`.",
                ]
            ),
            "reviewer": FakeAgent("Review complete."),
            "fixer": FakeAgent("Not used."),
            "orchestrator": FakeAgent(
                [
                    '{"accepted": false, "summary": "The implementation may be fine, but I need concrete verification evidence.", "findings": ["The current review summary does not show one concrete successful command run."], "requires_replan": false, "replan_summary": ""}',
                    '{"action": "clarify", "summary": "Ask the tester for one concrete successful command run before escalating.", "agent_id": "tester", "request": "Run one concrete successful command against the actual deliverable and report the exact command plus output.", "tool_mode": "none"}',
                    '{"accepted": true, "summary": "The added tester evidence resolves the gap and the workflow can be accepted.", "findings": [], "requires_replan": false, "replan_summary": ""}',
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.current_step_index, 5)
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[-2:], ["orchestrator", "tester"])
            event_kinds = [event.kind for event in runtime.list_events_for_workflow_run(workflow_run.id)]
            self.assertIn("workflow_clarification_requested", event_kinds)
            self.assertIn("workflow_clarification_cycle_requested", event_kinds)
            report = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-report"
            )
            body = runtime.read_artifact_body(report.id)
            self.assertIn("Clarification cycles: 1", body)

    async def test_runtime_uses_custom_followup_staffing_from_orchestrator(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "architect": FakeAgent("Architecture ready."),
            "coder": FakeAgent("Implementation ready."),
            "tester": FakeAgent(["Initial verification failed.", "Focused re-check passed."]),
            "reviewer": FakeAgent(["Initial review complete.", "Custom follow-up review complete."]),
            "fixer": FakeAgent("Applied the targeted correction."),
            "orchestrator": FakeAgent(
                [
                    '{"accepted": false, "summary": "The initial evidence is not good enough yet.", "findings": ["The fix needs another focused verification pass."], "requires_replan": false, "replan_summary": ""}',
                    '{"action": "repair", "summary": "Run a custom tester -> fixer -> reviewer loop.", "request": "Have the tester restate the failure first, then fix only that issue, then review the corrected result.", "step_groups": [["tester"], ["fixer"], ["reviewer"]], "tool_mode": "none"}',
                    '{"accepted": true, "summary": "The custom follow-up loop resolved the issue cleanly.", "findings": [], "requires_replan": false, "replan_summary": ""}',
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[-4:], ["orchestrator", "tester", "fixer", "reviewer"])
            event_kinds = [event.kind for event in runtime.list_events_for_workflow_run(workflow_run.id)]
            self.assertIn("workflow_auto_repair_started", event_kinds)
            self.assertIn("workflow_repair_cycle_requested", event_kinds)
            report = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-report"
            )
            body = runtime.read_artifact_body(report.id)
            self.assertIn("Automatic repair cycles: 1", body)

    async def test_runtime_recovers_blocked_step_with_orchestrator_clarification(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "coder": FakeAgent(
                [
                    "The deliverable is already implemented in the workspace and does not need another edit.",
                    "I still do not have a file edit to record because the implementation is already present.",
                    "Clarification: the current implementation is already correct, so no extra file write is required.",
                ]
            ),
            "orchestrator": FakeAgent(
                [
                    '{"action": "clarify", "summary": "Ask the coder to explain why no additional edit is needed.", "agent_id": "coder", "request": "Explain briefly why the current workspace already satisfies the goal and why no new file edit is necessary.", "tool_mode": "none"}',
                    '{"accepted": true, "summary": "The coder clarified the blocked step clearly enough to accept the result.", "findings": [], "requires_replan": false, "replan_summary": ""}',
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda agent_id: ("write_file",) if agent_id == "coder" else (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="single-agent-execution",
                    goal="Deliver the requested change cleanly.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.current_step_index, 2)
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[-2:], ["orchestrator", "coder"])
            event_kinds = [event.kind for event in runtime.list_events_for_workflow_run(workflow_run.id)]
            self.assertIn("workflow_clarification_requested", event_kinds)
            self.assertIn("workflow_clarification_cycle_requested", event_kinds)
            report = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-report"
            )
            body = runtime.read_artifact_body(report.id)
            self.assertIn("Clarification cycles: 1", body)

    async def test_runtime_uses_orchestrator_selected_initial_staffing(self) -> None:
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        fake_agents = {
            "researcher": FakeAgent("Collected the relevant context."),
            "coder": FakeAgent("Implemented the requested change."),
            "reviewer": FakeAgent("Reviewed the result and found it acceptable."),
            "orchestrator": FakeAgent(
                '{"accepted": true, "summary": "The selected team completed the work cleanly.", "findings": [], "requires_replan": false, "replan_summary": ""}'
            ),
        }

        async def fake_select(_runtime, *, workflow_id: str, goal: str, created_at: int):
            self.assertEqual(workflow_id, "standard-build")
            self.assertEqual(goal, "Build the feature end to end.")
            self.assertEqual(created_at, 1_710_755_200)
            return (("researcher",), ("coder",), ("reviewer",))

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "_select_initial_workflow_step_groups",
                autospec=True,
                side_effect=fake_select,
            ), patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime._required_tool_names",
                side_effect=lambda _agent_id: (),
            ):
                result = await runtime.run_workflow(
                    workflow_id="standard-build",
                    goal="Build the feature end to end.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            workflow_run = runtime.get_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(workflow_run)
            assert workflow_run is not None
            self.assertEqual(workflow_run.current_step_index, 3)
            team = [thread.assigned_agent_id or thread.summary for thread in runtime.list_threads_for_workflow_run(workflow_run.id)]
            self.assertEqual(team[:3], ["researcher", "coder", "reviewer"])
            graph = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(workflow_run.id)
                if artifact.kind == "workflow-graph"
            )
            body = runtime.read_artifact_body(graph.id)
            self.assertIn('step1["researcher"]', body)

    async def test_runtime_uses_orchestrator_selected_dynamic_participants(self) -> None:
        import ergon_studio.workflow_runtime as workflow_runtime
        from ergon_studio.runtime import load_runtime

        class FakeAgent:
            def __init__(self, responses: list[str] | str) -> None:
                self.responses = responses if isinstance(responses, list) else [responses]

            def create_session(self, *, session_id: str | None = None, **_: object) -> AgentSession:
                return AgentSession(session_id=session_id)

            async def run(self, messages=None, *, session=None, **_: object):
                del messages, session
                response_text = self.responses.pop(0)
                return SimpleNamespace(text=response_text)

        class FakeWorkflowResult(list):
            def __init__(self, events, outputs) -> None:
                super().__init__(events)
                self._outputs = outputs

            def get_outputs(self):
                return self._outputs

            def get_final_state(self):
                return "completed"

            def get_request_info_events(self):
                return []

        class DummyCtx:
            def __init__(self) -> None:
                self.output = None

            async def yield_output(self, value) -> None:
                self.output = value

            async def send_message(self, value, target_id=None) -> None:
                del target_id
                self.output = value

        class FakeBuiltMagenticWorkflow:
            def __init__(self, participants) -> None:
                self.participants = participants

            async def run(self, goal, include_status_events=True):
                del include_status_events
                participant_map = {participant.id: participant for participant in self.participants}
                opening = workflow_runtime.GroupChatParticipantMessage(
                    messages=[Message(role="user", text=goal, author_name="workflow")]
                )
                for participant in self.participants:
                    await participant.sync_messages(opening, DummyCtx())

                responses = []
                for agent_id in ("researcher", "coder"):
                    participant = participant_map[agent_id]
                    ctx = DummyCtx()
                    await participant.handle_request(
                        workflow_runtime.GroupChatRequestMessage(additional_instruction=f"Continue as {agent_id}."),
                        ctx,
                    )
                    response = ctx.output.message
                    responses.append(response)
                    broadcast = workflow_runtime.GroupChatParticipantMessage(messages=[response])
                    for peer in self.participants:
                        if peer.id == agent_id:
                            continue
                        await peer.sync_messages(broadcast, DummyCtx())

                return FakeWorkflowResult(
                    events=[
                        SimpleNamespace(
                            type="magentic_orchestrator",
                            data=SimpleNamespace(
                                event_type="PLAN_CREATED",
                                content=Message(
                                    role="assistant",
                                    text="Plan: researcher -> coder",
                                    author_name="magentic_manager",
                                ),
                            ),
                        )
                    ],
                    outputs=[[
                        Message(
                            role="assistant",
                            text="Dynamic workflow complete.",
                            author_name="magentic_manager",
                        )
                    ]],
                )

        class FakeMagenticBuilder:
            def __init__(self, *, participants, manager_agent=None, max_round_count=None, enable_plan_review=False) -> None:
                del manager_agent, max_round_count, enable_plan_review
                self.participants = participants

            def build(self):
                return FakeBuiltMagenticWorkflow(self.participants)

        fake_agents = {
            "researcher": FakeAgent("I found the relevant constraints and API shape."),
            "coder": FakeAgent("I implemented the first useful version."),
            "orchestrator": FakeAgent('{"accepted": true, "summary": "The smaller adaptive team completed the work cleanly."}'),
        }

        async def fake_select(_runtime, *, workflow_id: str, goal: str, created_at: int):
            self.assertEqual(workflow_id, "dynamic-open-ended")
            self.assertEqual(goal, "Build the feature adaptively.")
            self.assertEqual(created_at, 1_710_755_200)
            return (("researcher",), ("coder",))

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            with patch.object(
                type(runtime),
                "_select_initial_workflow_step_groups",
                autospec=True,
                side_effect=fake_select,
            ), patch.object(
                type(runtime),
                "build_agent",
                autospec=True,
                side_effect=lambda _runtime, agent_id: fake_agents[agent_id],
            ), patch(
                "ergon_studio.workflow_runtime.MagenticBuilder",
                FakeMagenticBuilder,
            ), patch(
                "ergon_studio.workflow_runtime._build_magentic_manager_agent",
                return_value=SimpleNamespace(),
            ):
                result = await runtime.run_workflow(
                    workflow_id="dynamic-open-ended",
                    goal="Build the feature adaptively.",
                    created_at=1_710_755_200,
                )

            self.assertEqual(result["status"], "completed")
            run_view = runtime.describe_workflow_run(result["workflow_run_id"])
            self.assertIsNotNone(run_view)
            assert run_view is not None
            workroom = run_view.steps[0].threads[0]
            self.assertEqual(workroom.kind, "group_workroom")
            self.assertIn("researcher + coder", workroom.summary or "")
            thread_messages = runtime.list_thread_messages(workroom.id)
            senders = [message.sender for message in thread_messages]
            self.assertEqual(
                senders,
                ["workflow", "researcher", "coder", "magentic_manager", "magentic_manager"],
            )
            graph = next(
                artifact for artifact in runtime.list_artifacts_for_workflow_run(run_view.workflow_run.id)
                if artifact.kind == "workflow-graph"
            )
            self.assertIn("researcher + coder", runtime.read_artifact_body(graph.id))

    def test_runtime_formats_workflow_summary_with_team_files_and_checks(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)
            workflow_run, threads = runtime.start_workflow_run(
                workflow_id="single-agent-execution",
                created_at=1_710_755_200,
            )
            thread = threads[0]
            runtime.tool_call_store.record_tool_call(
                session_id=runtime.main_session_id,
                tool_call_id="tool-call-1",
                tool_name="write_file",
                arguments={"path": "calc.py", "content": "print(5)\n"},
                result={"path": "calc.py", "status": "written"},
                status="completed",
                created_at=1_710_755_201,
                thread_id=thread.id,
                task_id=thread.parent_task_id,
                agent_id="coder",
            )
            (project_root / "calc.py").write_text("print(5)\n", encoding="utf-8")
            runtime.run_workspace_command(
                "python3 calc.py",
                created_at=1_710_755_202,
                thread_id=thread.id,
                task_id=thread.parent_task_id,
                agent_id="tester",
                require_approval=False,
            )

            summary = runtime._format_workflow_summary(
                workflow_id="single-agent-execution",
                result={
                    "workflow_run_id": workflow_run.id,
                    "status": "completed",
                    "review_summary": "ACCEPTED: Minimal working delivery.",
                },
            )

            self.assertIn("I used the `single-agent-execution` workflow.", summary)
            self.assertIn("Team: coder", summary)
            self.assertIn("Changed files:", summary)
            self.assertIn("- calc.py", summary)
            self.assertIn("Checks:", summary)
            self.assertIn("- python3 calc.py", summary)

    def test_runtime_formats_workflow_summary_with_explicit_acceptance_when_review_summary_is_blank(self) -> None:
        from ergon_studio.runtime import load_runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()

            runtime = load_runtime(project_root=project_root, home_dir=home_dir)

            summary = runtime._format_workflow_summary(
                workflow_id="standard-build",
                result={
                    "status": "completed",
                    "review_summary": "",
                    "review_accepted": True,
                },
            )

            self.assertIn("ACCEPTED:", summary)
