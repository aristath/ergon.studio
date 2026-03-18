from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual([thread.id for thread in runtime.list_threads()], ["thread-main"])
            self.assertEqual(runtime.list_main_messages(), [])
            self.assertEqual(runtime.list_events(), [])

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
