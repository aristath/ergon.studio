from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.tool_call_store import ToolCallStore


class ToolCallStoreTests(unittest.TestCase):
    def test_record_tool_call_persists_request_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            store = ToolCallStore(paths)

            record = store.record_tool_call(
                session_id="session-main",
                tool_call_id="tool-call-1",
                tool_name="read_file",
                arguments={"path": "README.md"},
                result={"content": "hello"},
                status="completed",
                created_at=10,
                thread_id="thread-main",
                agent_id="orchestrator",
            )

            self.assertEqual(record.tool_name, "read_file")
            self.assertEqual([tool_call.id for tool_call in store.list_tool_calls("session-main")], ["tool-call-1"])
            self.assertIn('"path": "README.md"', store.read_request(record))
            self.assertIn('"content": "hello"', store.read_response(record))
