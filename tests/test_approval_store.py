from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.approval_store import ApprovalStore
from ergon_studio.storage.sqlite import initialize_database


class ApprovalStoreTests(unittest.TestCase):
    def test_request_and_list_approvals_persist_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "repo"
            home_dir = base / "home"
            project_root.mkdir()
            home_dir.mkdir()
            paths = bootstrap_workspace(project_root=project_root, home_dir=home_dir)
            initialize_database(paths.state_db_path)
            store = ApprovalStore(paths)

            first = store.request_approval(
                session_id="session-main",
                approval_id="approval-1",
                requester="coder",
                action="write_file",
                risk_class="moderate",
                reason="Update project scaffold",
                created_at=10,
            )
            second = store.request_approval(
                session_id="session-main",
                approval_id="approval-2",
                requester="orchestrator",
                action="run_command",
                risk_class="high",
                reason="Install dependencies",
                created_at=20,
            )

            self.assertEqual(first.status, "pending")
            self.assertEqual(second.action, "run_command")
            self.assertEqual(
                [approval.id for approval in store.list_approvals("session-main")],
                ["approval-1", "approval-2"],
            )
