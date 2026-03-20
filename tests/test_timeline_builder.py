from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergon_studio.runtime import load_runtime
from ergon_studio.tui.timeline_builder import build_session_timeline
from ergon_studio.tui.timeline_models import ApprovalItem, ChatTurnItem, NoticeItem, WorkroomSegmentItem


class TimelineBuilderTests(unittest.TestCase):
    def test_builder_includes_live_main_chat_draft(self) -> None:
        runtime = _make_runtime(self)
        runtime.live_state.start_draft(
            draft_id="draft-1",
            thread_id=runtime.main_thread_id,
            sender="orchestrator",
            kind="chat",
            created_at=10,
        )
        runtime.live_state.append_delta(
            draft_id="draft-1",
            delta="Working on it",
            created_at=11,
        )

        items = build_session_timeline(runtime)

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], ChatTurnItem)
        assert isinstance(items[0], ChatTurnItem)
        self.assertTrue(items[0].is_live)
        self.assertEqual(items[0].body, "Working on it")

    def test_builder_includes_live_workroom_draft_inside_segment(self) -> None:
        runtime = _make_runtime(self)
        thread = runtime.create_thread(
            thread_id="thread-coder",
            kind="agent_direct",
            created_at=10,
            assigned_agent_id="coder",
            summary="Implementation",
        )
        runtime.live_state.start_draft(
            draft_id="draft-1",
            thread_id=thread.id,
            sender="coder",
            kind="chat",
            created_at=11,
        )
        runtime.live_state.append_delta(
            draft_id="draft-1",
            delta="Implementing B",
            created_at=12,
        )

        items = build_session_timeline(runtime)

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], WorkroomSegmentItem)
        assert isinstance(items[0], WorkroomSegmentItem)
        self.assertEqual(len(items[0].messages), 1)
        self.assertTrue(items[0].messages[0].is_live)
        self.assertEqual(items[0].messages[0].body, "Implementing B")

    def test_builder_keeps_main_chat_turns_as_individual_items(self) -> None:
        runtime = _make_runtime(self)
        runtime.append_message_to_main_thread(
            message_id="m-1",
            sender="user",
            kind="chat",
            body="We should do A",
            created_at=10,
        )
        runtime.append_message_to_main_thread(
            message_id="m-2",
            sender="orchestrator",
            kind="chat",
            body="OK",
            created_at=11,
        )

        items = build_session_timeline(runtime)

        self.assertEqual(len(items), 2)
        self.assertTrue(all(isinstance(item, ChatTurnItem) for item in items))
        self.assertEqual([item.sender for item in items], ["user", "orchestrator"])

    def test_builder_groups_contiguous_workroom_messages_into_one_segment(self) -> None:
        runtime = _make_runtime(self)
        thread = runtime.create_thread(
            thread_id="thread-architect",
            kind="agent_direct",
            created_at=10,
            assigned_agent_id="architect",
            summary="Architecture discussion",
        )
        runtime.append_message_to_thread(
            thread_id=thread.id,
            message_id="m-1",
            sender="orchestrator",
            kind="assignment",
            body="User wants A. How can we achieve that?",
            created_at=11,
        )
        runtime.append_message_to_thread(
            thread_id=thread.id,
            message_id="m-2",
            sender="architect",
            kind="chat",
            body="By doing B.",
            created_at=12,
        )

        items = build_session_timeline(runtime)

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], WorkroomSegmentItem)
        assert isinstance(items[0], WorkroomSegmentItem)
        self.assertEqual(items[0].title, "Orchestrator <-> architect")
        self.assertEqual([message.sender for message in items[0].messages], ["orchestrator", "architect"])

    def test_builder_splits_same_thread_when_main_chat_interrupts_it(self) -> None:
        runtime = _make_runtime(self)
        thread = runtime.create_thread(
            thread_id="thread-coder",
            kind="agent_direct",
            created_at=10,
            assigned_agent_id="coder",
            summary="Implementation",
        )
        runtime.append_message_to_thread(
            thread_id=thread.id,
            message_id="t-1",
            sender="orchestrator",
            kind="assignment",
            body="Implement B.",
            created_at=11,
        )
        runtime.append_message_to_main_thread(
            message_id="m-1",
            sender="user",
            kind="chat",
            body="Actually we also need C.",
            created_at=12,
        )
        runtime.append_message_to_thread(
            thread_id=thread.id,
            message_id="t-2",
            sender="coder",
            kind="chat",
            body="I will include C as well.",
            created_at=13,
        )

        items = build_session_timeline(runtime)

        self.assertEqual(len(items), 3)
        self.assertIsInstance(items[0], WorkroomSegmentItem)
        self.assertIsInstance(items[1], ChatTurnItem)
        self.assertIsInstance(items[2], WorkroomSegmentItem)

    def test_builder_includes_pending_approvals_and_notices_in_order(self) -> None:
        runtime = _make_runtime(self)
        runtime.append_message_to_main_thread(
            message_id="m-1",
            sender="user",
            kind="chat",
            body="Run the command.",
            created_at=10,
        )
        runtime.request_approval(
            approval_id="approval-1",
            requester="coder",
            action="run_command",
            risk_class="high",
            reason="Need to install a dependency.",
            created_at=11,
        )
        notices = (
            NoticeItem(
                item_id="notice-1",
                title="Permission mode",
                body="Auto-approve is enabled.",
                level="info",
                created_at=12,
            ),
        )

        items = build_session_timeline(runtime, notices=notices)

        self.assertEqual(len(items), 3)
        self.assertIsInstance(items[0], ChatTurnItem)
        self.assertIsInstance(items[1], ApprovalItem)
        self.assertIsInstance(items[2], NoticeItem)

    def test_builder_can_hide_rewound_main_messages(self) -> None:
        runtime = _make_runtime(self)
        runtime.append_message_to_main_thread(
            message_id="m-1",
            sender="user",
            kind="chat",
            body="First",
            created_at=10,
        )
        runtime.append_message_to_main_thread(
            message_id="m-2",
            sender="orchestrator",
            kind="chat",
            body="Second",
            created_at=11,
        )

        items = build_session_timeline(runtime, hidden_main_message_ids={"m-2"})

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], ChatTurnItem)
        assert isinstance(items[0], ChatTurnItem)
        self.assertEqual(items[0].message_id, "m-1")


def _make_runtime(test_case: unittest.TestCase):
    temp_dir = tempfile.TemporaryDirectory()
    test_case.addCleanup(temp_dir.cleanup)
    base = Path(temp_dir.name)
    project_root = base / "repo"
    home_dir = base / "home"
    project_root.mkdir()
    home_dir.mkdir()
    runtime = load_runtime(project_root=project_root, home_dir=home_dir)
    return runtime
