from __future__ import annotations

import unittest
from pathlib import Path

from ergon_studio.storage.models import MessageRecord


class MessageRecordTests(unittest.TestCase):
    def test_message_record_stores_metadata_and_body_path_only(self) -> None:
        record = MessageRecord(
            id="msg-1",
            thread_id="thread-1",
            sender="orchestrator",
            kind="chat",
            body_path=Path("threads/thread-1/messages/msg-1.md"),
            created_at=1_710_755_200,
        )

        self.assertEqual(record.body_path, Path("threads/thread-1/messages/msg-1.md"))
        self.assertEqual(record.created_at, 1_710_755_200)
        self.assertFalse(hasattr(record, "content"))

    def test_message_record_requires_unix_time_int(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unix time int"):
            MessageRecord(
                id="msg-1",
                thread_id="thread-1",
                sender="orchestrator",
                kind="chat",
                body_path=Path("threads/thread-1/messages/msg-1.md"),
                created_at=1.5,
            )
