from __future__ import annotations

import unittest

from ergon_studio.proxy.turn_state import ProxyTurnState


class TurnStateTests(unittest.TestCase):
    def test_record_output_item_deduplicates_same_kind(self) -> None:
        state = ProxyTurnState()
        state.record_output_item("content")
        state.record_output_item("content")
        self.assertEqual(len(state.output_items), 1)

    def test_record_output_item_deduplicates_same_tool_call(self) -> None:
        state = ProxyTurnState()
        state.record_output_item("tool_call", call_id="call_1")
        state.record_output_item("tool_call", call_id="call_1")
        self.assertEqual(len(state.output_items), 1)

    def test_record_output_item_keeps_distinct_kinds(self) -> None:
        state = ProxyTurnState()
        state.record_output_item("content")
        state.record_output_item("reasoning")
        state.record_output_item("tool_call", call_id="call_1")
        self.assertEqual(len(state.output_items), 3)

    def test_append_content_records_content_item(self) -> None:
        state = ProxyTurnState()
        state.append_content("hello")
        state.append_content(" world")
        self.assertEqual(state.content, "hello world")
        self.assertEqual(len(state.output_items), 1)
        self.assertEqual(state.output_items[0].kind, "content")
