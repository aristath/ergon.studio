from __future__ import annotations

import unittest

try:
    from textual.app import App, ComposeResult
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("textual is not installed") from exc

from ergon_studio.tui.timeline_models import ChatTurnItem
from ergon_studio.tui.timeline_widgets import TimelineChatTurnWidget, TimelineView


class _TimelineTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield TimelineView(id="timeline")


class TimelineWidgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeline_reuses_widgets_for_same_item_id(self) -> None:
        app = _TimelineTestApp()

        async with app.run_test():
            timeline = app.query_one("#timeline", TimelineView)
            first_item = ChatTurnItem(
                item_id="chat-message-1",
                message_id="message-1",
                sender="orchestrator",
                kind="chat",
                body="Working on it",
                created_at=1,
                is_live=True,
            )
            timeline.set_items((first_item,))
            first_widget = timeline.query(TimelineChatTurnWidget).first()

            updated_item = ChatTurnItem(
                item_id="chat-message-1",
                message_id="message-1",
                sender="orchestrator",
                kind="chat",
                body="Working on it now",
                created_at=1,
                is_live=False,
            )
            timeline.set_items((updated_item,))
            second_widget = timeline.query(TimelineChatTurnWidget).first()

            self.assertIs(first_widget, second_widget)
            self.assertEqual(timeline.plain_text(), "orchestrator: Working on it now")
