from __future__ import annotations

from collections.abc import Sequence

from rich.markdown import Markdown
from rich.panel import Panel as RichPanel
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, Static

from ergon_studio.tui.timeline_models import ApprovalItem, ChatTurnItem, NoticeItem, TimelineItem, WorkroomSegmentItem


class TimelineView(VerticalScroll):
    DEFAULT_CSS = """
    TimelineView {
        height: 1fr;
        padding: 0 0 1 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: tuple[TimelineItem, ...] = ()
        self._widgets: list[_TimelineItemWidget] = []

    def set_items(self, items: Sequence[TimelineItem]) -> None:
        self._items = tuple(items)
        self._widgets = []
        self.remove_children()
        for item in self._items:
            widget = _widget_for_item(item)
            self._widgets.append(widget)
            self.mount(widget)

    def plain_text(self) -> str:
        return "\n".join(
            widget.plain_text()
            for widget in self._widgets
            if widget.plain_text().strip()
        )


class _TimelineItemWidget(Widget):
    def plain_text(self) -> str:
        raise NotImplementedError


class TimelineChatTurnWidget(Static, _TimelineItemWidget):
    DEFAULT_CSS = """
    TimelineChatTurnWidget {
        height: auto;
        margin: 0;
        padding: 0 1 1 1;
        background: transparent;
    }
    """

    def __init__(self, item: ChatTurnItem, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item = item
        self.update(_message_renderable(item.sender, item.body, is_live=item.is_live))

    def plain_text(self) -> str:
        suffix = " <live>" if self.item.is_live else ""
        return f"{self.item.sender}: {self.item.body}{suffix}".strip()


class TimelineNoticeWidget(Static, _TimelineItemWidget):
    DEFAULT_CSS = """
    TimelineNoticeWidget {
        height: auto;
        margin: 0;
        padding: 0 1 1 1;
        color: $text-muted;
        background: transparent;
    }
    """

    def __init__(self, item: NoticeItem, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item = item
        title = f"[b]{item.title}[/b]\n" if item.title else ""
        self.update(f"{title}{item.body}")

    def plain_text(self) -> str:
        prefix = f"{self.item.title}\n" if self.item.title else ""
        return f"{prefix}{self.item.body}".strip()


class TimelineApprovalWidget(Static, _TimelineItemWidget):
    DEFAULT_CSS = """
    TimelineApprovalWidget {
        height: auto;
        margin: 0 1 1 1;
        background: transparent;
    }
    """

    def __init__(self, item: ApprovalItem, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item = item
        self.update(
            RichPanel(
                (
                    f"[bold yellow]Approval Required[/bold yellow]\n"
                    f"[{item.risk_class}] {item.action} by {item.requester}\n"
                    f"Reason: {item.reason}\n"
                    f"[dim]Ctrl+Y to approve │ Ctrl+R to reject[/dim]"
                ),
                border_style="yellow",
                title="Approval",
                expand=True,
            )
        )

    def plain_text(self) -> str:
        return f"Approval Required {self.item.action} by {self.item.requester} Reason: {self.item.reason}"


class TimelineWorkroomSegmentWidget(Collapsible, _TimelineItemWidget):
    DEFAULT_CSS = """
    TimelineWorkroomSegmentWidget {
        height: auto;
        margin: 0 1 1 2;
        padding: 0;
        background: transparent;
        border-left: solid $accent;
    }

    TimelineWorkroomSegmentWidget > CollapsibleTitle {
        background: transparent;
        padding: 0 1;
    }

    TimelineWorkroomSegmentWidget > CollapsibleTitle:hover {
        background: transparent;
    }

    TimelineWorkroomSegmentWidget > CollapsibleTitle:focus {
        background: transparent;
    }

    TimelineWorkroomSegmentWidget > Contents {
        background: transparent;
        padding: 0 0 0 1;
    }
    """

    def __init__(self, item: WorkroomSegmentItem, **kwargs) -> None:
        self.item = item
        body_text = "\n\n".join(
            _message_markup(message.sender, message.body, is_live=message.is_live)
            for message in item.messages
        )
        body = Static(body_text, classes="timeline-workroom-body")
        super().__init__(body, title=item.title, collapsed=False, **kwargs)

    def plain_text(self) -> str:
        lines = [self.item.title]
        for message in self.item.messages:
            suffix = " <live>" if message.is_live else ""
            lines.append(f"{message.sender}: {message.body}{suffix}")
        return "\n".join(lines)


def _widget_for_item(item: TimelineItem) -> _TimelineItemWidget:
    if isinstance(item, ChatTurnItem):
        return TimelineChatTurnWidget(item)
    if isinstance(item, WorkroomSegmentItem):
        return TimelineWorkroomSegmentWidget(item)
    if isinstance(item, ApprovalItem):
        return TimelineApprovalWidget(item)
    if isinstance(item, NoticeItem):
        return TimelineNoticeWidget(item)
    raise TypeError(f"unsupported timeline item: {type(item)!r}")


def _message_renderable(sender: str, body: str, *, is_live: bool = False):
    label = "you" if sender == "user" else sender
    live_suffix = "\n\n[dim]▌[/dim]" if is_live else ""
    if "```" in body or "\n#" in body:
        return Markdown(f"**{label}**\n\n{body}{live_suffix}")
    return _message_markup(sender, body, is_live=is_live)


def _message_markup(sender: str, body: str, *, is_live: bool = False) -> str:
    label = "you" if sender == "user" else sender
    suffix = " [dim]▌[/dim]" if is_live else ""
    return f"[bold]{label}[/bold] {body}{suffix}"
