from __future__ import annotations

from collections.abc import Sequence

from rich.markup import escape
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
        existing_by_id = {widget.item_id: widget for widget in self._widgets}
        next_widgets: list[_TimelineItemWidget] = []
        retained_ids: set[str] = set()

        for item in self._items:
            widget = existing_by_id.get(item.item_id)
            if widget is not None and _widget_matches_item(widget, item):
                widget.update_item(item)
                next_widgets.append(widget)
                retained_ids.add(item.item_id)
                continue
            next_widgets.append(_widget_for_item(item))

        removed_widgets = [widget for widget in self._widgets if widget.item_id not in retained_ids]
        if removed_widgets:
            self.remove_children(removed_widgets)

        anchor: _TimelineItemWidget | None = None
        for widget in next_widgets:
            if widget.parent is None:
                if anchor is None:
                    if self.children:
                        self.mount(widget, before=self.children[0])
                    else:
                        self.mount(widget)
                else:
                    self.mount(widget, after=anchor)
            elif anchor is None:
                if self.children and self.children[0] is not widget:
                    self.move_child(widget, before=self.children[0])
            else:
                self.move_child(widget, after=anchor)
            anchor = widget

        self._widgets = next_widgets

    def plain_text(self) -> str:
        return "\n".join(
            widget.plain_text()
            for widget in self._widgets
            if widget.plain_text().strip()
        )


class _TimelineItemWidget(Widget):
    item_id: str

    def plain_text(self) -> str:
        raise NotImplementedError

    def update_item(self, item: TimelineItem) -> None:
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
        self.item_id = item.item_id
        self.item = item
        self.update(_message_renderable(item.sender, item.body, is_live=item.is_live))

    def update_item(self, item: TimelineItem) -> None:
        assert isinstance(item, ChatTurnItem)
        self.item_id = item.item_id
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
        self.item_id = item.item_id
        self.item = item
        self.update(_notice_body(item))

    def update_item(self, item: TimelineItem) -> None:
        assert isinstance(item, NoticeItem)
        self.item_id = item.item_id
        self.item = item
        self.update(_notice_body(item))

    def plain_text(self) -> str:
        prefix = f"{self.item.title}\n" if self.item.title else ""
        return f"{prefix}{self.item.body}".strip()


class TimelineApprovalWidget(Static, _TimelineItemWidget):
    can_focus = True
    DEFAULT_CSS = """
    TimelineApprovalWidget {
        height: auto;
        margin: 0 1 1 1;
        background: transparent;
    }

    TimelineApprovalWidget:focus {
        border: round $accent;
    }
    """

    def __init__(self, item: ApprovalItem, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item_id = item.item_id
        self.item = item
        self.approval_id = item.approval_id
        self.update(_approval_renderable(item))

    def update_item(self, item: TimelineItem) -> None:
        assert isinstance(item, ApprovalItem)
        self.item_id = item.item_id
        self.item = item
        self.approval_id = item.approval_id
        self.update(_approval_renderable(item))

    def plain_text(self) -> str:
        return f"Approval Required {self.item.action} by {self.item.requester} Reason: {self.item.reason}"

    def on_click(self) -> None:
        self.focus()


class TimelineWorkroomSegmentWidget(Collapsible, _TimelineItemWidget):
    can_focus = True
    DEFAULT_CSS = """
    TimelineWorkroomSegmentWidget {
        height: auto;
        margin: 0 1 1 2;
        padding: 0;
        background: transparent;
        border-left: solid $accent;
    }

    TimelineWorkroomSegmentWidget:focus {
        border: tall $accent;
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
        self.item_id = item.item_id
        self.item = item
        self.thread_id = item.thread_id
        self.thread_kind = item.thread_kind
        self.assigned_agent_id = item.assigned_agent_id
        self._body = Static(_workroom_body(item), classes="timeline-workroom-body")
        super().__init__(self._body, title=item.title, collapsed=False, **kwargs)

    def update_item(self, item: TimelineItem) -> None:
        assert isinstance(item, WorkroomSegmentItem)
        self.item_id = item.item_id
        self.item = item
        self.thread_id = item.thread_id
        self.thread_kind = item.thread_kind
        self.assigned_agent_id = item.assigned_agent_id
        self.title = item.title
        self._body.update(_workroom_body(item))

    def plain_text(self) -> str:
        lines = [self.item.title]
        for message in self.item.messages:
            suffix = " <live>" if message.is_live else ""
            lines.append(f"{message.sender}: {message.body}{suffix}")
        return "\n".join(lines)

    def on_click(self) -> None:
        self.focus()


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


def _widget_matches_item(widget: _TimelineItemWidget, item: TimelineItem) -> bool:
    if isinstance(item, ChatTurnItem):
        return isinstance(widget, TimelineChatTurnWidget)
    if isinstance(item, WorkroomSegmentItem):
        return isinstance(widget, TimelineWorkroomSegmentWidget)
    if isinstance(item, ApprovalItem):
        return isinstance(widget, TimelineApprovalWidget)
    if isinstance(item, NoticeItem):
        return isinstance(widget, TimelineNoticeWidget)
    return False


def _message_renderable(sender: str, body: str, *, is_live: bool = False):
    label = "you" if sender == "user" else sender
    live_suffix = "\n\n[dim]▌[/dim]" if is_live else ""
    if "```" in body or "\n#" in body:
        return Markdown(f"**{label}**\n\n{body}{live_suffix}")
    return _message_markup(sender, body, is_live=is_live)


def _notice_body(item: NoticeItem) -> str:
    title = f"[b]{item.title}[/b]\n" if item.title else ""
    return f"{title}{item.body}"


def _approval_renderable(item: ApprovalItem):
    return RichPanel(
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


def _workroom_body(item: WorkroomSegmentItem) -> str:
    body = "\n\n".join(
        _message_markup(message.sender, message.body, is_live=message.is_live)
        for message in item.messages
    )
    hint = _workroom_hint(item)
    if hint:
        return f"{body}\n\n{hint}"
    return body


def _workroom_hint(item: WorkroomSegmentItem) -> str:
    if item.thread_kind == "agent_direct" and item.assigned_agent_id:
        return "[dim]Enter to reply here · Ctrl+I to inspect[/dim]"
    return "[dim]Ctrl+I to inspect this workroom[/dim]"


def _message_markup(sender: str, body: str, *, is_live: bool = False) -> str:
    label = escape("you" if sender == "user" else sender)
    safe_body = escape(body)
    suffix = " [dim]▌[/dim]" if is_live else ""
    return f"[bold]{label}[/bold] {safe_body}{suffix}"
