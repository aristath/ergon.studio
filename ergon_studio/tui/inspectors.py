from __future__ import annotations

import json
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static

from ergon_studio.runtime import RuntimeContext


@dataclass(frozen=True)
class InspectorEntry:
    entry_id: str
    label: str
    detail: str


class InspectorScreen(ModalScreen[None]):
    BINDINGS = [("escape", "close", "Close")]

    CSS = """
    InspectorScreen {
        align: center middle;
    }

    #inspector-container {
        width: 90%;
        height: 85%;
        border: round $accent;
        background: transparent;
        padding: 1;
    }

    #inspector-title {
        height: auto;
        padding: 0 0 1 0;
    }

    #inspector-body {
        height: 1fr;
    }

    #inspector-list {
        width: 34;
        min-width: 24;
        height: 1fr;
        border: round $surface;
    }

    #inspector-empty {
        width: 34;
        min-width: 24;
        height: 1fr;
        border: round $surface;
        padding: 1;
    }

    #inspector-detail {
        width: 1fr;
        height: 1fr;
        border: round $surface;
        padding: 1;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        entries: list[InspectorEntry],
        empty_message: str,
    ) -> None:
        super().__init__()
        self.title = title
        self.entries = entries
        self.empty_message = empty_message

    def compose(self) -> ComposeResult:
        with Vertical(id="inspector-container"):
            yield Static(f"[b]{self.title}[/b]\nEsc to close.", id="inspector-title")
            with Horizontal(id="inspector-body"):
                if self.entries:
                    yield OptionList(*(entry.label for entry in self.entries), id="inspector-list")
                else:
                    yield Static(self.empty_message, id="inspector-empty")
                with VerticalScroll(id="inspector-detail"):
                    yield Static(self._initial_detail(), id="inspector-detail-body")

    def on_mount(self) -> None:
        if not self.entries:
            return
        option_list = self.query_one("#inspector-list", OptionList)
        option_list.highlighted = 0
        self._update_detail(0)

    def action_close(self) -> None:
        self.dismiss()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "inspector-list":
            return
        index = event.option_list.highlighted
        if index is None:
            return
        self._update_detail(index)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "inspector-list":
            return
        index = event.option_list.highlighted
        if index is None:
            return
        self._update_detail(index)

    def _initial_detail(self) -> str:
        if not self.entries:
            return self.empty_message
        return self.entries[0].detail

    def _update_detail(self, index: int) -> None:
        if index < 0 or index >= len(self.entries):
            return
        self.query_one("#inspector-detail-body", Static).update(self.entries[index].detail)


def build_thread_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for thread in runtime.list_threads():
        if thread.id == runtime.main_thread_id:
            continue
        entry = build_thread_entry(runtime, thread.id)
        if entry is not None:
            entries.append(entry)
    return entries


def build_team_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for agent_id in runtime.list_agent_ids():
        provider_name = runtime.assigned_provider_name(agent_id)
        provider = runtime.provider_details(provider_name) if provider_name is not None else None
        model = "-"
        if isinstance(provider, dict):
            raw_model = provider.get("model")
            if isinstance(raw_model, str) and raw_model.strip():
                model = raw_model.strip()
        lines = [
            f"[b]{agent_id}[/b]",
            f"Status: {runtime.agent_status_summary(agent_id)}",
            f"Provider: {provider_name or '-'}",
            f"Model: {model}",
        ]
        reason = runtime.agent_unavailable_reason(agent_id)
        if reason is not None:
            lines.extend(["", "[b]Issue[/b]", reason])
        entries.append(
            InspectorEntry(
                entry_id=agent_id,
                label=f"{agent_id} · {runtime.agent_status_summary(agent_id)}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_thread_entry(runtime: RuntimeContext, thread_id: str) -> InspectorEntry | None:
    thread = runtime.get_thread(thread_id)
    if thread is None or thread.id == runtime.main_thread_id:
        return None
    messages = runtime.list_thread_messages(thread.id)
    lines = [
        f"[b]{thread.summary or thread.id}[/b]",
        f"ID: {thread.id}",
        f"Kind: {thread.kind}",
        f"Agent: {thread.assigned_agent_id or '-'}",
        f"Task: {thread.parent_task_id or '-'}",
        f"Parent thread: {thread.parent_thread_id or '-'}",
        "",
        "[b]Transcript[/b]",
    ]
    if not messages:
        lines.append("No messages yet.")
    else:
        for message in messages:
            body = runtime.conversation_store.read_message_body(message).rstrip("\n")
            lines.append(f"{message.sender}: {body}")
            lines.append("")
    label = f"{thread.assigned_agent_id or thread.kind} · {thread.summary or thread.id}"
    return InspectorEntry(
        entry_id=thread.id,
        label=label,
        detail="\n".join(lines).strip(),
    )


def build_task_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for task in runtime.list_tasks():
        related_threads = [
            thread.id
            for thread in runtime.list_threads()
            if thread.parent_task_id == task.id
        ]
        lines = [
            f"[b]{task.title}[/b]",
            f"ID: {task.id}",
            f"State: {task.state}",
            f"Parent task: {task.parent_task_id or '-'}",
        ]
        whiteboard = runtime.get_task_whiteboard(task.id)
        if whiteboard is not None:
            whiteboard_text = runtime.read_task_whiteboard_text(task.id).strip()
            if whiteboard_text:
                lines.extend(["", "[b]Whiteboard[/b]", whiteboard_text])
        if related_threads:
            lines.extend(["", "[b]Threads[/b]"])
            lines.extend(f"- {thread_id}" for thread_id in related_threads)
        entries.append(
            InspectorEntry(
                entry_id=task.id,
                label=f"{task.state} · {task.title}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_workflow_run_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for workflow_run in runtime.list_workflow_runs():
        run_view = runtime.describe_workflow_run(workflow_run.id)
        lines = [
            f"[b]{workflow_run.workflow_id}[/b]",
            f"Run ID: {workflow_run.id}",
            f"State: {workflow_run.state}",
            f"Current step: {workflow_run.current_step_index}",
            f"Root task: {workflow_run.root_task_id or '-'}",
        ]
        if run_view is not None and run_view.steps:
            lines.extend(["", "[b]Steps[/b]"])
            for step in run_view.steps:
                lines.append(f"- {step.task.state} · {step.task.title}")
                for thread in step.threads:
                    agent = thread.assigned_agent_id or thread.kind
                    lines.append(f"  - {agent}: {thread.summary or thread.id}")
        entries.append(
            InspectorEntry(
                entry_id=workflow_run.id,
                label=f"{workflow_run.state} · {workflow_run.workflow_id}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_workflow_definition_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for summary in runtime.list_workflow_summaries():
        detail = runtime.describe_workflow_definition(str(summary["id"]))
        lines = [
            f"[b]{detail['name']}[/b]",
            f"ID: {detail['id']}",
            f"Orchestration: {detail['orchestration']}",
            "",
            "[b]Step groups[/b]",
        ]
        for group in detail["step_groups"]:
            lines.append(f"- {' + '.join(group)}")
        purpose = str(detail["sections"].get("Purpose", "")).strip()
        if purpose:
            lines.extend(["", "[b]Purpose[/b]", purpose])
        exit_conditions = str(detail["sections"].get("Exit Conditions", "")).strip()
        if exit_conditions:
            lines.extend(["", "[b]Exit Conditions[/b]", exit_conditions])
        entries.append(
            InspectorEntry(
                entry_id=str(summary["id"]),
                label=f"{summary['orchestration']} · {summary['name']}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_artifact_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for artifact in runtime.list_artifacts():
        body = runtime.read_artifact_body(artifact.id).strip()
        lines = [
            f"[b]{artifact.title}[/b]",
            f"ID: {artifact.id}",
            f"Kind: {artifact.kind}",
            f"Task: {artifact.task_id or '-'}",
            f"Thread: {artifact.thread_id or '-'}",
            "",
            body or "(empty artifact)",
        ]
        entries.append(
            InspectorEntry(
                entry_id=artifact.id,
                label=f"{artifact.kind} · {artifact.title}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_memory_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for fact in runtime.list_memory_facts():
        lines = [
            f"[b]{fact.kind}[/b]",
            f"ID: {fact.id}",
            f"Scope: {fact.scope}",
            f"Source: {fact.source or '-'}",
        ]
        if fact.tags:
            lines.append(f"Tags: {', '.join(fact.tags)}")
        lines.extend(["", fact.content])
        entries.append(
            InspectorEntry(
                entry_id=fact.id,
                label=f"{fact.scope} · {fact.kind}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_approval_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for approval in runtime.list_approvals():
        payload = runtime.read_approval_payload(approval.id)
        payload_text = json.dumps(payload, indent=2, sort_keys=True) if payload else "(no payload)"
        lines = [
            f"[b]{approval.action}[/b]",
            f"ID: {approval.id}",
            f"Status: {approval.status}",
            f"Requester: {approval.requester}",
            f"Risk: {approval.risk_class}",
            f"Task: {approval.task_id or '-'}",
            f"Thread: {approval.thread_id or '-'}",
            "",
            f"Reason: {approval.reason}",
            "",
            "[b]Payload[/b]",
            payload_text,
        ]
        entries.append(
            InspectorEntry(
                entry_id=approval.id,
                label=f"{approval.status} · {approval.action}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries


def build_event_entries(runtime: RuntimeContext) -> list[InspectorEntry]:
    entries: list[InspectorEntry] = []
    for event in runtime.list_events():
        lines = [
            f"[b]{event.kind}[/b]",
            f"ID: {event.id}",
            f"Thread: {event.thread_id or '-'}",
            f"Task: {event.task_id or '-'}",
            "",
            event.summary,
        ]
        entries.append(
            InspectorEntry(
                entry_id=event.id,
                label=f"{event.kind} · {event.summary[:48]}",
                detail="\n".join(lines).strip(),
            )
        )
    return entries
