from __future__ import annotations

import time
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Static, TextArea

from ergon_studio.runtime import RuntimeContext


class Panel(Static):
    def __init__(self, title: str, body: str, *, panel_id: str, classes: str | None = None) -> None:
        self.title_text = title
        self.body = body
        super().__init__(self._render_panel(title, body), id=panel_id, classes=classes)

    @staticmethod
    def _render_panel(title: str, body: str) -> str:
        return f"[b]{title}[/b]\n{body}"

    def set_body(self, body: str) -> None:
        self.body = body
        self.update(self._render_panel(self.title_text, body))


class DefinitionEditorScreen(ModalScreen[None]):
    BINDINGS = [
        ("ctrl+s", "save", "Save"),
        ("escape", "cancel", "Cancel"),
    ]
    CSS = """
    #definition-editor-container {
      width: 90%;
      height: 90%;
      border: round $accent;
      background: $surface;
      padding: 1;
    }

    #definition-editor {
      height: 1fr;
      margin: 1 0;
    }

    #definition-error {
      color: $error;
      height: auto;
    }
    """

    def __init__(self, *, title: str, initial_text: str, on_save, language: str = "markdown") -> None:
        super().__init__()
        self.title = title
        self.initial_text = initial_text
        self.on_save = on_save
        self.language = language

    def compose(self) -> ComposeResult:
        with Vertical(id="definition-editor-container"):
            yield Static(f"[b]{self.title}[/b]\n`Ctrl+S` to save, `Esc` to cancel.")
            yield TextArea(self.initial_text, id="definition-editor", language=self.language)
            yield Static("", id="definition-error")

    def action_cancel(self) -> None:
        self.dismiss()

    def action_save(self) -> None:
        editor = self.query_one("#definition-editor", TextArea)
        error = self.query_one("#definition-error", Static)
        try:
            self.on_save(editor.text)
        except ValueError as exc:
            error.update(str(exc))
            return
        self.dismiss()


class ErgonStudioApp(App[None]):
    TITLE = "ergon.studio"
    BINDINGS = [
        ("escape", "clear_workflow_run_focus", "Clear Run Focus"),
        ("f1", "previous_approval", "Previous Approval"),
        ("f2", "next_approval", "Next Approval"),
        ("f3", "previous_workflow_run", "Previous Run"),
        ("f4", "next_workflow_run", "Next Run"),
        ("f5", "start_selected_workflow", "Start Workflow"),
        ("f6", "advance_selected_workflow_run", "Advance Workflow"),
        ("f10", "request_fix_cycle_for_selected_workflow_run", "Request Fix Cycle"),
        ("ctrl+j", "next_thread", "Next Thread"),
        ("ctrl+k", "previous_thread", "Previous Thread"),
        ("alt+h", "previous_activity_event", "Previous Activity"),
        ("alt+i", "next_command_run", "Next Command"),
        ("alt+j", "next_task", "Next Task"),
        ("alt+k", "previous_task", "Previous Task"),
        ("alt+l", "next_activity_event", "Next Activity"),
        ("alt+n", "next_memory_fact", "Next Memory"),
        ("alt+p", "previous_memory_fact", "Previous Memory"),
        ("alt+u", "previous_command_run", "Previous Command"),
        ("ctrl+n", "next_agent", "Next Agent"),
        ("ctrl+p", "previous_agent", "Previous Agent"),
        ("ctrl+a", "open_selected_agent_thread", "Open Agent Thread"),
        ("ctrl+t", "edit_selected_agent_definition", "Edit Agent"),
        ("ctrl+w", "edit_selected_task_whiteboard", "Edit Whiteboard"),
        ("ctrl+y", "approve_selected_approval", "Approve"),
        ("ctrl+r", "reject_selected_approval", "Reject"),
        ("f7", "previous_workflow", "Previous Workflow"),
        ("f8", "next_workflow", "Next Workflow"),
        ("f9", "edit_selected_workflow_definition", "Edit Workflow"),
        ("f11", "previous_artifact", "Previous Artifact"),
        ("f12", "next_artifact", "Next Artifact"),
        ("ctrl+x", "run_workspace_command", "Run Command"),
        ("ctrl+g", "edit_global_config", "Edit Config"),
        ("ctrl+e", "edit_orchestrator_definition", "Edit Orchestrator"),
    ]
    CSS = """
    Screen {
      layout: vertical;
    }

    #workspace {
      height: 1fr;
      layout: horizontal;
    }

    #left-sidebar {
      width: 28;
      layout: vertical;
    }

    #center-column {
      width: 1fr;
      layout: vertical;
    }

    #right-sidebar {
      width: 34;
      layout: vertical;
    }

    .panel {
      border: round $accent;
      padding: 1;
      margin: 0 1 1 0;
      height: 1fr;
    }

    #main-chat {
      height: 2fr;
    }

    #activity {
      height: 1fr;
    }

    #composer-input {
      dock: bottom;
      margin: 0 1 1 0;
    }
    """

    def __init__(self, runtime: RuntimeContext) -> None:
        super().__init__()
        self.runtime = runtime
        self.selected_thread_id = runtime.main_thread_id
        self.selected_agent_id = "orchestrator"
        self.selected_workflow_id = "standard-build"
        self.selected_workflow_run_id: str | None = None
        self.selected_approval_id: str | None = None
        self.selected_artifact_id: str | None = None
        self.selected_memory_fact_id: str | None = None
        self.selected_task_id: str | None = None
        self.selected_event_id: str | None = None
        self.selected_command_run_id: str | None = None
        self._time_cursor = int(time.time())
        self._normalize_selected_approval()
        self._normalize_selected_artifact()
        self._normalize_selected_memory_fact()
        self._normalize_selected_task()
        self._normalize_selected_event()
        self._normalize_selected_command_run()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="left-sidebar"):
                yield Panel("Tasks", self._render_tasks_body(), panel_id="tasks", classes="panel")
                yield Panel("Workflows", self._render_workflows_body(), panel_id="workflows", classes="panel")
                yield Panel("Runs", self._render_workflow_runs_body(), panel_id="workflow-runs", classes="panel")
                yield Panel("Threads", self._render_threads_body(), panel_id="threads", classes="panel")
                yield Panel("Activity", self._render_activity_body(), panel_id="activity", classes="panel")
            with Vertical(id="center-column"):
                yield Panel(
                    "Main Chat",
                    self._render_main_chat_body(),
                    panel_id="main-chat",
                    classes="panel",
                )
                yield Panel(
                    "Selected Thread",
                    self._render_selected_thread_body(),
                    panel_id="selected-thread",
                    classes="panel",
                )
                yield Panel("Commands", self._render_commands_body(), panel_id="commands", classes="panel")
                yield Panel("Artifacts", self._render_artifacts_body(), panel_id="artifacts", classes="panel")
            with Vertical(id="right-sidebar"):
                yield Panel("Team", self._render_team_body(), panel_id="team", classes="panel")
                yield Panel("Approvals", self._render_approvals_body(), panel_id="approvals", classes="panel")
                yield Panel(
                    "Memory",
                    self._render_memory_body(),
                    panel_id="memory",
                    classes="panel",
                )
                yield Panel(
                    "Settings",
                    self._render_settings_body(),
                    panel_id="settings",
                    classes="panel",
                )
        yield Input(placeholder="Message the orchestrator...", id="composer-input")
        yield Footer()

    def _render_threads_body(self) -> str:
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                threads = self.runtime.list_threads_for_workflow_run(self.selected_workflow_run_id)
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if not threads:
                    lines.append("No threads for selected run.")
                    return "\n".join(lines)
                lines.extend(
                    f"{'> ' if thread.id == self.selected_thread_id else '  '}{thread.id} ({self._thread_label(thread)})"
                    for thread in threads
                )
                return "\n".join(lines)

        threads = self.runtime.list_threads()
        if not threads:
            return "No threads yet."
        return "\n".join(
            f"{'> ' if thread.id == self.selected_thread_id else '  '}{thread.id} ({self._thread_label(thread)})"
            for thread in threads
        )

    def _render_tasks_body(self) -> str:
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                selected_task = self.runtime.get_task(self.selected_task_id) if self.selected_task_id is not None else None
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if run_view.root_task is not None:
                    lines.append(
                        f"Root: {'> ' if run_view.root_task.id == self.selected_task_id else ''}{run_view.root_task.id} [{run_view.root_task.state}] {run_view.root_task.title}"
                    )
                else:
                    lines.append("Root: missing")

                if not run_view.steps:
                    lines.append("No workflow steps yet.")
                    return "\n".join(lines)

                for step in run_view.steps:
                    lines.append(
                        f"  task: {'> ' if step.task.id == self.selected_task_id else ''}{step.task.id} [{step.task.state}] {step.task.title}"
                    )
                    if not step.threads:
                        lines.append("    thread: none")
                        continue
                    for thread in step.threads:
                        lines.append(f"    thread: {thread.id} ({self._thread_label(thread)})")
                lines.extend(self._task_preview_lines(selected_task))
                return "\n".join(lines)

        tasks = self._visible_tasks()
        if not tasks:
            return "No tasks yet."
        selected_task = self.runtime.get_task(self.selected_task_id) if self.selected_task_id is not None else None
        return "\n".join(
            [
                *[
                    f"{'> ' if task.id == self.selected_task_id else '  '}{task.id} [{task.state}] {task.title}"
                    for task in tasks
                ],
                *self._task_preview_lines(selected_task),
            ]
        )

    def _render_main_chat_body(self) -> str:
        messages = self.runtime.list_main_messages()
        if not messages:
            return (
                f"Project UUID: {self.runtime.paths.project_uuid}\n"
                f"Workspace: {self.runtime.paths.project_root}\n"
                "No messages yet."
            )

        rendered_messages = []
        for message in messages:
            body = self.runtime.conversation_store.read_message_body(message).rstrip("\n")
            rendered_messages.append(f"[{message.sender}] {body}")
        return "\n\n".join(rendered_messages)

    def _render_selected_thread_body(self) -> str:
        thread = self.runtime.get_thread(self.selected_thread_id)
        header_lines = [self.selected_thread_id]
        if thread is not None:
            header_lines.append(f"Kind: {thread.kind}")
            if thread.assigned_agent_id is not None:
                header_lines.append(f"Agent: {thread.assigned_agent_id}")
            if thread.parent_task_id is not None:
                header_lines.append(f"Task: {thread.parent_task_id}")

        messages = self.runtime.list_thread_messages(self.selected_thread_id)
        if not messages:
            return "\n".join(header_lines + ["No messages yet."])

        rendered_messages = header_lines + [""]
        for message in messages:
            body = self.runtime.conversation_store.read_message_body(message).rstrip("\n")
            rendered_messages.append(f"[{message.sender}] {body}")
        return "\n\n".join(rendered_messages)

    def _render_activity_body(self) -> str:
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                events = self._visible_events()
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if not events:
                    lines.append("No activity for selected run.")
                    return "\n".join(lines)
                lines.extend(self._event_lines(events))
                lines.extend(self._event_preview_lines(events))
                return "\n".join(lines)

        events = self._visible_events()
        if not events:
            return "No activity yet."
        return "\n".join(self._event_lines(events) + self._event_preview_lines(events))

    def _render_approvals_body(self) -> str:
        approvals = self._visible_approvals()
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if not approvals:
                    lines.append("No approvals for selected run.")
                    return "\n".join(lines)
                lines.extend(self._approval_lines(approvals))
                lines.extend(self._approval_preview_lines(approvals))
                return "\n".join(lines)

        if not approvals:
            return "No approvals pending."
        return "\n".join(self._approval_lines(approvals) + self._approval_preview_lines(approvals))

    def _render_memory_body(self) -> str:
        facts = self.runtime.list_memory_facts()
        whiteboard_lines = self._task_whiteboard_lines()
        if not facts and not whiteboard_lines:
            return "No memory facts yet."
        body_lines: list[str] = []
        body_lines.extend(whiteboard_lines)
        if facts:
            if body_lines:
                body_lines.append("")
            body_lines.extend(self._memory_lines(facts))
            body_lines.extend(self._memory_preview_lines(facts))
        return "\n".join(body_lines)

    def _render_artifacts_body(self) -> str:
        artifacts = self._visible_artifacts()
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if not artifacts:
                    lines.append("No artifacts for selected run.")
                    return "\n".join(lines)
                lines.extend(self._artifact_lines(artifacts))
                lines.extend(self._artifact_preview_lines())
                return "\n".join(lines)

        if not artifacts:
            return "No artifacts yet."
        return "\n".join(self._artifact_lines(artifacts) + self._artifact_preview_lines())

    def _render_commands_body(self) -> str:
        command_runs = self._visible_command_runs()
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                lines = [
                    (
                        f"Run: {run_view.workflow_run.id} "
                        f"[{run_view.workflow_run.state}] {run_view.workflow_run.workflow_id}"
                    )
                ]
                if not command_runs:
                    lines.append("No commands for selected run.")
                    return "\n".join(lines)
                lines.extend(self._command_lines(command_runs))
                lines.extend(self._command_preview_lines())
                return "\n".join(lines)

        if not command_runs:
            return "No commands yet."
        return "\n".join(self._command_lines(command_runs) + self._command_preview_lines())

    def _render_settings_body(self) -> str:
        providers = self.runtime.list_provider_ids()
        agents = self.runtime.list_agent_ids()
        workflows = self.runtime.list_workflow_ids()
        orchestrator_status = self.runtime.agent_status_summary("orchestrator")
        provider_map = self.runtime.registry.config.get("providers", {})
        role_assignments = self.runtime.registry.config.get("role_assignments", {})

        provider_text = ", ".join(providers) if providers else "none"
        agent_text = ", ".join(agents)
        workflow_text = ", ".join(workflows)
        assignment_lines = [
            f"{role}->{provider}"
            for role, provider in sorted(role_assignments.items())
        ]
        provider_lines = []
        for provider_id in providers:
            provider = provider_map.get(provider_id, {})
            provider_type = str(provider.get("type", "unknown"))
            model = str(provider.get("model", "unknown-model"))
            base_url = str(provider.get("base_url", ""))
            capabilities = provider.get("capabilities", {})
            capability_text = ""
            if isinstance(capabilities, dict) and capabilities:
                capability_items = [f"{key}={value}" for key, value in sorted(capabilities.items())]
                capability_text = f" capabilities({', '.join(capability_items)})"
            if base_url:
                provider_lines.append(f"{provider_id}: {provider_type} {model} @ {base_url}{capability_text}")
            else:
                provider_lines.append(f"{provider_id}: {provider_type} {model}{capability_text}")

        return (
            f"Config: {self.runtime.paths.config_path}\n"
            f"Agents Dir: {self.runtime.paths.agents_dir}\n"
            f"Workflows Dir: {self.runtime.paths.workflows_dir}\n"
            f"Orchestrator: {orchestrator_status}\n"
            "Shortcuts: Esc clear run focus, F3/F4 runs, F5 start workflow, F6 advance workflow, F10 fix cycle, Ctrl+N/P team, Ctrl+A thread, Ctrl+T agent, F7/F8 workflow, F9 edit workflow, Ctrl+G config\n"
            "Run Command: Ctrl+X\n"
            "Approvals: F1/F2 select, Ctrl+Y approve, Ctrl+R reject\n"
            "Commands: Alt+U / Alt+I select\n"
            "Artifacts: F11/F12 select\n"
            "Activity: Alt+H / Alt+L select\n"
            "Tasks: Alt+J / Alt+K select\n"
            "Memory: Alt+N / Alt+P select\n"
            "Whiteboard: Ctrl+W edit selected task\n"
            f"Providers: {provider_text}\n"
            f"Provider Details: {'; '.join(provider_lines) if provider_lines else 'none'}\n"
            f"Assignments: {', '.join(assignment_lines) if assignment_lines else 'none'}\n"
            f"Agents: {agent_text}\n"
            f"Workflows: {workflow_text}"
        )

    def _render_team_body(self) -> str:
        agent_ids = self.runtime.list_agent_ids()
        if not agent_ids:
            return "No agents defined."
        lines = [
            f"{'> ' if agent_id == self.selected_agent_id else '  '}{agent_id} [{self.runtime.agent_status_summary(agent_id)}]"
            for agent_id in agent_ids
        ]
        definition = self.runtime.registry.agent_definitions.get(self.selected_agent_id)
        if definition is None:
            return "\n".join(lines)

        role = str(definition.metadata.get("role", self.selected_agent_id))
        tools = definition.metadata.get("tools", [])
        if not isinstance(tools, list):
            tools = []
        identity = definition.sections.get("Identity", "")

        lines.extend(
            [
                "",
                f"Role: {role}",
                f"Status: {self.runtime.agent_status_summary(self.selected_agent_id)}",
                f"Tools: {', '.join(str(tool) for tool in tools) if tools else 'none'}",
            ]
        )
        if identity:
            lines.extend(["", identity])
        return "\n".join(lines)

    def _render_workflows_body(self) -> str:
        workflow_ids = self.runtime.list_workflow_ids()
        if not workflow_ids:
            return "No workflows defined."
        lines = [
            f"{'> ' if workflow_id == self.selected_workflow_id else '  '}{workflow_id}"
            for workflow_id in workflow_ids
        ]
        definition = self.runtime.registry.workflow_definitions.get(self.selected_workflow_id)
        if definition is None:
            return "\n".join(lines)

        orchestration = str(definition.metadata.get("orchestration", "unknown"))
        step_text = "none"
        workflow_step_groups = definition.metadata.get("step_groups")
        workflow_steps = definition.metadata.get("steps", [])
        if isinstance(workflow_step_groups, list) and workflow_step_groups:
            rendered_groups: list[str] = []
            for group in workflow_step_groups:
                if isinstance(group, list):
                    rendered_groups.append(" + ".join(str(step) for step in group))
                else:
                    rendered_groups.append(str(group))
            step_text = " -> ".join(rendered_groups)
        elif isinstance(workflow_steps, list) and workflow_steps:
            step_text = " -> ".join(str(step) for step in workflow_steps)
        purpose = definition.sections.get("Purpose", "")
        exit_conditions = definition.sections.get("Exit Conditions", "")

        lines.extend(["", f"Orchestration: {orchestration}", f"Steps: {step_text}"])
        if purpose:
            lines.extend(["", purpose])
        if exit_conditions:
            lines.extend(["", f"Exit: {exit_conditions}"])
        return "\n".join(lines)

    def _render_workflow_runs_body(self) -> str:
        runs = self.runtime.list_workflow_runs()
        if not runs:
            return "No workflow runs yet."
        lines = [
            f"{'> ' if run.id == self.selected_workflow_run_id else '  '}{run.id} [{run.state}] step={run.current_step_index} {run.workflow_id}"
            for run in runs[-8:]
        ]
        if self.selected_workflow_run_id is None:
            return "\n".join(lines)

        run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
        if run_view is None:
            return "\n".join(lines)

        lines.append("")
        if run_view.root_task is not None:
            lines.append(
                f"Root: {run_view.root_task.id} [{run_view.root_task.state}] {run_view.root_task.title}"
            )
        lines.append(f"Steps: {run_view.workflow_run.current_step_index}/{len(run_view.steps)}")
        if run_view.workflow_run.last_thread_id is not None:
            lines.append(f"Last thread: {run_view.workflow_run.last_thread_id}")
        if run_view.workflow_run.current_step_index < len(run_view.steps):
            next_step = run_view.steps[run_view.workflow_run.current_step_index]
            next_agent = " + ".join(thread.assigned_agent_id or "unknown" for thread in next_step.threads) if next_step.threads else "unknown"
            lines.append(f"Next agent: {next_agent}")
        else:
            lines.append("Next agent: none")
        return "\n".join(lines)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message_body = event.value.strip()
        if not message_body:
            event.input.value = ""
            return

        event.input.disabled = True
        try:
            created_at = self._next_timestamp()
            selected_thread = self.runtime.get_thread(self.selected_thread_id)
            if selected_thread is not None and selected_thread.id != self.runtime.main_thread_id:
                await self.runtime.send_message_to_agent_thread(
                    thread_id=selected_thread.id,
                    body=message_body,
                    created_at=created_at,
                )
            else:
                await self.runtime.send_user_message_to_orchestrator(
                    body=message_body,
                    created_at=created_at,
                )
            self._refresh_panels()
            event.input.value = ""
        finally:
            event.input.disabled = False

    def action_next_thread(self) -> None:
        self._cycle_thread(1)

    def action_previous_thread(self) -> None:
        self._cycle_thread(-1)

    def action_next_activity_event(self) -> None:
        self._cycle_activity_event(1)

    def action_previous_activity_event(self) -> None:
        self._cycle_activity_event(-1)

    def action_next_task(self) -> None:
        self._cycle_task(1)

    def action_previous_task(self) -> None:
        self._cycle_task(-1)

    def action_next_command_run(self) -> None:
        self._cycle_command_run(1)

    def action_previous_command_run(self) -> None:
        self._cycle_command_run(-1)

    def action_next_agent(self) -> None:
        self._cycle_agent(1)

    def action_previous_agent(self) -> None:
        self._cycle_agent(-1)

    def action_next_memory_fact(self) -> None:
        self._cycle_memory_fact(1)

    def action_previous_memory_fact(self) -> None:
        self._cycle_memory_fact(-1)

    def action_open_selected_agent_thread(self) -> None:
        created_at = self._next_timestamp()
        task = self.runtime.create_task(
            task_id=f"task-{uuid4().hex[:8]}",
            title=f"Agent thread: {self.selected_agent_id}",
            state="in_progress",
            created_at=created_at,
        )
        thread = self.runtime.create_agent_thread(
            agent_id=self.selected_agent_id,
            created_at=created_at + 1,
            parent_task_id=task.id,
        )
        self.selected_thread_id = thread.id
        self._refresh_panels()

    def action_edit_selected_agent_definition(self) -> None:
        self._open_agent_definition_editor(self.selected_agent_id)

    def action_edit_selected_task_whiteboard(self) -> None:
        task_id = self.selected_task_id
        if task_id is None:
            return
        initial_text = self.runtime.read_task_whiteboard_text(task_id)
        self.push_screen(
            DefinitionEditorScreen(
                title=f"Edit Task Whiteboard: {task_id}",
                initial_text=initial_text,
                on_save=lambda text: self._save_task_whiteboard(task_id, text),
            )
        )

    def action_next_workflow(self) -> None:
        self._cycle_workflow(1)

    def action_previous_workflow(self) -> None:
        self._cycle_workflow(-1)

    def action_edit_selected_workflow_definition(self) -> None:
        self._open_workflow_definition_editor(self.selected_workflow_id)

    async def action_start_selected_workflow(self) -> None:
        created_at = self._next_timestamp()
        workflow_run, threads = self.runtime.start_workflow_run(
            workflow_id=self.selected_workflow_id,
            created_at=created_at,
        )
        self.selected_workflow_run_id = workflow_run.id
        if threads:
            _, thread, _ = await self.runtime.advance_workflow_run(
                workflow_run_id=workflow_run.id,
                created_at=self._next_timestamp(),
            )
            if thread is not None:
                self.selected_thread_id = thread.id
        self._refresh_panels()

    async def action_advance_selected_workflow_run(self) -> None:
        if self.selected_workflow_run_id is None:
            return
        workflow_run, thread, _ = await self.runtime.advance_workflow_run(
            workflow_run_id=self.selected_workflow_run_id,
            created_at=self._next_timestamp(),
        )
        self.selected_workflow_run_id = workflow_run.id
        if thread is not None:
            self.selected_thread_id = thread.id
        self._refresh_panels()

    def action_request_fix_cycle_for_selected_workflow_run(self) -> None:
        if self.selected_workflow_run_id is None:
            return
        workflow_run, threads = self.runtime.request_workflow_fix_cycle(
            workflow_run_id=self.selected_workflow_run_id,
            created_at=self._next_timestamp(),
        )
        self.selected_workflow_run_id = workflow_run.id
        if threads:
            self.selected_thread_id = threads[0].id
        self._refresh_panels()

    def action_next_workflow_run(self) -> None:
        self._cycle_workflow_run(1)

    def action_previous_workflow_run(self) -> None:
        self._cycle_workflow_run(-1)

    def action_clear_workflow_run_focus(self) -> None:
        if self.selected_workflow_run_id is None:
            return
        self.selected_workflow_run_id = None
        self._refresh_panels()

    def action_next_approval(self) -> None:
        self._cycle_approval(1)

    def action_previous_approval(self) -> None:
        self._cycle_approval(-1)

    def action_next_artifact(self) -> None:
        self._cycle_artifact(1)

    def action_previous_artifact(self) -> None:
        self._cycle_artifact(-1)

    def action_approve_selected_approval(self) -> None:
        if self.selected_approval_id is None:
            return
        self.runtime.resolve_approval(
            approval_id=self.selected_approval_id,
            status="approved",
            created_at=self._next_timestamp(),
        )
        self._normalize_selected_approval()
        self._refresh_panels()

    def action_reject_selected_approval(self) -> None:
        if self.selected_approval_id is None:
            return
        self.runtime.resolve_approval(
            approval_id=self.selected_approval_id,
            status="rejected",
            created_at=self._next_timestamp(),
        )
        self._normalize_selected_approval()
        self._refresh_panels()

    def action_edit_orchestrator_definition(self) -> None:
        self._open_agent_definition_editor("orchestrator")

    def action_run_workspace_command(self) -> None:
        self.push_screen(
            DefinitionEditorScreen(
                title="Run Workspace Command",
                initial_text="",
                on_save=self._run_workspace_command_from_editor,
                language="bash",
            )
        )

    def action_edit_global_config(self) -> None:
        initial_text = self.runtime.read_global_config_text()
        self.push_screen(
            DefinitionEditorScreen(
                title="Edit Global Config",
                initial_text=initial_text,
                on_save=self._save_global_config,
                language="json",
            )
        )

    def _cycle_thread(self, direction: int) -> None:
        threads = self._visible_threads()
        if not threads:
            return

        thread_ids = [thread.id for thread in threads]
        try:
            current_index = thread_ids.index(self.selected_thread_id)
        except ValueError:
            current_index = 0

        self.selected_thread_id = thread_ids[(current_index + direction) % len(thread_ids)]
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        self.query_one("#selected-thread", Panel).set_body(self._render_selected_thread_body())

    def _cycle_task(self, direction: int) -> None:
        tasks = self._visible_tasks()
        if not tasks:
            self.selected_task_id = None
            self.query_one("#tasks", Panel).set_body(self._render_tasks_body())
            return

        task_ids = [task.id for task in tasks]
        if self.selected_task_id is None:
            current_index = 0
        else:
            try:
                current_index = task_ids.index(self.selected_task_id)
            except ValueError:
                current_index = 0

        self.selected_task_id = task_ids[(current_index + direction) % len(task_ids)]
        self.query_one("#tasks", Panel).set_body(self._render_tasks_body())

    def _cycle_activity_event(self, direction: int) -> None:
        events = self._visible_events()
        if not events:
            self.selected_event_id = None
            self.query_one("#activity", Panel).set_body(self._render_activity_body())
            return

        event_ids = [event.id for event in events]
        if self.selected_event_id is None:
            current_index = len(event_ids) - 1
        else:
            try:
                current_index = event_ids.index(self.selected_event_id)
            except ValueError:
                current_index = len(event_ids) - 1

        self.selected_event_id = event_ids[(current_index + direction) % len(event_ids)]
        self.query_one("#activity", Panel).set_body(self._render_activity_body())

    @staticmethod
    def _thread_label(thread) -> str:
        if getattr(thread, "assigned_agent_id", None):
            return f"{thread.kind}:{thread.assigned_agent_id}"
        return thread.kind

    def _cycle_agent(self, direction: int) -> None:
        agent_ids = self.runtime.list_agent_ids()
        if not agent_ids:
            return

        try:
            current_index = agent_ids.index(self.selected_agent_id)
        except ValueError:
            current_index = 0

        self.selected_agent_id = agent_ids[(current_index + direction) % len(agent_ids)]
        self.query_one("#team", Panel).set_body(self._render_team_body())

    def _cycle_workflow(self, direction: int) -> None:
        workflow_ids = self.runtime.list_workflow_ids()
        if not workflow_ids:
            return

        try:
            current_index = workflow_ids.index(self.selected_workflow_id)
        except ValueError:
            current_index = 0

        self.selected_workflow_id = workflow_ids[(current_index + direction) % len(workflow_ids)]
        self.query_one("#workflows", Panel).set_body(self._render_workflows_body())

    def _cycle_workflow_run(self, direction: int) -> None:
        runs = self.runtime.list_workflow_runs()
        if not runs:
            return

        run_ids = [run.id for run in runs]
        if self.selected_workflow_run_id is None:
            current_index = 0
        else:
            try:
                current_index = run_ids.index(self.selected_workflow_run_id)
            except ValueError:
                current_index = 0

        self.selected_workflow_run_id = run_ids[(current_index + direction) % len(run_ids)]
        preferred_thread_id = self.runtime.preferred_thread_id_for_workflow_run(self.selected_workflow_run_id)
        if preferred_thread_id is not None:
            self.selected_thread_id = preferred_thread_id
        self._normalize_selected_task()
        self._normalize_selected_event()
        self._normalize_selected_approval()
        self._normalize_selected_artifact()
        self._normalize_selected_command_run()
        self._normalize_selected_thread()
        self.query_one("#tasks", Panel).set_body(self._render_tasks_body())
        self.query_one("#workflow-runs", Panel).set_body(self._render_workflow_runs_body())
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        self.query_one("#selected-thread", Panel).set_body(self._render_selected_thread_body())
        self.query_one("#commands", Panel).set_body(self._render_commands_body())
        self.query_one("#artifacts", Panel).set_body(self._render_artifacts_body())
        self.query_one("#activity", Panel).set_body(self._render_activity_body())
        self.query_one("#approvals", Panel).set_body(self._render_approvals_body())

    def _cycle_approval(self, direction: int) -> None:
        approvals = self._visible_approvals()
        if not approvals:
            self.selected_approval_id = None
            self.query_one("#approvals", Panel).set_body(self._render_approvals_body())
            return

        approval_ids = [approval.id for approval in approvals]
        if self.selected_approval_id is None:
            current_index = 0
        else:
            try:
                current_index = approval_ids.index(self.selected_approval_id)
            except ValueError:
                current_index = 0

        self.selected_approval_id = approval_ids[(current_index + direction) % len(approval_ids)]
        self.query_one("#approvals", Panel).set_body(self._render_approvals_body())

    def _cycle_artifact(self, direction: int) -> None:
        artifacts = self._visible_artifacts()
        if not artifacts:
            self.selected_artifact_id = None
            self.query_one("#artifacts", Panel).set_body(self._render_artifacts_body())
            return

        artifact_ids = [artifact.id for artifact in artifacts]
        if self.selected_artifact_id is None:
            current_index = 0
        else:
            try:
                current_index = artifact_ids.index(self.selected_artifact_id)
            except ValueError:
                current_index = 0

        self.selected_artifact_id = artifact_ids[(current_index + direction) % len(artifact_ids)]
        self.query_one("#artifacts", Panel).set_body(self._render_artifacts_body())

    def _cycle_memory_fact(self, direction: int) -> None:
        facts = self.runtime.list_memory_facts()
        if not facts:
            self.selected_memory_fact_id = None
            self.query_one("#memory", Panel).set_body(self._render_memory_body())
            return

        fact_ids = [fact.id for fact in facts]
        if self.selected_memory_fact_id is None:
            current_index = 0
        else:
            try:
                current_index = fact_ids.index(self.selected_memory_fact_id)
            except ValueError:
                current_index = 0

        self.selected_memory_fact_id = fact_ids[(current_index + direction) % len(fact_ids)]
        self.query_one("#memory", Panel).set_body(self._render_memory_body())

    def _cycle_command_run(self, direction: int) -> None:
        command_runs = self._visible_command_runs()
        if not command_runs:
            self.selected_command_run_id = None
            self.query_one("#commands", Panel).set_body(self._render_commands_body())
            return

        command_run_ids = [command_run.id for command_run in command_runs]
        if self.selected_command_run_id is None:
            current_index = 0
        else:
            try:
                current_index = command_run_ids.index(self.selected_command_run_id)
            except ValueError:
                current_index = 0

        self.selected_command_run_id = command_run_ids[(current_index + direction) % len(command_run_ids)]
        self.query_one("#commands", Panel).set_body(self._render_commands_body())

    def _open_agent_definition_editor(self, agent_id: str) -> None:
        initial_text = self.runtime.read_agent_definition_text(agent_id)
        self.push_screen(
            DefinitionEditorScreen(
                title=f"Edit Agent Definition: {agent_id}",
                initial_text=initial_text,
                on_save=lambda text: self._save_agent_definition(agent_id, text),
            )
        )

    def _open_workflow_definition_editor(self, workflow_id: str) -> None:
        initial_text = self.runtime.read_workflow_definition_text(workflow_id)
        self.push_screen(
            DefinitionEditorScreen(
                title=f"Edit Workflow Definition: {workflow_id}",
                initial_text=initial_text,
                on_save=lambda text: self._save_workflow_definition(workflow_id, text),
            )
        )

    def _save_orchestrator_definition(self, text: str) -> None:
        self._save_agent_definition("orchestrator", text)

    def _save_agent_definition(self, agent_id: str, text: str) -> None:
        self.runtime.save_agent_definition_text(
            agent_id=agent_id,
            text=text,
            created_at=self._next_timestamp(),
        )
        self.selected_agent_id = agent_id
        self._refresh_panels()

    def _save_workflow_definition(self, workflow_id: str, text: str) -> None:
        self.runtime.save_workflow_definition_text(
            workflow_id=workflow_id,
            text=text,
            created_at=self._next_timestamp(),
        )
        self.selected_workflow_id = workflow_id
        self._refresh_panels()

    def _save_global_config(self, text: str) -> None:
        self.runtime.save_global_config_text(
            text=text,
            created_at=self._next_timestamp(),
        )
        self._refresh_panels()

    def _save_task_whiteboard(self, task_id: str, text: str) -> None:
        self.runtime.save_task_whiteboard_text(
            task_id=task_id,
            text=text,
            created_at=self._next_timestamp(),
        )
        self._refresh_panels()

    def _run_workspace_command_from_editor(self, text: str) -> None:
        command = text.strip()
        if not command:
            raise ValueError("Command cannot be empty.")

        thread = self.runtime.get_thread(self.selected_thread_id)
        result = self.runtime.run_workspace_command(
            command,
            created_at=self._next_timestamp(),
            thread_id=self.selected_thread_id,
            task_id=thread.parent_task_id if thread is not None else self.selected_task_id,
            agent_id="user",
        )
        command_run_id = result.get("command_run_id")
        if isinstance(command_run_id, str):
            self.selected_command_run_id = command_run_id
        approval_id = result.get("approval_id")
        if isinstance(approval_id, str):
            self.selected_approval_id = approval_id
        self._refresh_panels()

    def _refresh_panels(self) -> None:
        self._normalize_selected_approval()
        self._normalize_selected_artifact()
        self._normalize_selected_memory_fact()
        self._normalize_selected_task()
        self._normalize_selected_event()
        self._normalize_selected_command_run()
        self._normalize_selected_thread()
        self.query_one("#tasks", Panel).set_body(self._render_tasks_body())
        self.query_one("#workflows", Panel).set_body(self._render_workflows_body())
        self.query_one("#workflow-runs", Panel).set_body(self._render_workflow_runs_body())
        self.query_one("#threads", Panel).set_body(self._render_threads_body())
        self.query_one("#team", Panel).set_body(self._render_team_body())
        self.query_one("#activity", Panel).set_body(self._render_activity_body())
        self.query_one("#main-chat", Panel).set_body(self._render_main_chat_body())
        self.query_one("#selected-thread", Panel).set_body(self._render_selected_thread_body())
        self.query_one("#commands", Panel).set_body(self._render_commands_body())
        self.query_one("#artifacts", Panel).set_body(self._render_artifacts_body())
        self.query_one("#approvals", Panel).set_body(self._render_approvals_body())
        self.query_one("#memory", Panel).set_body(self._render_memory_body())
        self.query_one("#settings", Panel).set_body(self._render_settings_body())

    def _next_timestamp(self) -> int:
        self._time_cursor += 1
        return self._time_cursor

    def _normalize_selected_approval(self) -> None:
        approval_ids = [approval.id for approval in self._visible_approvals()]
        if not approval_ids:
            self.selected_approval_id = None
            return
        if self.selected_approval_id not in approval_ids:
            self.selected_approval_id = approval_ids[0]

    def _normalize_selected_artifact(self) -> None:
        artifact_ids = [artifact.id for artifact in self._visible_artifacts()]
        if not artifact_ids:
            self.selected_artifact_id = None
            return
        if self.selected_artifact_id not in artifact_ids:
            self.selected_artifact_id = artifact_ids[0]

    def _normalize_selected_memory_fact(self) -> None:
        fact_ids = [fact.id for fact in self.runtime.list_memory_facts()]
        if not fact_ids:
            self.selected_memory_fact_id = None
            return
        if self.selected_memory_fact_id not in fact_ids:
            self.selected_memory_fact_id = fact_ids[0]

    def _normalize_selected_task(self) -> None:
        task_ids = [task.id for task in self._visible_tasks()]
        if not task_ids:
            self.selected_task_id = None
            return
        if self.selected_task_id not in task_ids:
            self.selected_task_id = task_ids[0]

    def _normalize_selected_event(self) -> None:
        event_ids = [event.id for event in self._visible_events()]
        if not event_ids:
            self.selected_event_id = None
            return
        if self.selected_event_id not in event_ids:
            self.selected_event_id = event_ids[-1]

    def _normalize_selected_command_run(self) -> None:
        command_run_ids = [command_run.id for command_run in self._visible_command_runs()]
        if not command_run_ids:
            self.selected_command_run_id = None
            return
        if self.selected_command_run_id not in command_run_ids:
            self.selected_command_run_id = command_run_ids[-1]

    def _normalize_selected_thread(self) -> None:
        thread_ids = [thread.id for thread in self._visible_threads()]
        if not thread_ids:
            self.selected_thread_id = self.runtime.main_thread_id
            return
        if self.selected_thread_id not in thread_ids:
            self.selected_thread_id = thread_ids[0]

    def _visible_threads(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                return self.runtime.list_threads_for_workflow_run(self.selected_workflow_run_id)
        return self.runtime.list_threads()

    def _visible_events(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                return self.runtime.list_events_for_workflow_run(self.selected_workflow_run_id)[-8:]
        return self.runtime.list_events()[-8:]

    def _visible_tasks(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is None:
                return []
            tasks = []
            if run_view.root_task is not None:
                tasks.append(run_view.root_task)
            tasks.extend(step.task for step in run_view.steps)
            return tasks
        return self.runtime.list_tasks()

    def _visible_approvals(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                return self.runtime.list_pending_approvals_for_workflow_run(self.selected_workflow_run_id)
        return self.runtime.list_pending_approvals()

    def _visible_command_runs(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                return self.runtime.list_command_runs_for_workflow_run(self.selected_workflow_run_id)
        return self.runtime.list_command_runs()

    def _approval_lines(self, approvals) -> list[str]:
        return [
            f"{'> ' if approval.id == self.selected_approval_id else '  '}{approval.id} [{approval.risk_class}] {approval.action}"
            for approval in approvals
        ]

    def _approval_preview_lines(self, approvals) -> list[str]:
        selected = None
        for approval in approvals:
            if approval.id == self.selected_approval_id:
                selected = approval
                break
        if selected is None:
            return []
        lines = [
            "",
            "Preview:",
            f"Requester: {selected.requester}",
            f"Reason: {selected.reason}",
        ]
        payload = self.runtime.read_approval_payload(selected.id)
        if payload is not None and isinstance(payload.get("command"), str):
            lines.append(f"Command: {payload['command']}")
        if payload is not None and isinstance(payload.get("path"), str):
            lines.append(f"Path: {payload['path']}")
        if selected.task_id is not None:
            lines.append(f"Task: {selected.task_id}")
        if selected.thread_id is not None:
            lines.append(f"Thread: {selected.thread_id}")
        return lines

    def _visible_artifacts(self):
        if self.selected_workflow_run_id is not None:
            run_view = self.runtime.describe_workflow_run(self.selected_workflow_run_id)
            if run_view is not None:
                return self.runtime.list_artifacts_for_workflow_run(self.selected_workflow_run_id)
        return self.runtime.list_artifacts()

    def _artifact_lines(self, artifacts) -> list[str]:
        return [
            f"{'> ' if artifact.id == self.selected_artifact_id else '  '}{artifact.id} [{artifact.kind}] {artifact.title}"
            for artifact in artifacts[-8:]
        ]

    def _artifact_preview_lines(self) -> list[str]:
        if self.selected_artifact_id is None:
            return []
        body = self.runtime.read_artifact_body(self.selected_artifact_id).rstrip("\n")
        if not body:
            return ["", "Preview:", "(empty)"]
        return ["", "Preview:", body]

    def _command_lines(self, command_runs) -> list[str]:
        return [
            (
                f"{'> ' if command_run.id == self.selected_command_run_id else '  '}"
                f"{command_run.id} [{command_run.status}/{command_run.exit_code}] {command_run.command}"
            )
            for command_run in command_runs[-8:]
        ]

    def _command_preview_lines(self) -> list[str]:
        if self.selected_command_run_id is None:
            return []

        selected = None
        for command_run in self._visible_command_runs():
            if command_run.id == self.selected_command_run_id:
                selected = command_run
                break
        if selected is None:
            return []

        lines = [
            "",
            "Preview:",
            f"Cwd: {selected.cwd}",
            f"Exit: {selected.exit_code}",
        ]
        if selected.agent_id is not None:
            lines.append(f"Agent: {selected.agent_id}")
        if selected.thread_id is not None:
            lines.append(f"Thread: {selected.thread_id}")
        if selected.task_id is not None:
            lines.append(f"Task: {selected.task_id}")
        body = self.runtime.read_command_output(selected.id).rstrip("\n")
        lines.extend(["", body if body else "(empty)"])
        return lines

    def _memory_lines(self, facts) -> list[str]:
        return [
            f"{'> ' if fact.id == self.selected_memory_fact_id else '  '}{fact.id} [{fact.kind}] {fact.content}"
            for fact in facts[-8:]
        ]

    def _memory_preview_lines(self, facts) -> list[str]:
        selected = None
        for fact in facts:
            if fact.id == self.selected_memory_fact_id:
                selected = fact
                break
        if selected is None:
            return []
        return [
            "",
            "Preview:",
            f"Scope: {selected.scope}",
            f"Kind: {selected.kind}",
            f"Source: {selected.source or 'n/a'}",
            f"Confidence: {selected.confidence if selected.confidence is not None else 'n/a'}",
            f"Tags: {', '.join(selected.tags) if selected.tags else 'none'}",
            selected.content,
        ]

    def _task_whiteboard_lines(self) -> list[str]:
        task_id = self.selected_task_id
        if task_id is None:
            return []
        whiteboard = self.runtime.get_task_whiteboard(task_id)
        if whiteboard is None:
            return []
        lines = [
            f"Whiteboard: {whiteboard.task_id}",
            f"Title: {whiteboard.title}",
        ]
        goal = whiteboard.sections.get("Goal", "").strip()
        decisions = whiteboard.sections.get("Decisions", "").strip()
        acceptance = whiteboard.sections.get("Acceptance Criteria", "").strip()
        if goal:
            lines.extend(["", "Goal:", goal])
        if decisions:
            lines.extend(["", "Decisions:", decisions])
        if acceptance:
            lines.extend(["", "Acceptance Criteria:", acceptance])
        return lines

    def _task_preview_lines(self, selected_task) -> list[str]:
        if selected_task is None:
            return []
        related_threads = [
            thread.id
            for thread in self.runtime.list_threads()
            if thread.parent_task_id == selected_task.id
        ]
        lines = [
            "",
            "Preview:",
            f"State: {selected_task.state}",
        ]
        if selected_task.parent_task_id is not None:
            lines.append(f"Parent: {selected_task.parent_task_id}")
        lines.append(selected_task.title)
        if related_threads:
            lines.append(f"Threads: {', '.join(related_threads)}")
        return lines

    def _event_lines(self, events) -> list[str]:
        return [
            f"{'> ' if event.id == self.selected_event_id else '  '}{event.kind}: {event.summary}"
            for event in events
        ]

    def _event_preview_lines(self, events) -> list[str]:
        selected = None
        for event in events:
            if event.id == self.selected_event_id:
                selected = event
                break
        if selected is None:
            return []
        lines = [
            "",
            "Preview:",
            f"Kind: {selected.kind}",
            f"Summary: {selected.summary}",
        ]
        if selected.task_id is not None:
            lines.append(f"Task: {selected.task_id}")
        if selected.thread_id is not None:
            lines.append(f"Thread: {selected.thread_id}")
        return lines
