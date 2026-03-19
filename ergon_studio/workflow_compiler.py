from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import FunctionExecutor, WorkflowBuilder, WorkflowViz

from ergon_studio.definitions import DefinitionDocument


@dataclass(frozen=True)
class CompiledWorkflow:
    definition_id: str
    workflow: object
    step_groups: tuple[tuple[str, ...], ...]

    def to_mermaid(self) -> str:
        return WorkflowViz(self.workflow).to_mermaid()

    def to_digraph(self) -> str:
        return WorkflowViz(self.workflow).to_digraph()


def compile_workflow_definition(definition: DefinitionDocument) -> CompiledWorkflow:
    step_groups = _workflow_step_groups(definition)
    orchestration = str(definition.metadata.get("orchestration", "sequential"))
    if orchestration == "group_chat":
        workflow = _compile_group_chat_workflow(definition, step_groups)
        return CompiledWorkflow(
            definition_id=definition.id,
            workflow=workflow,
            step_groups=step_groups,
        )
    if orchestration == "magentic":
        workflow = _compile_star_workflow(definition, step_groups, coordinator_id="magentic_manager")
        return CompiledWorkflow(
            definition_id=definition.id,
            workflow=workflow,
            step_groups=step_groups,
        )
    if orchestration == "handoff":
        workflow = _compile_star_workflow(definition, step_groups, coordinator_id="handoff_router")
        return CompiledWorkflow(
            definition_id=definition.id,
            workflow=workflow,
            step_groups=step_groups,
        )
    root = _executor("workflow-start")
    builder = WorkflowBuilder(
        name=f"workflow-{definition.id}",
        description=str(definition.metadata.get("name", definition.id)),
        start_executor=root,
    )

    previous_group = (root,)
    for group_index, group in enumerate(step_groups):
        current_group = tuple(
            _executor(f"{agent_id}-{group_index + 1}-{agent_index + 1}")
            for agent_index, agent_id in enumerate(group)
        )
        _connect_group(builder, previous_group, current_group)
        previous_group = current_group

    workflow = builder.build()
    return CompiledWorkflow(
        definition_id=definition.id,
        workflow=workflow,
        step_groups=step_groups,
    )


def _compile_group_chat_workflow(
    definition: DefinitionDocument,
    step_groups: tuple[tuple[str, ...], ...],
):
    return _compile_star_workflow(definition, step_groups, coordinator_id="group_chat_orchestrator")


def _compile_star_workflow(
    definition: DefinitionDocument,
    step_groups: tuple[tuple[str, ...], ...],
    *,
    coordinator_id: str,
):
    participants = tuple(agent_id for group in step_groups for agent_id in group)
    if not participants:
        raise ValueError(f"orchestrated workflow '{definition.id}' must declare participants")
    orchestrator = _executor(coordinator_id)
    participant_executors = tuple(_executor(agent_id) for agent_id in participants)
    merge = _merge_executor(participant_executors, (_executor(f"{coordinator_id}_summary"),))
    builder = WorkflowBuilder(
        name=f"workflow-{definition.id}",
        description=str(definition.metadata.get("name", definition.id)),
        start_executor=orchestrator,
    )
    builder.add_fan_out_edges(orchestrator, list(participant_executors))
    builder.add_fan_in_edges(list(participant_executors), merge)
    return builder.build()


def _connect_group(builder: WorkflowBuilder, previous_group: tuple[object, ...], current_group: tuple[object, ...]) -> None:
    if len(previous_group) > 1:
        merge = _merge_executor(previous_group, current_group)
        builder.add_fan_in_edges(list(previous_group), merge)
        previous_group = (merge,)
    if len(previous_group) == 1 and len(current_group) == 1:
        builder.add_edge(previous_group[0], current_group[0])
        return
    if len(previous_group) == 1:
        builder.add_fan_out_edges(previous_group[0], list(current_group))
        return
    raise ValueError("workflow compiler reached an invalid connection state")


def _executor(executor_id: str) -> FunctionExecutor:
    return FunctionExecutor(
        _passthrough_message,
        id=executor_id,
        input=object,
        output=object,
        workflow_output=object,
    )


def _passthrough_message(payload: object) -> object:
    return payload


def _merge_executor(previous_group: tuple[object, ...], current_group: tuple[object, ...]) -> FunctionExecutor:
    previous_ids = "-".join(_executor_id(executor) for executor in previous_group)
    current_ids = "-".join(_executor_id(executor) for executor in current_group)
    return FunctionExecutor(
        _merge_messages,
        id=f"merge-{previous_ids}-to-{current_ids}",
        input=list[object],
        output=object,
        workflow_output=object,
    )


def _merge_messages(payload: list[object]) -> object:
    return payload


def _executor_id(executor: object) -> str:
    return str(getattr(executor, "id"))


def _workflow_step_groups(definition: DefinitionDocument) -> tuple[tuple[str, ...], ...]:
    configured_step_groups = definition.metadata.get("step_groups")
    if configured_step_groups is not None:
        if not isinstance(configured_step_groups, list):
            raise ValueError(f"workflow '{definition.id}' step_groups must be a list")
        return tuple(_validate_group(definition.id, group) for group in configured_step_groups)

    configured_steps = definition.metadata.get("steps", [])
    if not isinstance(configured_steps, list):
        raise ValueError(f"workflow '{definition.id}' steps must be a list")
    return tuple((step,) for step in _validate_group(definition.id, configured_steps)) if configured_steps else ()


def _validate_group(workflow_id: str, group: object) -> tuple[str, ...]:
    if isinstance(group, str):
        if not group:
            raise ValueError(f"workflow '{workflow_id}' step entries must be non-empty strings")
        return (group,)
    if not isinstance(group, list) or not group:
        raise ValueError(f"workflow '{workflow_id}' step groups must be non-empty lists")
    validated: list[str] = []
    for item in group:
        if not isinstance(item, str) or not item:
            raise ValueError(f"workflow '{workflow_id}' step entries must be non-empty strings")
        validated.append(item)
    return tuple(validated)
