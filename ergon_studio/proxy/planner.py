from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workflow_policy import (
    acceptance_mode_for_metadata,
    delivery_candidate_for_metadata,
    selection_hints_for_metadata,
)


@dataclass(frozen=True)
class ProxyTurnPlan:
    mode: str
    workflow_id: str | None = None
    agent_id: str | None = None
    request: str | None = None
    goal: str | None = None
    deliverable_expected: bool = False


def build_turn_planner_instructions(registry: RuntimeRegistry) -> str:
    workflow_lines = []
    for workflow_id, definition in sorted(registry.workflow_definitions.items()):
        hints = ", ".join(selection_hints_for_metadata(definition.metadata)) or "none"
        orchestration = definition.metadata.get("orchestration", "unknown")
        delivery_candidate = delivery_candidate_for_metadata(definition.metadata)
        acceptance_mode = acceptance_mode_for_metadata(definition.metadata)
        workflow_lines.append(
            f"- {workflow_id}: orchestration={orchestration} "
            f"delivery_candidate={delivery_candidate} "
            f"acceptance={acceptance_mode} "
            f"selection_hints={hints}"
        )
    specialist_lines = []
    for agent_id, definition in sorted(registry.agent_definitions.items()):
        if agent_id == "orchestrator":
            continue
        specialist_lines.append(
            f"- {agent_id}: role={definition.metadata.get('role', agent_id)}"
        )

    return "\n".join(
        [
            (
                "You are the internal execution planner for the lead developer "
                "in an AI software firm."
            ),
            (
                "The user is the product manager. The orchestrator is the lead "
                "developer."
            ),
            (
                "Choose the single best immediate move that helps the lead "
                "developer move the work forward."
            ),
            "Output JSON only.",
            "",
            "Allowed modes:",
            '- "act": the lead developer handles the next move directly.',
            '- "delegate": assign one specialist a focused task.',
            '- "workflow": invoke a named playbook involving one or more specialists.',
            '- "finish": deliver the current result back to the product manager.',
            "",
            "Rules:",
            (
                "- Think like a pragmatic lead developer, not a classifier."
            ),
            (
                "- Prefer act for discussion, clarification, planning with the product "
                "manager, and small direct actions."
            ),
            (
                "- Prefer finish when the work is materially done and the lead "
                "developer should hand back a concrete result."
            ),
            (
                "- Prefer delegate for narrow, well-bounded specialist work."
            ),
            (
                "- Prefer workflow when the work would benefit from multiple steps, "
                "comparison, review, repair, debate, or adaptive staffing."
            ),
            (
                "- Workflows are playbooks, not rigid laws. Choose one when it is a "
                "good tactic, not because keywords happen to match."
            ),
            (
                "- Preserve the full delivery goal when the product manager expects "
                "implemented output."
            ),
            (
                "- Do not rely on keyword matching. Infer intent from the actual "
                "request and transcript."
            ),
            "",
            "Available workflows:",
            *workflow_lines,
            "",
            "Available specialists:",
            *specialist_lines,
            "",
            "Required JSON shape:",
            '{"mode":"workflow|delegate|act|finish","workflow_id":null,"agent_id":null,"request":"","goal":"","deliverable_expected":false}',
        ]
    )


def build_turn_planner_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    active_workflow_id: str | None = None,
) -> str:
    lines = [
        "Conversation transcript:",
        *_transcript_lines(request.messages),
        "",
        "Latest user request:",
        request.latest_user_text() or "(none)",
    ]
    if goal:
        lines.extend(
            [
                "",
                "Current delivery goal:",
                goal,
            ]
        )
    if current_brief:
        lines.extend(
            [
                "",
                "Current brief:",
                current_brief,
            ]
        )
    if worklog:
        lines.extend(
            [
                "",
                "Team work so far:",
                *worklog[-12:],
            ]
        )
    if active_workflow_id:
        lines.extend(
            [
                "",
                "Playbook currently in progress:",
                active_workflow_id,
            ]
        )
    return "\n".join(lines).strip()


def parse_turn_plan(raw: str, *, registry: RuntimeRegistry) -> ProxyTurnPlan:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid planner json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("planner output must be a JSON object")

    mode = str(payload.get("mode", "act")).strip().lower()
    if mode not in {"act", "delegate", "workflow", "finish"}:
        mode = "act"
    workflow_id = resolve_workflow_reference(
        registry, _optional_text(payload.get("workflow_id"))
    )
    agent_id = _optional_text(payload.get("agent_id"))
    if agent_id is not None and agent_id not in registry.agent_definitions:
        agent_id = None
    return ProxyTurnPlan(
        mode=mode,
        workflow_id=workflow_id,
        agent_id=agent_id,
        request=_optional_text(payload.get("request")),
        goal=_optional_text(payload.get("goal")),
        deliverable_expected=bool(payload.get("deliverable_expected", False)),
    )


def resolve_workflow_reference(
    registry: RuntimeRegistry, value: str | None
) -> str | None:
    candidate = _optional_text(value)
    if candidate is None:
        return None
    if candidate in registry.workflow_definitions:
        return candidate

    lowered = candidate.casefold()
    by_name = [
        workflow_id
        for workflow_id, definition in registry.workflow_definitions.items()
        if str(definition.metadata.get("name", "")).strip().casefold() == lowered
    ]
    if len(by_name) == 1:
        return by_name[0]

    by_hint = [
        workflow_id
        for workflow_id, definition in registry.workflow_definitions.items()
        if lowered
        in {
            hint.casefold()
            for hint in selection_hints_for_metadata(definition.metadata)
        }
    ]
    if len(by_hint) == 1:
        return by_hint[0]
    return None


def summarize_conversation(
    messages: tuple[ProxyInputMessage, ...], *, limit: int = 10
) -> str:
    return "\n".join(_transcript_lines(messages)[-limit:]).strip()


def _transcript_lines(messages: tuple[ProxyInputMessage, ...]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if message.tool_calls:
            tool_names = ", ".join(tool_call.name for tool_call in message.tool_calls)
            lines.append(f"{message.role}: [tool_calls {tool_names}]")
        if message.content:
            label = message.name or message.role
            lines.append(f"{label}: {message.content}")
        if message.role == "tool" and message.tool_call_id:
            label = f"tool_result[{message.tool_call_id}]"
            if message.name:
                label = f"{label}<{message.name}>"
            lines.append(f"{label}: {message.content}")
    return lines or ["(empty)"]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped
