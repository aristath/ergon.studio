from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ergon_studio.proxy.delivery_requirements import (
    DELIVERY_REQUIREMENT_VALUES,
    normalize_delivery_requirements,
    unmet_delivery_requirements,
)
from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.proxy.playbook_focus import normalize_playbook_focus
from ergon_studio.proxy.selection_outcome import (
    ProxySelectionOutcome,
    selection_outcome_lines,
)
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
    staffing_action: str | None = None
    specialists: tuple[str, ...] = ()
    specialist_counts: tuple[tuple[str, int], ...] = ()
    playbook_request: str | None = None
    playbook_focus: str | None = None
    delivery_requirements: tuple[str, ...] | None = None
    comparison_mode: str | None = None
    comparison_criteria: str | None = None
    request: str | None = None
    goal: str | None = None
    rationale: str | None = None
    success_criteria: str | None = None
    deliverable_expected: bool = False


def build_turn_planner_instructions(registry: RuntimeRegistry) -> str:
    delivery_values = ", ".join(DELIVERY_REQUIREMENT_VALUES)
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
            "Allowed actions:",
            '- "reply": the lead developer handles the next move directly.',
            '- "delegate": message one specialist with a focused assignment.',
            '- "start_playbook": start a named playbook tactic.',
            '- "continue_playbook": continue the playbook already in progress.',
            '- "deliver": return the current result to the product manager.',
            "",
            "Rules:",
            "- Think like a pragmatic lead developer, not a classifier.",
            (
                "- Keep the JSON compact. Only include fields that matter for the "
                "very next move."
            ),
            (
                "- target names the specialist, playbook, or 'current' playbook "
                "for continue_playbook."
            ),
            (
                "- assignment is the natural-language brief for the next move."
            ),
            (
                "- staffing is optional and only for playbook actions. It is a list "
                "of specialist ids. Repeating a role means multiple independent "
                "instances, such as ['coder','coder','reviewer']."
            ),
            (
                "- You may optionally set delivery_requirements to express the "
                "quality bar that must be met before the lead developer delivers. "
                f"Allowed values: {delivery_values}."
            ),
            (
                "- rationale is optional. Use it when it helps explain the next move."
            ),
            (
                "- Prefer reply for discussion, clarification, planning with "
                "the product "
                "manager, and small direct actions."
            ),
            (
                "- Prefer deliver when the work is materially done and the lead "
                "developer should hand back a concrete result."
            ),
            (
                "- Prefer delegate for narrow, well-bounded specialist work."
            ),
            (
                "- Prefer start_playbook when the work would benefit from a known "
                "multi-agent tactic."
            ),
            (
                "- Prefer continue_playbook when the current playbook is still the "
                "right tactic and should advance another round."
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
            '{"action":"reply|delegate|start_playbook|continue_playbook|deliver","target":null,"assignment":"","staffing":[],"delivery_requirements":[],"rationale":""}',
        ]
    )


def build_turn_planner_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    active_workflow_id: str | None = None,
    active_specialists: tuple[str, ...] = (),
    active_specialist_counts: tuple[tuple[str, int], ...] = (),
    active_playbook_request: str | None = None,
    active_playbook_focus: str | None = None,
    active_delivery_requirements: tuple[str, ...] = (),
    satisfied_delivery_evidence: tuple[str, ...] = (),
    selection_outcome: ProxySelectionOutcome | None = None,
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
    if selection_outcome is not None:
        lines.extend(
            [
                "",
                *selection_outcome_lines(selection_outcome),
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
    if active_playbook_request:
        lines.extend(
            [
                "",
                "Current playbook round assignment:",
                active_playbook_request,
            ]
        )
    if active_playbook_focus:
        lines.extend(
            [
                "",
                "Current playbook round focus:",
                active_playbook_focus,
            ]
        )
    if active_delivery_requirements:
        lines.extend(
            [
                "",
                "Current delivery requirements:",
                ", ".join(active_delivery_requirements),
            ]
        )
        if satisfied_delivery_evidence:
            lines.extend(
                [
                    "Satisfied delivery evidence:",
                    ", ".join(satisfied_delivery_evidence),
                ]
            )
        unmet = unmet_delivery_requirements(
            active_delivery_requirements,
            satisfied_delivery_evidence,
        )
        if unmet:
            lines.extend(
                [
                    "Still missing before delivery:",
                    ", ".join(unmet),
                ]
            )
    if active_specialists:
        lines.extend(
            [
                "",
                "Currently staffed specialists:",
                ", ".join(active_specialists),
            ]
        )
    if active_specialist_counts:
        lines.extend(
            [
                "",
                "Current role instance counts:",
                ", ".join(
                    f"{agent_id} x{count}"
                    for agent_id, count in active_specialist_counts
                ),
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

    if "action" in payload:
        return _parse_action_plan(payload, registry=registry)
    return _parse_legacy_plan(payload, registry=registry)


def _parse_action_plan(
    payload: dict[str, object],
    *,
    registry: RuntimeRegistry,
) -> ProxyTurnPlan:
    action = _normalize_action(payload.get("action"))
    assignment = _optional_text(payload.get("assignment"))
    rationale = _optional_text(payload.get("rationale"))
    delivery_requirements = (
        normalize_delivery_requirements(payload["delivery_requirements"])
        if "delivery_requirements" in payload
        else None
    )
    specialists, specialist_counts = _normalize_staffing_list(
        payload.get("staffing"),
        registry=registry,
    )
    target = _optional_text(payload.get("target"))

    if action == "delegate":
        agent_id = target if target in registry.agent_definitions else None
        return ProxyTurnPlan(
            mode="delegate",
            agent_id=agent_id,
            request=assignment,
            delivery_requirements=delivery_requirements,
            rationale=rationale,
        )

    if action == "start_playbook":
        return ProxyTurnPlan(
            mode="workflow",
            workflow_id=resolve_workflow_reference(registry, target),
            specialists=specialists,
            specialist_counts=specialist_counts,
            playbook_request=assignment,
            delivery_requirements=delivery_requirements,
            rationale=rationale,
        )

    if action == "continue_playbook":
        workflow_id = None if target == "current" else resolve_workflow_reference(
            registry,
            target,
        )
        staffing_action = (
            "replace" if specialists or specialist_counts else None
        )
        return ProxyTurnPlan(
            mode="continue_playbook",
            workflow_id=workflow_id,
            staffing_action=staffing_action,
            specialists=specialists,
            specialist_counts=specialist_counts,
            playbook_request=assignment,
            delivery_requirements=delivery_requirements,
            rationale=rationale,
        )

    if action == "deliver":
        return ProxyTurnPlan(
            mode="finish",
            delivery_requirements=delivery_requirements,
            rationale=rationale,
        )

    return ProxyTurnPlan(
        mode="act",
        delivery_requirements=delivery_requirements,
        rationale=rationale,
    )


def _parse_legacy_plan(
    payload: dict[str, object],
    *,
    registry: RuntimeRegistry,
) -> ProxyTurnPlan:

    mode = str(payload.get("mode", "act")).strip().lower()
    if mode not in {"act", "delegate", "workflow", "continue_playbook", "finish"}:
        mode = "act"
    staffing_action = _normalize_staffing_action(payload.get("staffing_action"))
    workflow_id = resolve_workflow_reference(
        registry, _optional_text(payload.get("workflow_id"))
    )
    agent_id = _optional_text(payload.get("agent_id"))
    if agent_id is not None and agent_id not in registry.agent_definitions:
        agent_id = None
    specialists = _normalize_specialists(payload.get("specialists"), registry=registry)
    specialist_counts = _normalize_specialist_counts(
        payload.get("specialist_counts"),
        registry=registry,
    )
    comparison_mode = _normalize_comparison_mode(payload.get("comparison_mode"))
    playbook_focus = normalize_playbook_focus(payload.get("playbook_focus"))
    delivery_requirements = normalize_delivery_requirements(
        payload["delivery_requirements"]
    ) if "delivery_requirements" in payload else None
    if mode not in {"workflow", "continue_playbook"}:
        staffing_action = None
        playbook_focus = None
    elif comparison_mode is not None and playbook_focus is None:
        playbook_focus = "compare"
    return ProxyTurnPlan(
        mode=mode,
        workflow_id=workflow_id,
        agent_id=agent_id,
        staffing_action=staffing_action,
        specialists=specialists,
        specialist_counts=specialist_counts,
        playbook_request=_optional_text(payload.get("playbook_request")),
        playbook_focus=playbook_focus,
        delivery_requirements=delivery_requirements,
        comparison_mode=comparison_mode,
        comparison_criteria=_optional_text(payload.get("comparison_criteria")),
        request=_optional_text(payload.get("request")),
        goal=_optional_text(payload.get("goal")),
        rationale=_optional_text(payload.get("rationale")),
        success_criteria=_optional_text(payload.get("success_criteria")),
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


def _normalize_specialists(
    value: object,
    *,
    registry: RuntimeRegistry,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    specialists: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if (
            not candidate
            or candidate == "orchestrator"
            or candidate not in registry.agent_definitions
            or candidate in specialists
        ):
            continue
        specialists.append(candidate)
    return tuple(specialists)


def _normalize_specialist_counts(
    value: object,
    *,
    registry: RuntimeRegistry,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict):
        return ()
    specialist_counts: list[tuple[str, int]] = []
    for raw_agent_id, raw_count in value.items():
        if not isinstance(raw_agent_id, str):
            continue
        agent_id = raw_agent_id.strip()
        if (
            not agent_id
            or agent_id == "orchestrator"
            or agent_id not in registry.agent_definitions
        ):
            continue
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            continue
        if raw_count <= 0:
            continue
        specialist_counts.append((agent_id, raw_count))
    return tuple(specialist_counts)


def _normalize_staffing_list(
    value: object,
    *,
    registry: RuntimeRegistry,
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    if not isinstance(value, list):
        return (), ()
    counts: dict[str, int] = {}
    order: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if (
            not candidate
            or candidate == "orchestrator"
            or candidate not in registry.agent_definitions
        ):
            continue
        if candidate not in counts:
            counts[candidate] = 0
            order.append(candidate)
        counts[candidate] += 1
    specialists = tuple(order)
    specialist_counts = tuple(
        (agent_id, count) for agent_id in order if (count := counts[agent_id]) > 1
    )
    return specialists, specialist_counts


def _normalize_comparison_mode(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate in {"select_best", "synthesize_best", "critique_options"}:
        return candidate
    return None


def _normalize_staffing_action(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"keep", "replace", "augment", "trim"}:
        return normalized
    return None


def _normalize_action(value: object) -> str:
    if not isinstance(value, str):
        return "reply"
    normalized = value.strip().lower()
    if normalized in {
        "reply",
        "delegate",
        "start_playbook",
        "continue_playbook",
        "deliver",
    }:
        return normalized
    return "reply"


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
