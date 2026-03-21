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
from ergon_studio.proxy.workroom import AD_HOC_WORKROOM_ID
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workroom_policy import (
    workroom_acceptance_mode_for_metadata,
    workroom_delivery_candidate_for_metadata,
    workroom_selection_hints_for_metadata,
)


@dataclass(frozen=True)
class ProxyTurnPlan:
    mode: str
    workroom_id: str | None = None
    agent_id: str | None = None
    specialists: tuple[str, ...] = ()
    specialist_counts: tuple[tuple[str, int], ...] = ()
    workroom_request: str | None = None
    delivery_requirements: tuple[str, ...] | None = None
    request: str | None = None
    rationale: str | None = None


def build_turn_planner_instructions(registry: RuntimeRegistry) -> str:
    delivery_values = ", ".join(DELIVERY_REQUIREMENT_VALUES)
    workroom_lines = []
    for workroom_id, definition in sorted(registry.workroom_definitions.items()):
        hints = ", ".join(
            workroom_selection_hints_for_metadata(definition.metadata)
        ) or "none"
        shape = definition.metadata.get("shape", "unknown")
        delivery_candidate = workroom_delivery_candidate_for_metadata(
            definition.metadata
        )
        acceptance_mode = workroom_acceptance_mode_for_metadata(definition.metadata)
        workroom_lines.append(
            f"- {workroom_id}: shape={shape} "
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
            (
                '- "open_workroom": open a staffed collaboration room. '
                "You may target a named preset or leave target null for an "
                "ad-hoc room."
            ),
            '- "continue_workroom": continue the room already in progress.',
            '- "deliver": return the current result to the product manager.',
            "",
            "Rules:",
            "- Think like a pragmatic lead developer, not a classifier.",
            (
                "- Keep the JSON compact. Only include fields that matter for the "
                "very next move."
            ),
            (
                "- target names the specialist, a workroom preset, or "
                "'current' for continue_workroom."
            ),
            (
                "- assignment is the natural-language brief for the next move."
            ),
            (
                "- staffing is optional for workroom actions. For ad-hoc rooms, "
                "staffing defines who is in the room. Repeating a role means "
                "multiple independent instances, such as "
                "['coder','coder','reviewer']."
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
                "- Prefer open_workroom when the work would benefit from "
                "multi-agent collaboration, brainstorming, critique, or "
                "parallel attempts."
            ),
            (
                "- Prefer continue_workroom when the current room is still the "
                "right place to keep collaborating."
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
            "Available workroom presets:",
            *workroom_lines,
            "",
            "Available specialists:",
            *specialist_lines,
            "",
            "Required JSON shape:",
            '{"action":"reply|delegate|open_workroom|continue_workroom|deliver","target":null,"assignment":"","staffing":[],"delivery_requirements":[],"rationale":""}',
        ]
    )


def build_turn_planner_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    active_workroom_id: str | None = None,
    active_specialists: tuple[str, ...] = (),
    active_specialist_counts: tuple[tuple[str, int], ...] = (),
    active_workroom_request: str | None = None,
    active_delivery_requirements: tuple[str, ...] = (),
    satisfied_delivery_evidence: tuple[str, ...] = (),
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
    if active_workroom_id:
        lines.extend(
            [
                "",
                "Workroom currently in progress:",
                active_workroom_id,
            ]
        )
    if active_workroom_request:
        lines.extend(
            [
                "",
                "Current workroom assignment:",
                active_workroom_request,
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
    if "action" not in payload:
        raise ValueError("planner output must include action")
    return _parse_action_plan(payload, registry=registry)


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

    if action == "open_workroom":
        workroom_id = resolve_workroom_reference(registry, target)
        if workroom_id is None and (specialists or specialist_counts):
            workroom_id = AD_HOC_WORKROOM_ID
        return ProxyTurnPlan(
            mode="workroom",
            workroom_id=workroom_id,
            specialists=specialists,
            specialist_counts=specialist_counts,
            workroom_request=assignment,
            delivery_requirements=delivery_requirements,
            rationale=rationale,
        )

    if action == "continue_workroom":
        workroom_id = None if target == "current" else resolve_workroom_reference(
            registry,
            target,
        )
        if workroom_id is None and target == AD_HOC_WORKROOM_ID:
            workroom_id = AD_HOC_WORKROOM_ID
        return ProxyTurnPlan(
            mode="continue_workroom",
            workroom_id=workroom_id,
            specialists=specialists,
            specialist_counts=specialist_counts,
            workroom_request=assignment,
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


def resolve_workroom_reference(
    registry: RuntimeRegistry, value: str | None
) -> str | None:
    candidate = _optional_text(value)
    if candidate is None:
        return None
    if candidate in registry.workroom_definitions:
        return candidate

    lowered = candidate.casefold()
    by_name = [
        workroom_id
        for workroom_id, definition in registry.workroom_definitions.items()
        if str(definition.metadata.get("name", "")).strip().casefold() == lowered
    ]
    if len(by_name) == 1:
        return by_name[0]

    by_hint = [
        workroom_id
        for workroom_id, definition in registry.workroom_definitions.items()
        if lowered
        in {
            hint.casefold()
            for hint in workroom_selection_hints_for_metadata(definition.metadata)
        }
    ]
    if len(by_hint) == 1:
        return by_hint[0]
    return None


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


def _normalize_action(value: object) -> str:
    if not isinstance(value, str):
        return "reply"
    normalized = value.strip().lower()
    if normalized in {
        "reply",
        "delegate",
        "open_workroom",
        "continue_workroom",
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
