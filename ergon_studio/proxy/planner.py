from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ergon_studio.proxy.models import ProxyInputMessage, ProxyTurnRequest
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.workflow_policy import acceptance_mode_for_metadata, delivery_candidate_for_metadata, selection_hints_for_metadata


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
        workflow_lines.append(
            f"- {workflow_id}: orchestration={definition.metadata.get('orchestration', 'unknown')} "
            f"delivery_candidate={delivery_candidate_for_metadata(definition.metadata)} "
            f"acceptance={acceptance_mode_for_metadata(definition.metadata)} "
            f"selection_hints={hints}"
        )
    specialist_lines = []
    for agent_id, definition in sorted(registry.agent_definitions.items()):
        if agent_id == "orchestrator":
            continue
        specialist_lines.append(f"- {agent_id}: role={definition.metadata.get('role', agent_id)}")

    return "\n".join(
        [
            "You are the internal planning layer for an orchestration proxy.",
            "Choose the single best next action for the current host turn.",
            "Output JSON only.",
            "",
            "Allowed modes:",
            '- "act": let the orchestrator answer directly.',
            '- "delegate": hand the work to one specialist.',
            '- "workflow": run a named workflow end to end.',
            "",
            "Rules:",
            "- Prefer workflow for non-trivial implementation, debugging, review, or delivery work.",
            "- Prefer delegate only for narrow specialist work.",
            "- Use act for discussion, clarification, or direct orchestrator answers.",
            "- Preserve the full delivery goal when the user expects implemented output.",
            "- Do not rely on keyword matching. Infer intent from the actual request and transcript.",
            "",
            "Available workflows:",
            *workflow_lines,
            "",
            "Available specialists:",
            *specialist_lines,
            "",
            "Required JSON shape:",
            '{"mode":"workflow|delegate|act","workflow_id":null,"agent_id":null,"request":"","goal":"","deliverable_expected":false}',
        ]
    )


def build_turn_planner_prompt(request: ProxyTurnRequest) -> str:
    return "\n".join(
        [
            "Conversation transcript:",
            *_transcript_lines(request.messages),
            "",
            "Latest user request:",
            request.latest_user_text() or "(none)",
        ]
    ).strip()


def parse_turn_plan(raw: str, *, registry: RuntimeRegistry) -> ProxyTurnPlan:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid planner json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("planner output must be a JSON object")

    mode = str(payload.get("mode", "act")).strip().lower()
    if mode not in {"act", "delegate", "workflow"}:
        mode = "act"
    workflow_id = _optional_text(payload.get("workflow_id"))
    if workflow_id is not None and workflow_id not in registry.workflow_definitions:
        workflow_id = None
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


def summarize_conversation(messages: tuple[ProxyInputMessage, ...], *, limit: int = 10) -> str:
    return "\n".join(_transcript_lines(messages)[-limit:]).strip()


def _transcript_lines(messages: tuple[ProxyInputMessage, ...]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if message.tool_calls:
            lines.append(f"{message.role}: [tool_calls x{len(message.tool_calls)}]")
        if message.content:
            label = message.name or message.role
            lines.append(f"{label}: {message.content}")
        if message.role == "tool" and message.tool_call_id:
            lines.append(f"tool_result[{message.tool_call_id}]: {message.content}")
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
