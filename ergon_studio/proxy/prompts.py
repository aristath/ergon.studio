from __future__ import annotations

import json

from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.planner import summarize_conversation


def direct_reply_prompt(request: ProxyTurnRequest) -> str:
    return "\n".join(
        [
            "You are the lead developer replying to the product manager.",
            "Be practical, collaborative, and decisive.",
            "Use the full conversation transcript below as context.",
            "",
            summarize_conversation(request.messages, limit=12),
            "",
            "Latest user request:",
            request.latest_user_text() or "(none)",
        ]
    ).strip()


def specialist_prompt(
    *,
    specialist_id: str,
    request_text: str,
    transcript_summary: str,
    current_brief: str | None = None,
) -> str:
    lines = [
        f"You are the {specialist_id} working for the lead developer.",
        "You were brought in for a focused assignment inside a software team.",
        "Solve the assigned slice well instead of re-litigating the whole project.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Assigned request:",
        request_text or "(none)",
    ]
    if current_brief:
        lines.extend(
            [
                "",
                "Current progress:",
                current_brief,
            ]
        )
    return "\n".join(lines).strip()


def workflow_step_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    goal: str,
    current_brief: str,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} working inside playbook {workflow_id}.",
        "The lead developer is using this playbook as a tactic, not as a rigid script.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Overall goal:",
        goal or "(none)",
        "",
        "Current brief:",
        current_brief or "(none)",
    ]
    if prior_outputs:
        lines.extend(
            [
                "",
                "Prior workflow outputs:",
                *prior_outputs[-6:],
            ]
        )
    return "\n".join(lines).strip()


def group_chat_turn_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    goal: str,
    transcript_summary: str,
    current_brief: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} speaking in playbook {workflow_id}.",
        "Respond to the current discussion and move the decision forward.",
        "Add real value from your role instead of repeating the room.",
        "",
        "Conversation summary:",
        transcript_summary or "(none)",
        "",
        "Goal:",
        goal or "(none)",
        "",
        "Current brief:",
        current_brief or "(none)",
    ]
    if prior_outputs:
        lines.extend(
            [
                "",
                "Discussion so far:",
                *prior_outputs[-8:],
            ]
        )
    return "\n".join(lines).strip()


def workflow_manager_instructions(participants: tuple[str, ...]) -> str:
    return "\n".join(
        [
            (
                "You are helping the lead developer choose the next specialist "
                "for an adaptive playbook."
            ),
            "Return JSON only.",
            f"Allowed agents: {', '.join(participants) or '(none)'}",
            (
                'Return {"agent_id":"<agent>" } to continue or '
                '{"agent_id":null} to finish.'
            ),
        ]
    )


def workflow_manager_prompt(
    *,
    workflow_id: str,
    goal: str,
    current_brief: str,
    participants: tuple[str, ...],
    prior_outputs: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"Workflow: {workflow_id}",
            f"Goal: {goal or '(none)'}",
            f"Current brief: {current_brief or '(none)'}",
            f"Available specialists: {', '.join(participants) or '(none)'}",
            "",
            "Progress so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    ).strip()


def handoff_selection_instructions(allowed: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "You are choosing the next specialist handoff in a collaborative playbook.",
            "Return JSON only.",
            f"Allowed next agents: {', '.join(allowed) or '(none)'}",
            (
                'Return {"agent_id":"<agent>" } to continue or '
                '{"agent_id":null} to finish.'
            ),
        ]
    )


def handoff_selection_prompt(
    *,
    workflow_id: str,
    current_agent: str,
    goal: str,
    current_brief: str,
    prior_outputs: tuple[str, ...],
    allowed: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"Workflow: {workflow_id}",
            f"You are {current_agent}.",
            f"Goal: {goal or '(none)'}",
            f"Current brief: {current_brief or '(none)'}",
            f"You may hand off to: {', '.join(allowed) or '(none)'}",
            "",
            "Work so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    ).strip()


def parse_agent_selection(
    raw: str | None,
    *,
    participants: tuple[str, ...],
) -> str | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("agent_id")
    if candidate is None:
        return None
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    if not stripped or stripped.casefold() in {"none", "null", "finish", "done"}:
        return None
    if stripped not in participants:
        return None
    return stripped


def summary_instructions() -> str:
    return "\n".join(
        [
            "Summarize the completed work for the product manager.",
            "Be concise and concrete.",
            "State what was decided or produced.",
            "Do not mention hidden chain-of-thought.",
        ]
    )


def delegation_summary_prompt(
    *,
    request_text: str,
    specialist_id: str,
    specialist_text: str,
) -> str:
    return "\n".join(
        [
            (
                f"The specialist {specialist_id} completed delegated work for "
                "the lead developer."
            ),
            "",
            "Original request:",
            request_text or "(none)",
            "",
            "Specialist output:",
            specialist_text or "(none)",
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()


def workflow_summary_prompt(
    *,
    workflow_id: str,
    goal: str,
    outputs: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"The playbook {workflow_id} completed.",
            "",
            "Goal:",
            goal or "(none)",
            "",
            "Workflow outputs:",
            *(outputs or ("(none)",)),
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()
