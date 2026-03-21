from __future__ import annotations

from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.transcript import summarize_conversation


def orchestrator_turn_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    active_workroom_id: str | None = None,
    active_workroom_participants: tuple[str, ...] = (),
    active_workroom_message: str | None = None,
) -> str:
    lines = [
        "You are the lead developer in an AI software firm.",
        "The user is the product manager.",
        (
            "Reply directly to the product manager when you can move the work "
            "forward yourself."
        ),
        (
            "When you need help from the team, message one workroom at a time "
            "instead of describing a plan in JSON."
        ),
        (
            "Do not send a product-manager-facing answer while you are still "
            "gathering internal help."
        ),
        "",
        "Conversation summary:",
        summarize_conversation(request.messages, limit=12) or "(none)",
        "",
        "Latest user request:",
        request.latest_user_text() or "(none)",
    ]
    if goal:
        lines.extend(["", "Current goal:", goal])
    if current_brief:
        lines.extend(["", "Current best brief:", current_brief])
    if active_workroom_id:
        lines.extend(["", "Workroom currently in progress:", active_workroom_id])
    if active_workroom_participants:
        lines.extend(
            [
                "",
                "Current workroom staffing:",
                ", ".join(active_workroom_participants),
            ]
        )
    if active_workroom_message:
        lines.extend(
            [
                "",
                "Latest message to the active workroom:",
                active_workroom_message,
            ]
        )
    if worklog:
        lines.extend(["", "Team work so far:", *worklog[-12:]])
    return "\n".join(lines).strip()


def workroom_round_prompt(
    *,
    workroom_id: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    goal: str,
    workroom_message: str | None = None,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} working inside workroom {workroom_id}.",
        (
            "The lead developer is using this workroom for collaboration, not as "
            "a rigid script."
        ),
    ]
    if role_instance_label:
        lines.extend(
            [
                "",
                f"Current staffed instance: {role_instance_label}",
            ]
        )
    if role_instance_context:
        lines.extend(
            [
                "",
                role_instance_context,
            ]
        )
    lines.extend(
        [
            "",
            "Conversation summary:",
            transcript_summary or "(none)",
            "",
            "Overall goal:",
            goal or "(none)",
        ]
    )
    if workroom_message:
        lines.extend(
            [
                "",
                "Latest lead-dev message to this workroom:",
                workroom_message,
            ]
        )
    if prior_outputs:
        lines.extend(
            [
                "",
                "Prior workroom outputs:",
                *prior_outputs[-6:],
            ]
        )
    return "\n".join(lines).strip()
