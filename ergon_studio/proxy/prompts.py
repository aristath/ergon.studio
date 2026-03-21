from __future__ import annotations

from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.transcript import summarize_conversation


def orchestrator_turn_prompt(
    request: ProxyTurnRequest,
    *,
    worklog: tuple[str, ...] = (),
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
        (
            "If you are going to message a workroom or use a tool in this move, "
            "emit only the tool call and no assistant text."
        ),
        "",
        "Conversation summary:",
        summarize_conversation(request.messages, limit=12) or "(none)",
        "",
        "Latest user request:",
        request.latest_user_text() or "(none)",
    ]
    if worklog:
        lines.extend(["", "Team work so far:", *worklog[-12:]])
    return "\n".join(lines).strip()


def workroom_round_prompt(
    *,
    workroom_name: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    user_request: str,
    workroom_message: str | None = None,
    transcript_summary: str,
    prior_work: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} working inside workroom {workroom_name}.",
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
            user_request or "(none)",
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
    if prior_work:
        lines.extend(
            [
                "",
                "Relevant team work so far:",
                *prior_work[-6:],
            ]
        )
    lines.extend(
        [
            "",
            "Keep ownership of the task while you are still actively working.",
            (
                "Do not hand the task back to the lead developer after every small "
                "status update."
            ),
            (
                "Use tools when you need them and keep working through the task "
                "until you are done, blocked, or truly need a decision."
            ),
            (
                "When you are done, blocked, or need a decision from the lead "
                "developer, call `reply_lead_dev` with a concise update."
            ),
        ]
    )
    return "\n".join(lines).strip()
