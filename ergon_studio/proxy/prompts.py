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
    active_workroom_request: str | None = None,
) -> str:
    lines = [
        "You are the lead developer in an AI software firm.",
        "The user is the product manager.",
        (
            "Reply directly to the product manager when you can move the work "
            "forward yourself."
        ),
        (
            "When you need help from the team, use one internal communication "
            "tool at a time instead of describing a plan in JSON."
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
    if active_workroom_request:
        lines.extend(["", "Current workroom assignment:", active_workroom_request])
    if worklog:
        lines.extend(["", "Team work so far:", *worklog[-12:]])
    return "\n".join(lines).strip()


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


def workroom_round_prompt(
    *,
    workroom_id: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    goal: str,
    current_brief: str,
    workroom_request: str | None = None,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
    comparison_candidates: tuple[str, ...] = (),
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
        "",
        "Current brief:",
        current_brief or "(none)",
        ]
    )
    if workroom_request:
        lines.extend(
            [
                "",
                "Current workroom assignment:",
                workroom_request,
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
    if comparison_candidates:
        lines.extend(
            [
                "",
                "Alternative attempts from the previous stage:",
                *comparison_candidates[-8:],
                "",
                (
                    "Treat these as competing options to compare, select, or "
                    "build on deliberately."
                ),
            ]
        )
    return "\n".join(lines).strip()


def group_chat_turn_prompt(
    *,
    workroom_id: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    goal: str,
    transcript_summary: str,
    current_brief: str,
    workroom_request: str | None = None,
    prior_outputs: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} speaking in workroom {workroom_id}.",
        "Respond to the current discussion and move the decision forward.",
        "Add real value from your role instead of repeating the room.",
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
            "Goal:",
            goal or "(none)",
            "",
            "Current brief:",
            current_brief or "(none)",
        ]
    )
    if workroom_request:
        lines.extend(
            [
                "",
                "Current workroom assignment:",
                workroom_request,
            ]
        )
    if prior_outputs:
        lines.extend(
            [
                "",
                "Discussion so far:",
                *prior_outputs[-8:],
            ]
    )
    return "\n".join(lines).strip()


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


def workroom_summary_prompt(
    *,
    workroom_id: str,
    goal: str,
    outputs: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            f"The workroom {workroom_id} completed.",
            "",
            "Goal:",
            goal or "(none)",
            "",
            "Workroom outputs:",
            *(outputs or ("(none)",)),
            "",
            "Write the final host-facing answer.",
        ]
    ).strip()

def _optional_prompt_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
