from __future__ import annotations

from ergon_studio.proxy.channels import ChannelMessage
from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.transcript import summarize_conversation


def orchestrator_turn_prompt(
    request: ProxyTurnRequest,
    *,
    worklog: tuple[str, ...] = (),
    open_channels: tuple[str, ...] = (),
) -> str:
    lines = [
        "You are the lead developer in an AI software firm.",
        "The user is the product manager.",
        (
            "Reply directly to the product manager when you can move the work "
            "forward yourself."
        ),
        (
            "When you need help from the team, open a channel to the relevant "
            "people instead of describing a plan in JSON."
        ),
        (
            "When you message a channel, explicitly target the people you want "
            "to answer. Do not rely on the runtime to decide who should respond."
        ),
        (
            "Do not send a product-manager-facing answer while you are still "
            "gathering internal help."
        ),
        (
            "When you need a channel or a host tool, use the tool call directly."
        ),
        "",
        "Conversation summary:",
        summarize_conversation(request.messages, limit=12) or "(none)",
        "",
        "Latest user request:",
        request.latest_user_text() or "(none)",
    ]
    if open_channels:
        lines.extend(["", "Open channels:", *open_channels])
    if worklog:
        lines.extend(["", "Team work so far:", *worklog[-12:]])
    return "\n".join(lines).strip()


def channel_message_prompt(
    *,
    channel_name: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    user_request: str,
    transcript_summary: str,
    channel_transcript: tuple[ChannelMessage, ...],
    prior_work: tuple[str, ...],
) -> str:
    lines = [
        f"You are {agent_id} in channel {channel_name}.",
        (
            "The lead developer opened this channel to collaborate with you like a "
            "real teammate, not to run a scripted process."
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
    if channel_transcript:
        lines.extend(
            [
                "",
                "Channel transcript so far:",
                *(message.render() for message in channel_transcript[-8:]),
            ]
        )
    if prior_work:
        lines.extend(
            [
                "",
                "Recent team notes:",
                *prior_work[-6:],
            ]
        )
    lines.extend(
        [
            "",
            "Reply naturally in the conversation.",
            "Use tools when you need them.",
            (
                "Do not invent process markers or status keywords. Just say what "
                "you found, changed, or need in plain language."
            ),
        ]
    )
    return "\n".join(lines).strip()
