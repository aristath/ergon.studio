from __future__ import annotations


def orchestrator_turn_prompt(
    *,
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
            "When you need collaborative help from the team, open a channel to "
            "the relevant people instead of describing a plan in JSON."
        ),
        (
            "When you need multiple independent implementations of the same task "
            "— for example a best-of-N exploration — use run_parallel instead. "
            "Each sub-session runs in isolation with its own workspace and no host "
            "tools; results are returned to you automatically."
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
            "When you need a channel, run_parallel, or a host tool, "
            "use the tool call directly."
        ),
        "",
        "The product-manager conversation is provided separately as message history.",
    ]
    if open_channels:
        lines.extend(["", "Open channels:", *open_channels])
    return "\n".join(lines).strip()


def subsession_prompt(*, agent_id: str) -> str:
    """System prompt injected into every run_parallel sub-session."""
    return "\n".join(
        [
            f"You are {agent_id} working in an isolated coding sub-session.",
            (
                "Use the read_file, write_file, and list_files tools to interact "
                "with the codebase. Always use absolute paths."
            ),
            "You do not have access to host tools in this session.",
            "When you are done, reply with your findings or the result in plain text.",
        ]
    )


def channel_message_prompt(
    *,
    channel_name: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
) -> str:
    lines = [
        f"You are {agent_id} in channel {channel_name}.",
        "This channel is a real team conversation, not a scripted process.",
        "The orchestrator appears as the user in the channel history.",
        "Your own prior channel replies appear as assistant messages.",
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
            "Channel history is provided separately as conversation messages.",
            "Reply naturally in the conversation.",
            "Use tools when you need them.",
            (
                "Do not invent process markers or status keywords. Just say what "
                "you found, changed, or need in plain language."
            ),
        ]
    )
    return "\n".join(lines).strip()
