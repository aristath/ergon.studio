from __future__ import annotations

import json

from ergon_studio.proxy.models import ProxyTurnRequest
from ergon_studio.proxy.planner import summarize_conversation
from ergon_studio.proxy.playbook_focus import playbook_focus_instruction
from ergon_studio.proxy.selection_outcome import (
    ProxySelectionOutcome,
    selection_outcome_lines,
)


def direct_reply_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        "You are the lead developer replying to the product manager.",
        "Be practical, collaborative, and decisive.",
        "Use the full conversation transcript below as context.",
        "",
        summarize_conversation(request.messages, limit=12),
        "",
        "Latest user request:",
        request.latest_user_text() or "(none)",
    ]
    if goal:
        lines.extend(
            [
                "",
                "Current goal:",
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
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer chose to handle this directly:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "Good outcome for this move:",
                success_criteria,
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
    return "\n".join(lines).strip()


def finish_reply_prompt(
    request: ProxyTurnRequest,
    *,
    goal: str | None = None,
    current_brief: str | None = None,
    worklog: tuple[str, ...] = (),
    delivery_requirements: tuple[str, ...] = (),
    delivery_evidence: tuple[str, ...] = (),
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        (
            "You are the lead developer delivering the current result to the "
            "product manager."
        ),
        "Assume the internal work is done unless you explicitly say otherwise.",
        "Write the host-facing response now.",
        "",
        "Conversation summary:",
        summarize_conversation(request.messages, limit=12) or "(none)",
    ]
    if goal:
        lines.extend(
            [
                "",
                "Goal:",
                goal,
            ]
        )
    if current_brief:
        lines.extend(
            [
                "",
                "Best current result:",
                current_brief,
            ]
        )
    if delivery_requirements:
        lines.extend(
            [
                "",
                "Delivery requirements already satisfied:",
                ", ".join(delivery_requirements),
            ]
        )
        if delivery_evidence:
            lines.extend(
                [
                    "Supporting evidence gathered:",
                    ", ".join(delivery_evidence),
                ]
            )
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer is delivering now:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "What this delivery should achieve:",
                success_criteria,
            ]
        )
    if worklog:
        lines.extend(
            [
                "",
                "Internal worklog:",
                *worklog[-12:],
            ]
        )
    return "\n".join(lines).strip()


def specialist_prompt(
    *,
    specialist_id: str,
    request_text: str,
    transcript_summary: str,
    current_brief: str | None = None,
    move_rationale: str | None = None,
    success_criteria: str | None = None,
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
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer assigned you this slice:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "What a good result looks like:",
                success_criteria,
            ]
        )
    return "\n".join(lines).strip()


def workflow_step_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    goal: str,
    current_brief: str,
    playbook_request: str | None = None,
    playbook_focus: str | None = None,
    transcript_summary: str,
    prior_outputs: tuple[str, ...],
    comparison_candidates: tuple[str, ...] = (),
    selection_outcome: ProxySelectionOutcome | None = None,
    comparison_mode: str | None = None,
    comparison_criteria: str | None = None,
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        f"You are {agent_id} working inside playbook {workflow_id}.",
        "The lead developer is using this playbook as a tactic, not as a rigid script.",
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
    if playbook_request:
        lines.extend(
            [
                "",
                "Current playbook round assignment:",
                playbook_request,
            ]
        )
    if playbook_focus:
        lines.extend(
            [
                "",
                "Current playbook round focus:",
                playbook_focus,
                playbook_focus_instruction(playbook_focus),
            ]
        )
    if prior_outputs:
        lines.extend(
            [
                "",
                "Prior workflow outputs:",
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
    if selection_outcome is not None:
        lines.extend(
            [
                "",
                *selection_outcome_lines(selection_outcome),
            ]
        )
    if comparison_mode:
        lines.extend(
            [
                "",
                "Current comparison task:",
                _comparison_mode_instruction(comparison_mode),
            ]
        )
    if comparison_criteria:
        lines.extend(
            [
                "",
                "Comparison criteria:",
                comparison_criteria,
            ]
        )
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer is using this playbook move now:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "What a good result for this move looks like:",
                success_criteria,
            ]
        )
    return "\n".join(lines).strip()


def group_chat_turn_prompt(
    *,
    workflow_id: str,
    agent_id: str,
    role_instance_label: str | None = None,
    role_instance_context: str | None = None,
    goal: str,
    transcript_summary: str,
    current_brief: str,
    playbook_request: str | None = None,
    playbook_focus: str | None = None,
    prior_outputs: tuple[str, ...],
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        f"You are {agent_id} speaking in playbook {workflow_id}.",
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
    if playbook_request:
        lines.extend(
            [
                "",
                "Current playbook round assignment:",
                playbook_request,
            ]
        )
    if playbook_focus:
        lines.extend(
            [
                "",
                "Current playbook round focus:",
                playbook_focus,
                playbook_focus_instruction(playbook_focus),
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
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer wants another discussion turn now:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "What a useful discussion turn should accomplish:",
                success_criteria,
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
    playbook_request: str | None,
    playbook_focus: str | None,
    participants: tuple[str, ...],
    prior_outputs: tuple[str, ...],
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        f"Workflow: {workflow_id}",
        f"Goal: {goal or '(none)'}",
        f"Current brief: {current_brief or '(none)'}",
        f"Available specialists: {', '.join(participants) or '(none)'}",
    ]
    if playbook_request:
        lines.append(f"Current round assignment: {playbook_request}")
    if playbook_focus:
        lines.append(
            f"Current round focus: {playbook_focus} "
            f"({playbook_focus_instruction(playbook_focus)})"
        )
    if move_rationale:
        lines.append(f"Why continue this playbook now: {move_rationale}")
    if success_criteria:
        lines.append(f"What the next round should achieve: {success_criteria}")
    lines.extend(
        [
            "",
            "Progress so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    )
    return "\n".join(lines).strip()


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
    playbook_request: str | None,
    playbook_focus: str | None,
    prior_outputs: tuple[str, ...],
    allowed: tuple[str, ...],
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        f"Workflow: {workflow_id}",
        f"You are {current_agent}.",
        f"Goal: {goal or '(none)'}",
        f"Current brief: {current_brief or '(none)'}",
        f"You may hand off to: {', '.join(allowed) or '(none)'}",
    ]
    if playbook_request:
        lines.append(f"Current round assignment: {playbook_request}")
    if playbook_focus:
        lines.append(
            f"Current round focus: {playbook_focus} "
            f"({playbook_focus_instruction(playbook_focus)})"
        )
    if move_rationale:
        lines.append(
            f"Why the lead developer is continuing this handoff now: {move_rationale}"
        )
    if success_criteria:
        lines.append(
            f"What the next handoff should accomplish: {success_criteria}"
        )
    lines.extend(
        [
            "",
            "Work so far:",
            *(prior_outputs[-8:] or ["(none)"]),
        ]
    )
    return "\n".join(lines).strip()


def comparison_outcome_instructions(*, candidate_count: int) -> str:
    return "\n".join(
        [
            (
                "You are extracting the structured result of a comparison stage "
                "for the lead developer."
            ),
            "Return JSON only.",
            (
                "Use selected_candidate_number as a 1-based number from the "
                "candidate list, or null if no single candidate was chosen."
            ),
            (
                'Return {"selected_candidate_number":1|null,'
                '"summary":"","next_refinement":""}.'
            ),
            (
                "The summary should explain the decision or synthesis clearly "
                "enough for the next stage to build on it."
            ),
            f"Candidate count: {candidate_count}",
        ]
    )


def comparison_outcome_prompt(
    *,
    workflow_id: str,
    goal: str,
    comparison_mode: str,
    comparison_candidates: tuple[str, ...],
    stage_outputs: tuple[str, ...],
    comparison_criteria: str | None = None,
    move_rationale: str | None = None,
    success_criteria: str | None = None,
) -> str:
    lines = [
        f"Workflow: {workflow_id}",
        f"Goal: {goal or '(none)'}",
        f"Comparison task: {_comparison_mode_instruction(comparison_mode)}",
        "",
        "Candidates:",
    ]
    for index, candidate in enumerate(comparison_candidates, start=1):
        lines.append(f"{index}. {candidate}")
    if comparison_criteria:
        lines.extend(
            [
                "",
                "Comparison criteria:",
                comparison_criteria,
            ]
        )
    if move_rationale:
        lines.extend(
            [
                "",
                "Why the lead developer requested this comparison:",
                move_rationale,
            ]
        )
    if success_criteria:
        lines.extend(
            [
                "",
                "What this comparison stage should achieve:",
                success_criteria,
            ]
        )
    lines.extend(
        [
            "",
            "Outputs from the current comparison stage:",
            *(stage_outputs or ("(none)",)),
        ]
    )
    return "\n".join(lines).strip()


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


def parse_comparison_outcome(
    raw: str | None,
    *,
    comparison_mode: str,
    comparison_candidates: tuple[str, ...],
) -> ProxySelectionOutcome | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    selected_candidate_number = payload.get("selected_candidate_number")
    if isinstance(selected_candidate_number, str):
        stripped = selected_candidate_number.strip()
        if stripped.isdigit():
            selected_candidate_number = int(stripped)
    if selected_candidate_number is not None and not isinstance(
        selected_candidate_number, int
    ):
        return None
    selected_candidate_index: int | None = None
    selected_candidate_text: str | None = None
    if isinstance(selected_candidate_number, int):
        if 1 <= selected_candidate_number <= len(comparison_candidates):
            selected_candidate_index = selected_candidate_number - 1
            selected_candidate_text = comparison_candidates[selected_candidate_index]
        else:
            return None
    summary = _optional_prompt_text(payload.get("summary"))
    next_refinement = _optional_prompt_text(payload.get("next_refinement"))
    if summary is None and next_refinement is None and selected_candidate_text is None:
        return None
    return ProxySelectionOutcome(
        mode=comparison_mode,
        selected_candidate_index=selected_candidate_index,
        selected_candidate_text=selected_candidate_text,
        summary=summary,
        next_refinement=next_refinement,
    )


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


def _comparison_mode_instruction(mode: str) -> str:
    if mode == "select_best":
        return "Choose the strongest candidate and explain why it wins."
    if mode == "synthesize_best":
        return "Combine the strongest parts of the alternatives into a better result."
    if mode == "critique_options":
        return (
            "Critique the alternatives clearly and surface the most important "
            "tradeoffs."
        )
    return mode


def _optional_prompt_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
