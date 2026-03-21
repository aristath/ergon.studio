from __future__ import annotations

import json
import zlib
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass

from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall
from ergon_studio.proxy.playbook_focus import normalize_playbook_focus
from ergon_studio.proxy.selection_outcome import ProxySelectionOutcome

_TOKEN_PREFIX = "ergon:"
_TOKEN_VERSION = 1


@dataclass(frozen=True)
class ContinuationState:
    mode: str
    agent_id: str
    participant_label: str | None = None
    workflow_id: str | None = None
    workflow_specialists: tuple[str, ...] = ()
    workflow_specialist_counts: tuple[tuple[str, int], ...] = ()
    workflow_request: str | None = None
    workflow_focus: str | None = None
    last_stage_outputs: tuple[str, ...] = ()
    last_stage_parallel_attempts: bool = False
    selection_outcome: ProxySelectionOutcome | None = None
    step_index: int | None = None
    agent_index: int | None = None
    request_text: str | None = None
    goal: str | None = None
    current_brief: str | None = None
    decision_history: tuple[str, ...] = ()
    workflow_outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingContinuation:
    state: ContinuationState
    assistant_message: ProxyInputMessage | None
    tool_results: tuple[ProxyInputMessage, ...]


def encode_continuation_tool_call(
    tool_call: ProxyToolCall, *, state: ContinuationState
) -> ProxyToolCall:
    payload = {
        "v": _TOKEN_VERSION,
        "m": state.mode,
        "a": state.agent_id,
        "tn": tool_call.name,
        "ta": tool_call.arguments_json,
    }
    if state.participant_label is not None:
        payload["al"] = state.participant_label
    if state.workflow_id is not None:
        payload["w"] = state.workflow_id
    if state.workflow_specialists:
        payload["p"] = list(state.workflow_specialists)
    if state.workflow_specialist_counts:
        payload["pc"] = {
            agent_id: count for agent_id, count in state.workflow_specialist_counts
        }
    if state.workflow_request is not None:
        payload["pr"] = state.workflow_request
    if state.workflow_focus is not None:
        payload["pf"] = state.workflow_focus
    if state.last_stage_outputs:
        payload["ls"] = list(state.last_stage_outputs)
    if state.last_stage_parallel_attempts:
        payload["lp"] = True
    if state.selection_outcome is not None:
        payload["so"] = {
            "m": state.selection_outcome.mode,
            "i": state.selection_outcome.selected_candidate_index,
            "t": state.selection_outcome.selected_candidate_text,
            "s": state.selection_outcome.summary,
            "n": state.selection_outcome.next_refinement,
        }
    if state.step_index is not None:
        payload["s"] = state.step_index
    if state.agent_index is not None:
        payload["i"] = state.agent_index
    if state.request_text is not None:
        payload["r"] = state.request_text
    if state.goal is not None:
        payload["g"] = state.goal
    if state.current_brief is not None:
        payload["c"] = state.current_brief
    if state.decision_history:
        payload["h"] = list(state.decision_history)
    if state.workflow_outputs:
        payload["o"] = list(state.workflow_outputs)
    encoded = (
        urlsafe_b64encode(
            zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        )
        .decode("ascii")
        .rstrip("=")
    )
    return ProxyToolCall(
        id=f"{_TOKEN_PREFIX}{encoded}:{tool_call.id}",
        name=tool_call.name,
        arguments_json=tool_call.arguments_json,
    )


def decode_continuation_from_tool_call_id(
    tool_call_id: str,
) -> ContinuationState | None:
    payload = _decode_payload(tool_call_id)
    if payload is None:
        return None
    if payload.get("v") != _TOKEN_VERSION:
        return None
    mode = payload.get("m")
    agent_id = payload.get("a")
    participant_label = payload.get("al")
    workflow_id = payload.get("w")
    workflow_specialists = payload.get("p", [])
    workflow_specialist_counts_payload = payload.get("pc", {})
    workflow_request = payload.get("pr")
    workflow_focus = payload.get("pf")
    last_stage_outputs = payload.get("ls", [])
    last_stage_parallel_attempts = payload.get("lp", False)
    selection_outcome_payload = payload.get("so")
    step_index = payload.get("s")
    agent_index = payload.get("i")
    request_text = payload.get("r")
    goal = payload.get("g")
    current_brief = payload.get("c")
    decision_history = payload.get("h", [])
    workflow_outputs = payload.get("o", [])
    if not isinstance(mode, str) or not isinstance(agent_id, str):
        return None
    if participant_label is not None and not isinstance(participant_label, str):
        return None
    if workflow_id is not None and not isinstance(workflow_id, str):
        return None
    if not isinstance(workflow_specialists, list) or not all(
        isinstance(item, str) for item in workflow_specialists
    ):
        return None
    workflow_specialist_counts = _decode_specialist_counts(
        workflow_specialist_counts_payload
    )
    if (
        workflow_specialist_counts_payload is not None
        and workflow_specialist_counts_payload != {}
        and workflow_specialist_counts is None
    ):
        return None
    if not isinstance(last_stage_outputs, list) or not all(
        isinstance(item, str) for item in last_stage_outputs
    ):
        return None
    if workflow_request is not None and not isinstance(workflow_request, str):
        return None
    if workflow_focus is not None and not isinstance(workflow_focus, str):
        return None
    workflow_focus = normalize_playbook_focus(workflow_focus)
    if payload.get("pf") is not None and workflow_focus is None:
        return None
    if not isinstance(last_stage_parallel_attempts, bool):
        return None
    selection_outcome = _decode_selection_outcome(selection_outcome_payload)
    if selection_outcome_payload is not None and selection_outcome is None:
        return None
    if step_index is not None and not isinstance(step_index, int):
        return None
    if agent_index is not None and not isinstance(agent_index, int):
        return None
    if request_text is not None and not isinstance(request_text, str):
        return None
    if goal is not None and not isinstance(goal, str):
        return None
    if current_brief is not None and not isinstance(current_brief, str):
        return None
    if not isinstance(decision_history, list) or not all(
        isinstance(item, str) for item in decision_history
    ):
        return None
    if not isinstance(workflow_outputs, list) or not all(
        isinstance(item, str) for item in workflow_outputs
    ):
        return None
    return ContinuationState(
        mode=mode,
        agent_id=agent_id,
        participant_label=participant_label,
        workflow_id=workflow_id,
        workflow_specialists=tuple(workflow_specialists),
        workflow_specialist_counts=workflow_specialist_counts or (),
        workflow_request=workflow_request,
        workflow_focus=workflow_focus,
        last_stage_outputs=tuple(last_stage_outputs),
        last_stage_parallel_attempts=last_stage_parallel_attempts,
        selection_outcome=selection_outcome,
        step_index=step_index,
        agent_index=agent_index,
        request_text=request_text,
        goal=goal,
        current_brief=current_brief,
        decision_history=tuple(decision_history),
        workflow_outputs=tuple(workflow_outputs),
    )


def decode_original_tool_call(tool_call_id: str) -> ProxyToolCall | None:
    payload = _decode_payload(tool_call_id)
    original = original_tool_call_id(tool_call_id)
    if payload is None or original is None:
        return None
    name = payload.get("tn")
    arguments = payload.get("ta", "")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        return None
    return ProxyToolCall(
        id=tool_call_id,
        name=name,
        arguments_json=arguments,
    )


def _decode_payload(tool_call_id: str) -> dict[str, object] | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    try:
        encoded_payload, _original = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 1)
    except ValueError:
        return None
    padding = "=" * (-len(encoded_payload) % 4)
    try:
        payload = json.loads(
            zlib.decompress(
                urlsafe_b64decode((encoded_payload + padding).encode("ascii"))
            ).decode("utf-8")
        )
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def original_tool_call_id(tool_call_id: str) -> str | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    try:
        _encoded_payload, original = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 1)
    except ValueError:
        return None
    return original or None


def _decode_selection_outcome(
    payload: object,
) -> ProxySelectionOutcome | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    mode = payload.get("m")
    if not isinstance(mode, str) or not mode:
        return None
    selected_candidate_index = payload.get("i")
    if selected_candidate_index is not None and not isinstance(
        selected_candidate_index, int
    ):
        return None
    selected_candidate_text = payload.get("t")
    if selected_candidate_text is not None and not isinstance(
        selected_candidate_text, str
    ):
        return None
    summary = payload.get("s")
    if summary is not None and not isinstance(summary, str):
        return None
    next_refinement = payload.get("n")
    if next_refinement is not None and not isinstance(next_refinement, str):
        return None
    return ProxySelectionOutcome(
        mode=mode,
        selected_candidate_index=selected_candidate_index,
        selected_candidate_text=selected_candidate_text,
        summary=summary,
        next_refinement=next_refinement,
    )


def _decode_specialist_counts(
    payload: object,
) -> tuple[tuple[str, int], ...] | None:
    if payload is None:
        return ()
    if not isinstance(payload, dict):
        return None
    specialist_counts: list[tuple[str, int]] = []
    for raw_agent_id, raw_count in payload.items():
        if not isinstance(raw_agent_id, str) or not raw_agent_id:
            return None
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            return None
        if raw_count <= 0:
            return None
        specialist_counts.append((raw_agent_id, raw_count))
    return tuple(specialist_counts)


def latest_continuation(
    messages: tuple[ProxyInputMessage, ...],
) -> ContinuationState | None:
    pending = latest_pending_continuation(messages)
    if pending is None:
        return None
    return pending.state


def latest_pending_continuation(
    messages: tuple[ProxyInputMessage, ...],
) -> PendingContinuation | None:
    if not messages or messages[-1].role != "tool":
        return None

    tool_results_reversed: list[ProxyInputMessage] = []
    index = len(messages) - 1
    while index >= 0 and messages[index].role == "tool":
        tool_results_reversed.append(messages[index])
        index -= 1
    tool_results = tuple(reversed(tool_results_reversed))
    assistant_message: ProxyInputMessage | None = None
    if index >= 0:
        candidate = messages[index]
        if candidate.role == "assistant" and candidate.tool_calls:
            assistant_message = candidate
            assistant_call_ids = {tool_call.id for tool_call in candidate.tool_calls}
            if any(
                (message.tool_call_id or "") not in assistant_call_ids
                for message in tool_results
            ):
                return None

    state: ContinuationState | None = None
    for message in tool_results:
        decoded = decode_continuation_from_tool_call_id(message.tool_call_id or "")
        if decoded is None:
            return None
        if state is None:
            state = decoded
        elif decoded != state:
            return None
    if state is None:
        return None

    return PendingContinuation(
        state=state,
        assistant_message=assistant_message,
        tool_results=tool_results,
    )


def continuation_tool_calls(pending: PendingContinuation) -> tuple[ProxyToolCall, ...]:
    if pending.assistant_message is None:
        return ()
    result_ids = {message.tool_call_id for message in pending.tool_results}
    return tuple(
        tool_call
        for tool_call in pending.assistant_message.tool_calls
        if tool_call.id in result_ids
    )


def continuation_result_map(pending: PendingContinuation) -> dict[str, str]:
    return {
        message.tool_call_id or "": message.content
        for message in pending.tool_results
        if message.tool_call_id
    }
