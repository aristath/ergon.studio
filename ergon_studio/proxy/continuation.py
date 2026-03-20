from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
import json
import zlib

from ergon_studio.proxy.models import ProxyInputMessage, ProxyToolCall


_TOKEN_PREFIX = "ergon:"
_TOKEN_VERSION = 1


@dataclass(frozen=True)
class ContinuationState:
    mode: str
    agent_id: str
    workflow_id: str | None = None
    step_index: int | None = None
    request_text: str | None = None
    goal: str | None = None
    current_brief: str | None = None
    workflow_outputs: tuple[str, ...] = ()


def encode_continuation_tool_call(tool_call: ProxyToolCall, *, state: ContinuationState) -> ProxyToolCall:
    payload = {
        "v": _TOKEN_VERSION,
        "m": state.mode,
        "a": state.agent_id,
    }
    if state.workflow_id is not None:
        payload["w"] = state.workflow_id
    if state.step_index is not None:
        payload["s"] = state.step_index
    if state.request_text is not None:
        payload["r"] = state.request_text
    if state.goal is not None:
        payload["g"] = state.goal
    if state.current_brief is not None:
        payload["c"] = state.current_brief
    if state.workflow_outputs:
        payload["o"] = list(state.workflow_outputs)
    encoded = urlsafe_b64encode(zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))).decode("ascii").rstrip("=")
    return ProxyToolCall(
        id=f"{_TOKEN_PREFIX}{encoded}:{tool_call.id}",
        name=tool_call.name,
        arguments_json=tool_call.arguments_json,
    )


def decode_continuation_from_tool_call_id(tool_call_id: str) -> ContinuationState | None:
    if not tool_call_id.startswith(_TOKEN_PREFIX):
        return None
    try:
        encoded_payload, _original = tool_call_id[len(_TOKEN_PREFIX) :].split(":", 1)
    except ValueError:
        return None
    padding = "=" * (-len(encoded_payload) % 4)
    try:
        payload = json.loads(zlib.decompress(urlsafe_b64decode((encoded_payload + padding).encode("ascii"))).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("v") != _TOKEN_VERSION:
        return None
    mode = payload.get("m")
    agent_id = payload.get("a")
    workflow_id = payload.get("w")
    step_index = payload.get("s")
    request_text = payload.get("r")
    goal = payload.get("g")
    current_brief = payload.get("c")
    workflow_outputs = payload.get("o", [])
    if not isinstance(mode, str) or not isinstance(agent_id, str):
        return None
    if workflow_id is not None and not isinstance(workflow_id, str):
        return None
    if step_index is not None and not isinstance(step_index, int):
        return None
    if request_text is not None and not isinstance(request_text, str):
        return None
    if goal is not None and not isinstance(goal, str):
        return None
    if current_brief is not None and not isinstance(current_brief, str):
        return None
    if not isinstance(workflow_outputs, list) or not all(isinstance(item, str) for item in workflow_outputs):
        return None
    return ContinuationState(
        mode=mode,
        agent_id=agent_id,
        workflow_id=workflow_id,
        step_index=step_index,
        request_text=request_text,
        goal=goal,
        current_brief=current_brief,
        workflow_outputs=tuple(workflow_outputs),
    )


def latest_continuation(messages: tuple[ProxyInputMessage, ...]) -> ContinuationState | None:
    for message in reversed(messages):
        if message.role != "tool" or not message.tool_call_id:
            continue
        return decode_continuation_from_tool_call_id(message.tool_call_id)
    return None
