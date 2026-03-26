from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ergon_studio.debug_log import log_event
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import (
    PendingToolContext,
    continuation_result_map,
    continuation_tool_calls,
    decode_pending_id_from_tool_call_id,
    encode_continuation_tool_call,
    original_tool_call_id,
)
from ergon_studio.proxy.models import (
    ProxyFunctionTool,
    ProxyInputMessage,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.pending_store import PendingStore
from ergon_studio.proxy.tool_policy import validate_tool_choice
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream

ProxyToolChoice = str | dict[str, Any] | None


@dataclass(frozen=True)
class AgentInvocation:
    agent_id: str
    temperature: float | int | None
    max_tokens: int | None
    model: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[ProxyFunctionTool, ...]
    tool_choice: str | dict[str, Any] | None
    parallel_tool_calls: bool | None


@dataclass(frozen=True)
class AgentRunResult:
    text: str
    tool_calls: tuple[ProxyToolCall, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0


AgentInvoker = Callable[[AgentInvocation], ResponseStream[str, AgentRunResult]]


class ProxyAgentRunner:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        invoker: AgentInvoker | None = None,
        pending_store: PendingStore | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.registry = registry
        self._invoker = invoker or self._invoke_upstream
        self._pending_store = pending_store or PendingStore()
        api_key = registry.upstream.api_key or "not-needed"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=registry.upstream.base_url,
            timeout=timeout,
        )

    def stream_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        prompt_role: str = "user",
        model_id_override: str,
        conversation_messages: tuple[ProxyInputMessage, ...] = (),
        host_tools: tuple[ProxyFunctionTool, ...] = (),
        extra_tools: tuple[ProxyFunctionTool, ...] = (),
        tool_choice: ProxyToolChoice = None,
        parallel_tool_calls: bool | None = None,
        pending_continuation: PendingToolContext | None = None,
    ) -> ResponseStream[str, AgentRunResult]:
        invocation = self._build_invocation(
            agent_id=agent_id,
            prompt=prompt,
            prompt_role=prompt_role,
            model_id_override=model_id_override,
            conversation_messages=conversation_messages,
            host_tools=host_tools,
            extra_tools=extra_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            pending_continuation=pending_continuation,
        )
        log_event("agent_invocation", invocation=invocation)
        return self._invoker(invocation)

    def emit_tool_call_events(
        self,
        *,
        response: AgentRunResult | None = None,
        tool_calls: tuple[ProxyToolCall, ...] | None = None,
        request: ProxyTurnRequest,
        session_id: str,
        actor: str,
        active_channel_id: str | None = None,
        state: ProxyTurnState,
    ) -> list[ProxyToolCallEvent]:
        if tool_calls is None:
            if response is None:
                raise ValueError(
                    "response is required when tool_calls are not provided"
                )
            validated_tool_calls = self._validate_host_tool_calls(
                response.tool_calls,
                request=request,
            )
        else:
            validated_tool_calls = self._validate_host_tool_calls(
                tuple(tool_calls),
                request=request,
            )
        pending_record = self._pending_store.create(
            session_id=session_id,
            actor=actor,
            active_channel_id=active_channel_id,
            tool_calls=validated_tool_calls,
        )
        encoded_calls = tuple(
            encode_continuation_tool_call(
                tool_call,
                pending_id=pending_record.pending_id,
            )
            for tool_call in validated_tool_calls
        )
        events = [
            ProxyToolCallEvent(call=call, index=index)
            for index, call in enumerate(encoded_calls)
        ]
        if not events:
            return []
        log_event(
            "host_tool_calls_emitted",
            session_id=session_id,
            actor=actor,
            active_channel_id=active_channel_id,
            pending_id=pending_record.pending_id,
            tool_calls=encoded_calls,
        )
        state.tool_calls = encoded_calls
        state.finish_reason = "tool_calls"
        for call in encoded_calls:
            state.record_output_item("tool_call", call_id=call.id)
        return events

    def _validate_host_tool_calls(
        self,
        tool_calls: tuple[ProxyToolCall, ...],
        *,
        request: ProxyTurnRequest,
    ) -> tuple[ProxyToolCall, ...]:
        if not tool_calls:
            return ()
        available_tool_names = {tool.name for tool in request.tools}
        for tool_call in tool_calls:
            if tool_call.name not in available_tool_names:
                raise ValueError(
                    f"model requested unavailable host tool: {tool_call.name}"
                )

        tool_choice = request.tool_choice
        if tool_choice == "none":
            raise ValueError("model requested tool calls despite tool_choice='none'")
        if isinstance(tool_choice, dict):
            required_name = tool_choice["function"]["name"]
            unexpected = [
                tool_call.name
                for tool_call in tool_calls
                if tool_call.name != required_name
            ]
            if unexpected:
                raise ValueError(
                    "model requested tool calls outside required tool "
                    f"'{required_name}': {', '.join(unexpected)}"
                )
        if request.parallel_tool_calls is False and len(tool_calls) > 1:
            raise ValueError(
                "model requested multiple tool calls despite parallel_tool_calls=false"
            )
        return tool_calls

    def _build_invocation(
        self,
        *,
        agent_id: str,
        prompt: str,
        prompt_role: str,
        model_id_override: str,
        conversation_messages: tuple[ProxyInputMessage, ...],
        host_tools: tuple[ProxyFunctionTool, ...],
        extra_tools: tuple[ProxyFunctionTool, ...],
        tool_choice: ProxyToolChoice,
        parallel_tool_calls: bool | None,
        pending_continuation: PendingToolContext | None,
    ) -> AgentInvocation:
        definition = self.registry.agent_definitions[agent_id]
        resolved_tool_choice = validate_tool_choice(tool_choice, tools=host_tools)
        allowed_tools = tuple(host_tools)
        if resolved_tool_choice == "none":
            allowed_tools = ()
        elif isinstance(resolved_tool_choice, dict):
            required_name = resolved_tool_choice["function"]["name"]
            allowed_tools = tuple(
                tool for tool in host_tools if tool.name == required_name
            )
        declared_tools = tuple(allowed_tools) + tuple(extra_tools)
        if declared_tools and not self.registry.upstream.tool_calling:
            if resolved_tool_choice not in (None, "auto", "none"):
                raise ValueError(
                    f"provider for agent '{agent_id}' does not support tool calling"
                )
            declared_tools = ()
            resolved_tool_choice = None
            parallel_tool_calls = None
        instructions = compose_instructions(definition, registry=self.registry)
        messages = tuple(
            build_agent_messages(
                registry=self.registry,
                instructions=instructions,
                prompt=prompt,
                prompt_role=prompt_role,
                conversation_messages=conversation_messages,
                pending_continuation=pending_continuation,
            )
        )
        return AgentInvocation(
            agent_id=definition.id,
            temperature=_metadata_number(definition.metadata.get("temperature")),
            max_tokens=_metadata_int(definition.metadata.get("max_tokens")),
            model=model_id_override,
            messages=messages,
            tools=declared_tools,
            tool_choice=resolved_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _invoke_upstream(
        self,
        invocation: AgentInvocation,
    ) -> ResponseStream[str, AgentRunResult]:
        accumulator = _StreamAccumulator()

        async def _events():
            create_kwargs: dict[str, Any] = {
                "model": invocation.model,
                "messages": list(invocation.messages),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if invocation.temperature is not None:
                create_kwargs["temperature"] = invocation.temperature
            if invocation.max_tokens is not None:
                create_kwargs["max_tokens"] = invocation.max_tokens
            if invocation.tools:
                create_kwargs["tools"] = [
                    _tool_to_openai(tool) for tool in invocation.tools
                ]
                if invocation.tool_choice is not None:
                    create_kwargs["tool_choice"] = invocation.tool_choice
                if invocation.parallel_tool_calls is not None:
                    create_kwargs["parallel_tool_calls"] = (
                        invocation.parallel_tool_calls
                    )

            async for chunk in await self._client.chat.completions.create(
                **create_kwargs
            ):
                choices = getattr(chunk, "choices", None)
                if isinstance(choices, list):
                    for choice in choices:
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue
                        content = getattr(delta, "content", None)
                        if isinstance(content, str) and content:
                            accumulator.text += content
                            log_event(
                                "agent_delta",
                                agent_id=invocation.agent_id,
                                delta=content,
                            )
                            yield content
                        tool_deltas = getattr(delta, "tool_calls", None)
                        if isinstance(tool_deltas, list):
                            accumulator.append_tool_deltas(tool_deltas)
                            log_event(
                                "agent_tool_delta",
                                agent_id=invocation.agent_id,
                                tool_delta_count=len(tool_deltas),
                            )
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    accumulator.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    accumulator.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        def _finalize() -> AgentRunResult:
            result = AgentRunResult(
                text=accumulator.text,
                tool_calls=accumulator.tool_calls(),
                prompt_tokens=accumulator.prompt_tokens,
                completion_tokens=accumulator.completion_tokens,
            )
            log_event(
                "agent_result",
                agent_id=invocation.agent_id,
                result=result,
            )
            return result

        return ResponseStream(
            _events(),
            finalizer=_finalize,
        )
# Sections that are injected at runtime with resolved placeholders.
# They must not be included verbatim in the static instructions.
_RUNTIME_SECTIONS: frozenset[str] = frozenset({"Orchestration", "Subsession"})


def compose_instructions(
    definition: DefinitionDocument,
    *,
    registry: RuntimeRegistry | None = None,
) -> str:
    if definition.sections:
        parts: list[str] = []
        for title, content in definition.sections.items():
            if title in _RUNTIME_SECTIONS:
                continue
            parts.append(f"## {title}")
            if content:
                parts.append(content)
        base = "\n\n".join(parts).strip()
    else:
        base = definition.body.strip()
    context = _agent_profile_context(definition, registry=registry)
    if not context:
        return base
    if not base:
        return context
    return f"{base}\n\n{context}"


def build_agent_messages(
    *,
    registry: RuntimeRegistry,
    instructions: str,
    prompt: str,
    prompt_role: str = "user",
    conversation_messages: tuple[ProxyInputMessage, ...],
    pending_continuation: PendingToolContext | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instruction_role = registry.upstream.instruction_role or "system"
    messages.append(
        {
            "role": instruction_role,
            "content": instructions,
        }
    )
    if prompt:
        messages.append({"role": prompt_role, "content": prompt})
    sanitized_conversation = _strip_pending_messages(
        conversation_messages,
        pending_continuation=pending_continuation,
    )
    messages.extend(
        _openai_message_from_proxy(message) for message in sanitized_conversation
    )

    assistant_message = _pending_assistant_message(pending_continuation)
    if assistant_message is not None:
        messages.append(assistant_message)
    messages.extend(_pending_tool_messages(pending_continuation))
    return messages


def _strip_pending_messages(
    conversation_messages: tuple[ProxyInputMessage, ...],
    *,
    pending_continuation: PendingToolContext | None,
) -> tuple[ProxyInputMessage, ...]:
    if pending_continuation is None or not conversation_messages:
        return conversation_messages

    trimmed = list(conversation_messages)
    pending_id = pending_continuation.pending_id
    while trimmed and trimmed[-1].role == "tool":
        message = trimmed[-1]
        if (
            decode_pending_id_from_tool_call_id(message.tool_call_id or "")
            != pending_id
        ):
            break
        trimmed.pop()

    if trimmed:
        message = trimmed[-1]
        tool_pending_ids = {
            decode_pending_id_from_tool_call_id(tool_call.id)
            for tool_call in message.tool_calls
        }
        if message.role == "assistant" and message.tool_calls:
            if tool_pending_ids == {pending_id}:
                trimmed.pop()
            elif pending_id in tool_pending_ids:
                log_event(
                    "strip_pending_mixed_ids",
                    pending_id=pending_id,
                    all_pending_ids=list(tool_pending_ids - {None}),
                )
    return tuple(trimmed)


def _pending_assistant_message(
    pending_continuation: PendingToolContext | None,
) -> dict[str, Any] | None:
    if pending_continuation is None:
        return None

    tool_calls = [
        _tool_call_message(tool_call)
        for tool_call in continuation_tool_calls(pending_continuation)
    ]
    if not tool_calls:
        return None
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
    }
    return payload


def _pending_tool_messages(
    pending_continuation: PendingToolContext | None,
) -> list[dict[str, Any]]:
    if pending_continuation is None:
        return []

    messages: list[dict[str, Any]] = []
    result_map = continuation_result_map(pending_continuation)
    for tool_call in continuation_tool_calls(pending_continuation):
        result_text = result_map.get(tool_call.id, "")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": original_tool_call_id(tool_call.id) or tool_call.id,
                "content": result_text,
            }
        )
    return messages


def _tool_call_message(tool_call: ProxyToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments_json,
        },
    }


def _openai_message_from_proxy(message: ProxyInputMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            _tool_call_message(tool_call) for tool_call in message.tool_calls
        ]
    return payload


def _tool_to_openai(tool: ProxyFunctionTool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict:
        function["strict"] = True
    return {
        "type": "function",
        "function": function,
    }


def _metadata_number(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _metadata_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

def _agent_profile_context(
    definition: DefinitionDocument,
    *,
    registry: RuntimeRegistry | None,
) -> str:
    role = str(definition.metadata.get("role", definition.id))
    tools = definition.metadata.get("tools", [])
    tool_summary = (
        ", ".join(str(tool) for tool in tools)
        if isinstance(tools, list) and tools
        else "none"
    )
    lines = [
        f"Agent profile: {definition.id}",
        f"Role: {role}",
        f"Tools: {tool_summary}",
    ]
    if role == "orchestrator" and registry is not None:
        agent_summaries = []
        for agent_id, candidate in sorted(registry.agent_definitions.items()):
            if agent_id == definition.id:
                continue
            agent_role = str(candidate.metadata.get("role", agent_id))
            agent_summaries.append(f"{agent_id}({agent_role})")
        preset_summaries = []
        for channel_id, participants in sorted(registry.channel_presets.items()):
            participant_summary = ", ".join(participants)
            preset_summaries.append(f"{channel_id}({participant_summary})")
        lines.append(
            "Available specialists: "
            + (", ".join(agent_summaries) if agent_summaries else "none")
        )
        lines.append(
            "Available channel presets: "
            + (", ".join(preset_summaries) if preset_summaries else "none")
        )
    return "\n".join(lines)


@dataclass
class _ToolAccumulator:
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAccumulator:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _tool_calls: dict[int, _ToolAccumulator] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tool_calls = {}

    def append_tool_deltas(self, tool_deltas: list[Any]) -> None:
        for tool_delta in tool_deltas:
            index = getattr(tool_delta, "index", None)
            if not isinstance(index, int):
                continue
            accumulator = self._tool_calls.setdefault(index, _ToolAccumulator())
            tool_id = getattr(tool_delta, "id", None)
            if isinstance(tool_id, str) and tool_id:
                accumulator.call_id = tool_id
            function = getattr(tool_delta, "function", None)
            if function is None:
                continue
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                accumulator.name = name
            arguments = getattr(function, "arguments", None)
            if isinstance(arguments, str) and arguments:
                accumulator.arguments += arguments

    def tool_calls(self) -> tuple[ProxyToolCall, ...]:
        tool_calls: list[ProxyToolCall] = []
        for index in sorted(self._tool_calls):
            accumulator = self._tool_calls[index]
            if not accumulator.call_id or not accumulator.name:
                continue
            tool_calls.append(
                ProxyToolCall(
                    id=accumulator.call_id,
                    name=accumulator.name,
                    arguments_json=accumulator.arguments,
                )
            )
        return tuple(tool_calls)
