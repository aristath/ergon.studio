from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import (
    ContinuationState,
    PendingContinuation,
    continuation_result_map,
    continuation_tool_calls,
    decode_original_tool_call,
    encode_continuation_tool_call,
    original_tool_call_id,
)
from ergon_studio.proxy.models import (
    ProxyFunctionTool,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.proxy.tool_policy import resolve_agent_tool_policy
from ergon_studio.proxy.turn_state import ProxyTurnState
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream

ProxyToolChoice = str | dict[str, Any] | None


@dataclass(frozen=True)
class RuntimeAgent:
    id: str
    name: str
    role: str
    instructions: str
    temperature: float | int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class AgentInvocation:
    agent: RuntimeAgent
    model: str
    session_id: str
    prompt: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[ProxyFunctionTool, ...]
    tool_choice: str | dict[str, Any] | None
    parallel_tool_calls: bool | None


@dataclass(frozen=True)
class AgentRunResult:
    text: str
    tool_calls: tuple[ProxyToolCall, ...]


AgentInvoker = Callable[[AgentInvocation], ResponseStream[str, AgentRunResult]]


class ProxyAgentRunner:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        invoker: AgentInvoker | None = None,
    ) -> None:
        self.registry = registry
        self._invoker = invoker or self._invoke_upstream
        api_key = registry.upstream.api_key or "not-needed"
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=registry.upstream.base_url,
        )

    def stream_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        preamble: str = "",
        host_tools: tuple[ProxyFunctionTool, ...] = (),
        extra_tools: tuple[ProxyFunctionTool, ...] = (),
        tool_choice: ProxyToolChoice = None,
        parallel_tool_calls: bool | None = None,
        pending_continuation: PendingContinuation | None = None,
    ) -> ResponseStream[str, AgentRunResult]:
        invocation = self._build_invocation(
            agent_id=agent_id,
            prompt=_merge_preamble(preamble, prompt),
            session_id=session_id,
            model_id_override=model_id_override,
            host_tools=host_tools,
            extra_tools=extra_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            pending_continuation=pending_continuation,
        )
        return self._invoker(invocation)

    def emit_tool_calls(
        self,
        *,
        response: AgentRunResult,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
    ) -> tuple[tuple[ProxyToolCall, ...], list[ProxyToolCallEvent]]:
        tool_calls = self.validate_host_tool_calls(
            response.tool_calls,
            request=request,
        )
        if not tool_calls:
            return (), []
        encoded_calls = tuple(
            encode_continuation_tool_call(tool_call, state=continuation)
            for tool_call in tool_calls
        )
        return encoded_calls, [
            ProxyToolCallEvent(call=call, index=index)
            for index, call in enumerate(encoded_calls)
        ]

    def emit_tool_call_events(
        self,
        *,
        response: AgentRunResult | None = None,
        tool_calls: tuple[ProxyToolCall, ...] | None = None,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
        state: ProxyTurnState,
    ) -> list[ProxyToolCallEvent]:
        if tool_calls is None:
            if response is None:
                raise ValueError(
                    "response is required when tool_calls are not provided"
                )
            encoded_calls, events = self.emit_tool_calls(
                response=response,
                request=request,
                continuation=continuation,
            )
        else:
            validated_tool_calls = self.validate_host_tool_calls(
                tuple(tool_calls),
                request=request,
            )
            encoded_calls = tuple(
                encode_continuation_tool_call(tool_call, state=continuation)
                for tool_call in validated_tool_calls
            )
            events = [
                ProxyToolCallEvent(call=call, index=index)
                for index, call in enumerate(encoded_calls)
            ]
        if not events:
            return []
        state.tool_calls = encoded_calls
        state.finish_reason = "tool_calls"
        for call in encoded_calls:
            state.record_output_item("tool_call", call_id=call.id)
        return events

    def validate_host_tool_calls(
        self,
        tool_calls: tuple[ProxyToolCall, ...],
        *,
        request: ProxyTurnRequest,
    ) -> tuple[ProxyToolCall, ...]:
        return self._validated_tool_calls(tool_calls, request=request)

    def _validated_tool_calls(
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
        session_id: str,
        model_id_override: str,
        host_tools: tuple[ProxyFunctionTool, ...],
        extra_tools: tuple[ProxyFunctionTool, ...],
        tool_choice: ProxyToolChoice,
        parallel_tool_calls: bool | None,
        pending_continuation: PendingContinuation | None,
    ) -> AgentInvocation:
        agent = build_runtime_agent(self.registry, agent_id)
        allowed_tools, tool_options = resolve_agent_tool_policy(
            tools=tuple(host_tools),
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        declared_tools = tuple(allowed_tools) + tuple(extra_tools)
        if declared_tools and not provider_supports_tool_calling(self.registry):
            if tool_options.get("tool_choice") not in (None, "auto", "none"):
                raise ValueError(
                    f"provider for agent '{agent_id}' does not support tool calling"
                )
            declared_tools = ()
            tool_options = {}
        translated_tool_choice = _translate_tool_choice(tool_options.get("tool_choice"))
        translated_parallel = _translate_parallel_tool_calls(tool_options)
        messages = tuple(
            build_agent_messages(
                registry=self.registry,
                agent=agent,
                prompt=prompt,
                pending_continuation=pending_continuation,
            )
        )
        return AgentInvocation(
            agent=agent,
            model=model_id_override,
            session_id=session_id,
            prompt=prompt,
            messages=messages,
            tools=declared_tools,
            tool_choice=translated_tool_choice,
            parallel_tool_calls=translated_parallel,
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
            if invocation.agent.temperature is not None:
                create_kwargs["temperature"] = invocation.agent.temperature
            if invocation.agent.max_tokens is not None:
                create_kwargs["max_tokens"] = invocation.agent.max_tokens
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
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    content = getattr(delta, "content", None)
                    if isinstance(content, str) and content:
                        accumulator.text += content
                        yield content
                    tool_deltas = getattr(delta, "tool_calls", None)
                    if isinstance(tool_deltas, list):
                        accumulator.append_tool_deltas(tool_deltas)

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: AgentRunResult(
                text=accumulator.text,
                tool_calls=accumulator.tool_calls(),
            ),
        )


def build_runtime_agent(
    registry: RuntimeRegistry,
    agent_id: str,
) -> RuntimeAgent:
    definition = registry.agent_definitions[agent_id]
    role = str(definition.metadata.get("role", definition.id))
    return RuntimeAgent(
        id=definition.id,
        name=str(definition.metadata.get("name", definition.id)),
        role=role,
        instructions=compose_instructions(definition, registry=registry),
        temperature=_metadata_number(definition.metadata.get("temperature")),
        max_tokens=_metadata_int(definition.metadata.get("max_tokens")),
    )


def compose_instructions(
    definition: DefinitionDocument,
    *,
    registry: RuntimeRegistry | None = None,
) -> str:
    if definition.sections:
        parts: list[str] = []
        for title, content in definition.sections.items():
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


def provider_supports_tool_calling(registry: RuntimeRegistry) -> bool:
    return registry.upstream.tool_calling


def build_agent_messages(
    *,
    registry: RuntimeRegistry,
    agent: RuntimeAgent,
    prompt: str,
    pending_continuation: PendingContinuation | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instruction_role = registry.upstream.instruction_role or "system"
    messages.append(
        {
            "role": instruction_role,
            "content": agent.instructions,
        }
    )

    assistant_message = _pending_assistant_message(pending_continuation)
    if assistant_message is not None:
        messages.append(assistant_message)
    messages.extend(_pending_tool_messages(pending_continuation))
    messages.append({"role": "user", "content": prompt})
    return messages


def _pending_assistant_message(
    pending_continuation: PendingContinuation | None,
) -> dict[str, Any] | None:
    if pending_continuation is None:
        return None

    assistant_content = ""
    assistant_message = pending_continuation.assistant_message
    if assistant_message is not None and assistant_message.content:
        assistant_content = assistant_message.content

    tool_calls = [
        _tool_call_message(tool_call)
        for tool_call in continuation_tool_calls(pending_continuation)
    ]
    if not tool_calls and assistant_message is None:
        tool_calls = [
            _tool_call_message(tool_call)
            for tool_call in _synthetic_tool_calls_from_results(pending_continuation)
        ]
    if not assistant_content and not tool_calls:
        return None
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
    }
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _pending_tool_messages(
    pending_continuation: PendingContinuation | None,
) -> list[dict[str, Any]]:
    if pending_continuation is None:
        return []

    messages: list[dict[str, Any]] = []
    for tool_call in continuation_tool_calls(pending_continuation):
        result_text = continuation_result_map(pending_continuation).get(
            tool_call.id,
            "",
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            }
        )
    if pending_continuation.assistant_message is None:
        for tool_result in pending_continuation.tool_results:
            original_call_id = (
                original_tool_call_id(tool_result.tool_call_id or "")
                or tool_result.tool_call_id
                or ""
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": original_call_id,
                    "content": tool_result.content,
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


def _translate_tool_choice(
    tool_choice: object,
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str) and tool_choice in {
        "auto",
        "required",
        "none",
    }:
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("mode") != "required":
        return None
    function_name = tool_choice.get("required_function_name")
    if not isinstance(function_name, str) or not function_name:
        return None
    return {
        "type": "function",
        "function": {"name": function_name},
    }


def _translate_parallel_tool_calls(tool_options: dict[str, Any]) -> bool | None:
    value = tool_options.get("allow_multiple_tool_calls")
    if type(value) is bool:
        return value
    return None


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


def _merge_preamble(preamble: str, prompt: str) -> str:
    preamble = preamble.strip()
    prompt = prompt.strip()
    if preamble and prompt:
        return f"{preamble}\n\n{prompt}"
    return preamble or prompt


def _synthetic_tool_calls_from_results(
    pending_continuation: PendingContinuation,
) -> list[ProxyToolCall]:
    return [
        tool_call
        for tool_result in pending_continuation.tool_results
        if (tool_call := decode_original_tool_call(tool_result.tool_call_id or ""))
        is not None
    ]


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
        workroom_summaries = []
        for workroom_id, participants in sorted(registry.workroom_definitions.items()):
            participant_summary = ", ".join(participants)
            workroom_summaries.append(f"{workroom_id}({participant_summary})")
        lines.append(
            "Available specialists: "
            + (", ".join(agent_summaries) if agent_summaries else "none")
        )
        lines.append(
            "Available workroom presets: "
            + (", ".join(workroom_summaries) if workroom_summaries else "none")
        )
    return "\n".join(lines)


@dataclass
class _ToolAccumulator:
    index: int
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAccumulator:
    text: str = ""
    _tool_calls: dict[int, _ToolAccumulator] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tool_calls = {}

    def append_tool_deltas(self, tool_deltas: list[Any]) -> None:
        for tool_delta in tool_deltas:
            index = getattr(tool_delta, "index", None)
            if not isinstance(index, int):
                continue
            accumulator = self._tool_calls.setdefault(index, _ToolAccumulator(index))
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
