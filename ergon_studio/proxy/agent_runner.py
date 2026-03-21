from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from agent_framework import Content, Message

from ergon_studio.agent_factory import build_agent, provider_supports_tool_calling
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
from ergon_studio.proxy.tool_passthrough import (
    build_declaration_tools,
    extract_tool_calls,
)
from ergon_studio.proxy.tool_policy import resolve_agent_tool_policy
from ergon_studio.registry import RuntimeRegistry

ProxyToolChoice = str | dict[str, Any] | None


class ProxyAgentRunner:
    def __init__(
        self,
        registry: RuntimeRegistry,
        *,
        agent_builder: Callable[..., Any] = build_agent,
    ) -> None:
        self.registry = registry
        self._agent_builder = agent_builder

    async def run_text_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        preamble: str = "",
        pending_continuation: PendingContinuation | None = None,
    ) -> str | None:
        full_prompt = _merge_preamble(preamble, prompt)
        response = await self._run_agent(
            agent_id=agent_id,
            prompt=full_prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            stream=False,
            pending_continuation=pending_continuation,
        )
        if response is None:
            return None
        response_text = getattr(response, "text", response)
        if not isinstance(response_text, str):
            return None
        return response_text.strip() or None

    async def stream_text_agent(
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
        final_response_sink: Callable[[Any], None] | None = None,
    ) -> AsyncIterator[str]:
        full_prompt = _merge_preamble(preamble, prompt)
        run_result = self._run_agent(
            agent_id=agent_id,
            prompt=full_prompt,
            session_id=session_id,
            model_id_override=model_id_override,
            stream=True,
            host_tools=host_tools,
            extra_tools=extra_tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            pending_continuation=pending_continuation,
        )
        if hasattr(run_result, "__aiter__") and hasattr(
            run_result, "get_final_response"
        ):
            emitted = False
            async for update in run_result:
                delta = getattr(update, "text", "")
                if not delta:
                    continue
                emitted = True
                yield delta
            response = await run_result.get_final_response()
            if final_response_sink is not None:
                final_response_sink(response)
            final_text = getattr(response, "text", "")
            if final_text and not emitted:
                yield final_text
            return
        response = await run_result
        if final_response_sink is not None:
            final_response_sink(response)
        final_text = getattr(response, "text", "")
        if final_text:
            yield final_text

    def emit_tool_calls(
        self,
        *,
        response: Any,
        request: ProxyTurnRequest,
        continuation: ContinuationState,
    ) -> tuple[tuple[ProxyToolCall, ...], list[ProxyToolCallEvent]]:
        tool_calls = self.validate_host_tool_calls(
            extract_tool_calls(response),
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

    def _run_agent(
        self,
        *,
        agent_id: str,
        prompt: str,
        session_id: str,
        model_id_override: str,
        stream: bool,
        host_tools: tuple[ProxyFunctionTool, ...] = (),
        extra_tools: tuple[ProxyFunctionTool, ...] = (),
        tool_choice: ProxyToolChoice = None,
        parallel_tool_calls: bool | None = None,
        pending_continuation: PendingContinuation | None = None,
    ) -> Any:
        agent = self._agent_builder(
            self.registry,
            agent_id,
            model_id_override=model_id_override,
        )
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
            allowed_tools = ()
            declared_tools = ()
            tool_options.pop("tool_choice", None)
            tool_options.pop("allow_multiple_tool_calls", None)
        run_kwargs = {
            "session": agent.create_session(session_id=session_id),
            "stream": stream,
        }
        declaration_tools = build_declaration_tools(declared_tools)
        if declaration_tools:
            run_kwargs["tools"] = declaration_tools
        run_kwargs.update(tool_options)
        return agent.run(
            _build_agent_messages(
                prompt=prompt,
                pending_continuation=pending_continuation,
            ),
            **run_kwargs,
        )

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


def _merge_preamble(preamble: str, prompt: str) -> str:
    preamble = preamble.strip()
    prompt = prompt.strip()
    if preamble and prompt:
        return f"{preamble}\n\n{prompt}"
    return preamble or prompt


def _build_agent_messages(
    *, prompt: str, pending_continuation: PendingContinuation | None
) -> list[Message]:
    messages = [
        Message(
            role="user",
            text=prompt,
            author_name="proxy",
        )
    ]
    if pending_continuation is None:
        return messages

    assistant_contents: list[Content | str] = []
    assistant_message = pending_continuation.assistant_message
    if assistant_message is not None and assistant_message.content:
        assistant_contents.append(assistant_message.content)
    for tool_call in continuation_tool_calls(pending_continuation):
        assistant_contents.append(
            Content.from_function_call(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments_json,
            )
        )
    if assistant_contents:
        messages.append(
            Message(
                role="assistant",
                contents=assistant_contents,
                author_name=pending_continuation.state.agent_id,
            )
        )
    elif pending_continuation.assistant_message is None:
        synthetic_tool_calls = _synthetic_tool_calls_from_results(pending_continuation)
        if synthetic_tool_calls:
            messages.append(
                Message(
                    role="assistant",
                    contents=[
                        Content.from_function_call(
                            call_id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments_json,
                        )
                        for tool_call in synthetic_tool_calls
                    ],
                    author_name=pending_continuation.state.agent_id,
                )
            )

    for tool_call in continuation_tool_calls(pending_continuation):
        result_text = continuation_result_map(pending_continuation).get(
            tool_call.id,
            "",
        )
        messages.append(
            Message(
                role="tool",
                contents=[
                    Content.from_function_result(
                        call_id=tool_call.id,
                        result=result_text,
                    )
                ],
                author_name=tool_call.name,
            )
        )
    if pending_continuation.assistant_message is None:
        synthetic_tool_calls_by_id = {
            tool_call.id: tool_call
            for tool_call in _synthetic_tool_calls_from_results(pending_continuation)
        }
        for tool_result in pending_continuation.tool_results:
            original_call_id = (
                original_tool_call_id(tool_result.tool_call_id or "")
                or tool_result.tool_call_id
                or ""
            )
            synthetic_tool_call = synthetic_tool_calls_by_id.get(
                tool_result.tool_call_id or ""
            )
            messages.append(
                Message(
                    role="tool",
                    contents=[
                        Content.from_function_result(
                            call_id=original_call_id,
                            result=tool_result.content,
                        )
                    ],
                    author_name=synthetic_tool_call.name
                    if synthetic_tool_call is not None
                    else "host_tool",
                )
            )
    return messages


def _synthetic_tool_calls_from_results(
    pending_continuation: PendingContinuation,
) -> list[ProxyToolCall]:
    return [
        tool_call
        for tool_result in pending_continuation.tool_results
        if (tool_call := decode_original_tool_call(tool_result.tool_call_id or ""))
        is not None
    ]
