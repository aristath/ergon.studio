from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from ergon_studio.definitions import format_definition_section
from ergon_studio.proxy.agent_runner import AgentRunResult
from ergon_studio.proxy.models import (
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
)
from ergon_studio.proxy.session_overlay import SessionOverlay
from ergon_studio.proxy.workspace_tools import (
    build_workspace_tools,
    parse_list_files_action,
    parse_read_file_action,
    parse_write_file_action,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream

MAX_SUBSESSION_ITERATIONS = 50


class SubSessionExecutor:
    """Run an agent in an isolated sub-session with a workspace file overlay.

    The agent may call ``read_file``, ``write_file``, and ``list_files`` tools;
    these are handled internally against the overlay.  No host tools are passed
    to the agent.  Each agent text chunk is yielded as a
    ``ProxyReasoningDeltaEvent``.  The final response text is the stream's
    terminal value.
    """

    def __init__(
        self,
        *,
        stream_text_agent: Callable[..., ResponseStream[str, AgentRunResult]],
        registry: RuntimeRegistry,
    ) -> None:
        self._stream_text_agent = stream_text_agent
        self._registry = registry

    def execute(
        self,
        *,
        agent_id: str,
        task: str,
        session_id: str,
        session_index: int,
        overlay: SessionOverlay,
        model_id: str,
    ) -> ResponseStream[ProxyReasoningDeltaEvent, str]:
        final_text: list[str] = []
        workspace_tools = build_workspace_tools()
        workspace = f"/workspace/{session_index}"
        definition = self._registry.agent_definitions.get(agent_id)
        framing = (
            format_definition_section(definition, "Subsession", workspace=workspace)
            if definition is not None
            else ""
        )

        async def _events() -> AsyncIterator[ProxyReasoningDeltaEvent]:
            # Start the conversation with the task as the first user message so
            # the framing prompt stays as a separate system injection.
            conversation: list[ProxyInputMessage] = [
                ProxyInputMessage(role="user", content=task)
            ]
            iteration = 0
            while True:
                if iteration >= MAX_SUBSESSION_ITERATIONS:
                    raise ValueError(
                        f"sub-session for {agent_id!r} exceeded "
                        f"{MAX_SUBSESSION_ITERATIONS} iterations without "
                        "producing a final text response"
                    )
                iteration += 1
                stream = self._stream_text_agent(
                    agent_id=agent_id,
                    prompt=framing,
                    prompt_role="system",
                    model_id_override=model_id,
                    conversation_messages=tuple(conversation),
                    host_tools=(),
                    extra_tools=workspace_tools,
                )
                async for delta in stream:
                    if delta:
                        yield ProxyReasoningDeltaEvent(delta)
                result = await stream.get_final_response()

                # Record this assistant turn (with tool_calls if present)
                conversation.append(
                    ProxyInputMessage(
                        role="assistant",
                        content=result.text,
                        tool_calls=result.tool_calls,
                    )
                )

                if not result.tool_calls:
                    final_text.append(result.text)
                    break

                # Execute each workspace tool call and feed results back
                for tool_call in result.tool_calls:
                    tool_result = _execute_workspace_tool(tool_call, overlay)
                    conversation.append(
                        ProxyInputMessage(
                            role="tool",
                            content=tool_result,
                            tool_call_id=tool_call.id,
                        )
                    )

                # Safety net: if the agent produced non-empty text alongside
                # tool calls, treat this as the final response after executing
                # the tools.  This breaks fixed-point loops where the model
                # simultaneously summarises completed work and re-calls a tool.
                if result.text.strip():
                    final_text.append(result.text)
                    break

        return ResponseStream(
            _events(),
            finalizer=lambda: final_text[0] if final_text else "",
        )


def _execute_workspace_tool(
    tool_call: ProxyToolCall, overlay: SessionOverlay
) -> str:
    if tool_call.name == "read_file":
        read_action = parse_read_file_action(tool_call)
        try:
            return overlay.read_file(read_action.path)
        except FileNotFoundError:
            return f"Error: file not found: {read_action.path}"
    if tool_call.name == "write_file":
        write_action = parse_write_file_action(tool_call)
        overlay.write_file(write_action.path, write_action.content)
        return f"Written: {write_action.path}"
    if tool_call.name == "list_files":
        list_action = parse_list_files_action(tool_call)
        files = overlay.list_files(list_action.directory)
        return "\n".join(files) if files else "(empty directory)"
    return f"Error: unknown workspace tool: {tool_call.name}"
