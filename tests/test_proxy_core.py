from __future__ import annotations

import unittest
from pathlib import Path

from agent_framework import ResponseStream

from ergon_studio.proxy.continuation import ContinuationState, decode_continuation_from_tool_call_id, encode_continuation_tool_call
from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyInputMessage, ProxyReasoningDeltaEvent, ProxyToolCall, ProxyToolCallEvent, ProxyTurnRequest
from ergon_studio.registry import RuntimeRegistry


class ProxyCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_handles_direct_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": ["{\"mode\":\"act\"}", "Hello world"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertTrue(any(isinstance(event, ProxyReasoningDeltaEvent) for event in events))
        self.assertEqual("".join(event.delta for event in events if isinstance(event, ProxyContentDeltaEvent)), "Hello world")
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.mode, "act")

    async def test_stream_turn_handles_delegate_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"delegate\",\"agent_id\":\"coder\",\"request\":\"Implement it\"}",
                "Final summary",
            ],
            "coder": ["Patch", " applied"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("delegating", reasoning.lower())
        self.assertIn("coder: Patch", reasoning)
        self.assertEqual(result.content, "Final summary")
        self.assertEqual(result.mode, "delegate")

    async def test_stream_turn_handles_workflow_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"standard-build\",\"goal\":\"Build calculator\"}",
                "Workflow final summary",
            ],
            "architect": ["Plan"],
            "coder": ["Built"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("workflow standard-build", reasoning)
        self.assertIn("architect: Plan", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(result.content, "Workflow final summary")
        self.assertEqual(result.mode, "workflow")

    async def test_stream_turn_emits_tool_call_events_for_direct_mode(self) -> None:
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"act\"}",
                {
                    "text": "",
                    "tool_calls": [
                        {"id": "call_1", "name": "read_file", "arguments": "{\"path\":\"main.py\"}"},
                    ],
                },
            ],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        tool_events = [event for event in events if isinstance(event, ProxyToolCallEvent)]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call.name, "read_file")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        continuation = decode_continuation_from_tool_call_id(result.tool_calls[0].id)
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation.mode, "act")

    async def test_stream_turn_resumes_workflow_from_tool_result(self) -> None:
        first_core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"standard-build\",\"goal\":\"Build calculator\"}",
            ],
            "architect": [
                {
                    "text": "",
                    "tool_calls": [
                        {"id": "call_arch_1", "name": "read_file", "arguments": "{\"path\":\"main.py\"}"},
                    ],
                }
            ],
        }))
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
            tools=(_host_tool("read_file"),),
        )
        first_stream = first_core.stream_turn(first_request, created_at=1)
        first_events = [event async for event in first_stream]
        first_result = await first_stream.get_final_response()
        tool_call = next(event.call for event in first_events if isinstance(event, ProxyToolCallEvent))

        resumed_core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "architect": ["Architecture plan"],
            "coder": ["Built feature"],
            "orchestrator": ["Workflow final summary"],
        }))
        resumed_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=(tool_call,),
                ),
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=tool_call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )
        resumed_stream = resumed_core.stream_turn(resumed_request, created_at=2)
        resumed_events = [event async for event in resumed_stream]
        resumed_result = await resumed_stream.get_final_response()

        reasoning = "".join(event.delta for event in resumed_events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("continuing workflow standard-build with architect", reasoning)
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(resumed_result.content, "Workflow final summary")
        self.assertEqual(resumed_result.finish_reason, "stop")

    async def test_stream_turn_does_not_resume_stale_tool_loop(self) -> None:
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(mode="delegate", agent_id="coder"),
            call_id="call_1",
            name="read_file",
        )
        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_fake_agent_builder({
            "orchestrator": ["{\"mode\":\"act\"}", "Fresh reply"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(role="assistant", content="", tool_calls=(tool_call,)),
                ProxyInputMessage(role="tool", content="file contents", tool_call_id=tool_call.id),
                ProxyInputMessage(role="assistant", content="That is done."),
                ProxyInputMessage(role="user", content="Now explain the design"),
            ),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("handling this turn directly", reasoning.lower())
        self.assertEqual(result.content, "Fresh reply")

    async def test_workflow_continuation_keeps_remaining_agents_in_same_group(self) -> None:
        registry = _grouped_workflow_registry()
        first_core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"grouped-build\",\"goal\":\"Build calculator\"}",
            ],
            "architect": [
                {
                    "text": "",
                    "tool_calls": [
                        {"id": "call_arch_1", "name": "read_file", "arguments": "{\"path\":\"main.py\"}"},
                    ],
                }
            ],
        }))
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
            tools=(_host_tool("read_file"),),
        )
        first_stream = first_core.stream_turn(first_request, created_at=1)
        first_events = [event async for event in first_stream]
        tool_call = next(event.call for event in first_events if isinstance(event, ProxyToolCallEvent))

        resumed_core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "architect": ["Architecture plan"],
            "coder": ["Built feature"],
            "reviewer": ["Reviewed result"],
            "orchestrator": ["Workflow final summary"],
        }))
        resumed_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(role="assistant", content="", tool_calls=(tool_call,)),
                ProxyInputMessage(role="tool", content="print('current main')", tool_call_id=tool_call.id),
            ),
            tools=(_host_tool("read_file"),),
        )

        resumed_stream = resumed_core.stream_turn(resumed_request, created_at=2)
        resumed_events = [event async for event in resumed_stream]
        resumed_result = await resumed_stream.get_final_response()

        reasoning = "".join(event.delta for event in resumed_events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertIn("reviewer: Reviewed", reasoning)
        self.assertEqual(resumed_result.content, "Workflow final summary")

    async def test_group_chat_workflow_uses_selection_sequence(self) -> None:
        registry = _advanced_workflow_registry()
        core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"debate\",\"goal\":\"Choose an approach\"}",
                "Debate final summary",
            ],
            "architect": ["Option A", "Refined option A"],
            "brainstormer": ["Option B"],
            "reviewer": ["Decision-ready recommendation"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Choose an approach"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertGreaterEqual(reasoning.count("architect:"), 2)
        self.assertIn("brainstormer: Option", reasoning)
        self.assertIn("reviewer: Decision-ready", reasoning)
        self.assertEqual(result.content, "Debate final summary")

    async def test_magentic_workflow_uses_manager_agent_selection(self) -> None:
        registry = _advanced_workflow_registry()
        core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"dynamic-open-ended\",\"goal\":\"Build it\"}",
                "{\"agent_id\":\"architect\"}",
                "{\"agent_id\":\"reviewer\"}",
                "{\"agent_id\":null}",
                "Dynamic final summary",
            ],
            "architect": ["Architecture pass"],
            "reviewer": ["Review pass"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("reviewer: Review", reasoning)
        self.assertEqual(result.content, "Dynamic final summary")

    async def test_handoff_workflow_uses_specialist_handoff_selection(self) -> None:
        registry = _advanced_workflow_registry()
        core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "orchestrator": [
                "{\"mode\":\"workflow\",\"workflow_id\":\"specialist-handoff\",\"goal\":\"Research and decide\"}",
                "Handoff final summary",
            ],
            "architect": ["Initial direction", "{\"agent_id\":\"reviewer\"}"],
            "reviewer": ["Final recommendation"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Research and decide"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(event.delta for event in events if isinstance(event, ProxyReasoningDeltaEvent))
        self.assertIn("architect: Initial", reasoning)
        self.assertIn("reviewer: Final", reasoning)
        self.assertEqual(result.content, "Handoff final summary")

    async def test_stream_turn_respects_host_tool_policy(self) -> None:
        captured: dict[str, object] = {}
        remaining = {
            "orchestrator": ["{\"mode\":\"act\"}", {"text": "", "tool_calls": []}],
        }

        class _CaptureAgent(_FakeAgent):
            def run(self, _messages, *, session, stream: bool = False, tools=None, **kwargs):
                captured["tools"] = tools
                captured["kwargs"] = kwargs
                return super().run(_messages, session=session, stream=stream, tools=tools, **kwargs)

        def _builder(_registry, agent_id: str, **_kwargs):
            return _CaptureAgent([remaining[agent_id].pop(0)])

        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_builder)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"), _host_tool("write_file")),
            tool_choice={"type": "function", "function": {"name": "write_file"}},
            parallel_tool_calls=False,
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        tools = captured["tools"]
        kwargs = captured["kwargs"]
        self.assertEqual([tool.name for tool in tools], ["write_file"])
        self.assertEqual(kwargs["tool_choice"], {"mode": "required", "required_function_name": "write_file"})
        self.assertFalse(kwargs["allow_multiple_tool_calls"])

    async def test_stream_turn_strips_optional_tools_when_provider_cannot_call_tools(self) -> None:
        captured: dict[str, object] = {}
        remaining = {
            "orchestrator": ["{\"mode\":\"act\"}", {"text": "Done", "tool_calls": []}],
        }

        class _CaptureAgent(_FakeAgent):
            def run(self, _messages, *, session, stream: bool = False, tools=None, **kwargs):
                captured["tools"] = tools
                captured["kwargs"] = kwargs
                return super().run(_messages, session=session, stream=stream, tools=tools, **kwargs)

        def _builder(_registry, agent_id: str, **_kwargs):
            return _CaptureAgent([remaining[agent_id].pop(0)])

        registry = _provider_registry(tool_calling=False)
        core = ProxyOrchestrationCore(registry, agent_builder=_builder)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
            tool_choice="auto",
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        self.assertIsNone(captured["tools"])
        self.assertNotIn("tool_choice", captured["kwargs"])

    async def test_stream_turn_errors_when_required_tool_choice_hits_toolless_provider(self) -> None:
        registry = _provider_registry(tool_calling=False)
        core = ProxyOrchestrationCore(registry, agent_builder=_fake_agent_builder({
            "orchestrator": ["{\"mode\":\"act\"}", "unused"],
        }))
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
            tool_choice="required",
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn("does not support tool calling", result.content)
        self.assertTrue(any(isinstance(event, ProxyContentDeltaEvent) for event in events))

    async def test_stream_turn_rebuilds_structured_tool_history_for_continuations(self) -> None:
        captured: dict[str, object] = {}
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(mode="act", agent_id="orchestrator"),
            call_id="call_1",
            name="read_file",
        )

        class _CaptureAgent(_FakeAgent):
            def run(self, messages, *, session, stream: bool = False, tools=None, **kwargs):
                captured["messages"] = messages
                return super().run(messages, session=session, stream=stream, tools=tools, **kwargs)

        def _builder(_registry, agent_id: str, **_kwargs):
            if agent_id != "orchestrator":
                raise AssertionError(f"unexpected agent: {agent_id}")
            return _CaptureAgent(["Final answer"])

        core = ProxyOrchestrationCore(_provider_registry(tool_calling=True), agent_builder=_builder)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect main.py"),
                ProxyInputMessage(role="assistant", content="", tool_calls=(tool_call,)),
                ProxyInputMessage(role="tool", content="print('current main')", tool_call_id=tool_call.id),
            ),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        messages = captured["messages"]
        self.assertEqual([message.role for message in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1].contents[0].type, "function_call")
        self.assertEqual(messages[2].contents[0].type, "function_result")


class _FakeRegistry:
    def __init__(self) -> None:
        self.inner = RuntimeRegistry(
            config={},
            agent_definitions={
                "orchestrator": DefinitionDocument(
                    id="orchestrator",
                    path=Path("orchestrator.md"),
                    metadata={"id": "orchestrator", "role": "orchestrator"},
                    body="## Identity\nLead engineer.",
                    sections={"Identity": "Lead engineer."},
                ),
                "architect": DefinitionDocument(
                    id="architect",
                    path=Path("architect.md"),
                    metadata={"id": "architect", "role": "architect"},
                    body="## Identity\nArchitect.",
                    sections={"Identity": "Architect."},
                ),
                "coder": DefinitionDocument(
                    id="coder",
                    path=Path("coder.md"),
                    metadata={"id": "coder", "role": "coder"},
                    body="## Identity\nCoder.",
                    sections={"Identity": "Coder."},
                ),
            },
            workflow_definitions={
                "standard-build": DefinitionDocument(
                    id="standard-build",
                    path=Path("standard-build.md"),
                    metadata={
                        "id": "standard-build",
                        "orchestration": "sequential",
                        "steps": ["architect", "coder"],
                    },
                    body="## Purpose\nBuild.",
                    sections={"Purpose": "Build."},
                )
            },
        )

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class _FakeAgent:
    def __init__(self, responses) -> None:
        self._responses = list(responses)

    def create_session(self, *, session_id: str):
        return object()

    def run(self, _messages, *, session, stream: bool = False, tools=None, **_kwargs):
        raw = self._responses.pop(0)
        if isinstance(raw, str):
            payload = {"text": raw, "tool_calls": []}
        else:
            payload = raw
        text = payload.get("text", "")
        tool_calls = payload.get("tool_calls", [])
        if not stream:
            return _immediate_response(text, tool_calls=tool_calls)
        parts = [piece for piece in text.split(" ") if piece]

        async def _events():
            for index, part in enumerate(parts):
                suffix = " " if index < len(parts) - 1 else ""
                yield type("Update", (), {"text": part + suffix})()

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: _response_object(text, tool_calls=tool_calls),
        )


def _fake_agent_builder(mapping: dict[str, list[str]]):
    remaining = {agent_id: list(responses) for agent_id, responses in mapping.items()}

    def _build(_registry, agent_id: str, **_kwargs):
        queue = remaining[agent_id]
        if not queue:
            raise AssertionError(f"no fake responses left for {agent_id}")
        return _FakeAgent([queue.pop(0)])

    return _build


def _fake_registry():
    return _FakeRegistry()


def _grouped_workflow_registry():
    registry = _FakeRegistry()
    registry.workflow_definitions["grouped-build"] = DefinitionDocument(
        id="grouped-build",
        path=Path("grouped-build.md"),
        metadata={
            "id": "grouped-build",
            "orchestration": "concurrent",
            "step_groups": [["architect", "coder", "reviewer"]],
        },
        body="## Purpose\nGrouped build.",
        sections={"Purpose": "Grouped build."},
    )
    return registry


def _provider_registry(*, tool_calling: bool) -> RuntimeRegistry:
    return RuntimeRegistry(
        config={
            "providers": {
                "local": {
                    "type": "openai_chat",
                    "model": "ergon",
                    "capabilities": {"tool_calling": tool_calling},
                }
            },
            "role_assignments": {
                "orchestrator": "local",
            },
        },
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
        },
        workflow_definitions={},
    )


def _advanced_workflow_registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        config={},
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
            "brainstormer": DefinitionDocument(
                id="brainstormer",
                path=Path("brainstormer.md"),
                metadata={"id": "brainstormer", "role": "brainstormer"},
                body="## Identity\nBrainstormer.",
                sections={"Identity": "Brainstormer."},
            ),
            "reviewer": DefinitionDocument(
                id="reviewer",
                path=Path("reviewer.md"),
                metadata={"id": "reviewer", "role": "reviewer"},
                body="## Identity\nReviewer.",
                sections={"Identity": "Reviewer."},
            ),
        },
        workflow_definitions={
            "debate": DefinitionDocument(
                id="debate",
                path=Path("debate.md"),
                metadata={
                    "id": "debate",
                    "orchestration": "group_chat",
                    "step_groups": [["architect", "brainstormer", "reviewer"]],
                    "selection_sequence": ["architect", "brainstormer", "architect", "reviewer"],
                    "max_rounds": 4,
                },
                body="## Purpose\nDebate.",
                sections={"Purpose": "Debate."},
            ),
            "dynamic-open-ended": DefinitionDocument(
                id="dynamic-open-ended",
                path=Path("dynamic-open-ended.md"),
                metadata={
                    "id": "dynamic-open-ended",
                    "orchestration": "magentic",
                    "step_groups": [["architect", "reviewer"]],
                    "max_rounds": 3,
                },
                body="## Purpose\nAdaptive.",
                sections={"Purpose": "Adaptive."},
            ),
            "specialist-handoff": DefinitionDocument(
                id="specialist-handoff",
                path=Path("specialist-handoff.md"),
                metadata={
                    "id": "specialist-handoff",
                    "orchestration": "handoff",
                    "step_groups": [["architect", "reviewer"]],
                    "start_agent": "architect",
                    "finalizers": ["reviewer"],
                    "handoffs": {"architect": ["reviewer"]},
                    "max_rounds": 3,
                },
                body="## Purpose\nHandoff.",
                sections={"Purpose": "Handoff."},
            ),
        },
    )


def _response_object(text: str, *, tool_calls: list[dict[str, str]]):
    contents = [type("Content", (), {"type": "text", "text": text})()] if text else []
    for tool_call in tool_calls:
        contents.append(
            type(
                "Content",
                (),
                {
                    "type": "function_call",
                    "call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "arguments": tool_call["arguments"],
                },
            )()
        )
    message = type("Message", (), {"contents": contents})()
    return type("Response", (), {"text": text, "messages": [message]})()


async def _immediate_response(text: str, *, tool_calls: list[dict[str, str]]):
    return _response_object(text, tool_calls=tool_calls)


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )


def _host_continuation_tool_call(*, state: ContinuationState, call_id: str, name: str) -> ProxyToolCall:
    return encode_continuation_tool_call(
        ProxyToolCall(id=call_id, name=name, arguments_json="{}"),
        state=state,
    )


if __name__ == "__main__":
    unittest.main()
