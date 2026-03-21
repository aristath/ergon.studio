from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent_framework import ResponseStream

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import (
    ContinuationState,
    decode_continuation_from_tool_call_id,
    encode_continuation_tool_call,
)
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnRequest,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class ProxyCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_replies_directly_when_orchestrator_just_answers(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": ["Hello world"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
            "Hello world",
        )
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.mode, "orchestrator")
        self.assertFalse(
            any(isinstance(event, ProxyReasoningDeltaEvent) for event in events)
        )

    async def test_stream_turn_converts_unexpected_exceptions_to_error_results(
        self,
    ) -> None:
        def _builder(_registry, _agent_id: str, **_kwargs):
            raise RuntimeError("core exploded")

        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_builder)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertEqual(result.content, "RuntimeError: core exploded")
        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
            "RuntimeError: core exploded",
        )

    async def test_stream_turn_passes_requested_model_to_agent_builder(self) -> None:
        captured: dict[str, object] = {}

        def _builder(_registry, agent_id: str, **kwargs):
            captured["agent_id"] = agent_id
            captured["model_id_override"] = kwargs.get("model_id_override")
            return _FakeAgent(["Hello world"])

        core = ProxyOrchestrationCore(_fake_registry(), agent_builder=_builder)
        request = ProxyTurnRequest(
            model="gpt-oss-20b",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(captured["agent_id"], "orchestrator")
        self.assertEqual(captured["model_id_override"], "gpt-oss-20b")

    async def test_stream_turn_opens_single_person_workroom_and_then_replies(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            participants=["coder"],
                            message="Implement it",
                        ),
                        "Final summary",
                    ],
                    "coder": ["Patch", " applied"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("opening an ad hoc workroom", reasoning.lower())
        self.assertIn("coder: Patch", reasoning)
        self.assertEqual(result.content, "Final summary")
        self.assertEqual(result.mode, "orchestrator")

    async def test_stream_turn_handles_workroom_rounds(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="standard-build",
                            message="Build calculator",
                        ),
                        _internal_action(
                            "continue_workroom",
                            message="Implement the approved plan",
                        ),
                        "Workroom final summary",
                    ],
                    "architect": ["Plan"],
                    "coder": ["Built"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("workroom standard-build", reasoning)
        self.assertIn("architect: Plan", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(result.content, "Workroom final summary")
        self.assertEqual(result.mode, "orchestrator")

    async def test_stream_turn_workroom_can_staff_specific_specialists(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="standard-build",
                            participants=["coder"],
                            message="Build calculator",
                        ),
                        "Workroom final summary",
                    ],
                    "coder": ["Built"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertNotIn("architect:", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(result.content, "Workroom final summary")

    async def test_stream_turn_emits_tool_call_events_for_direct_mode(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            ],
                        },
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        tool_events = [
            event for event in events if isinstance(event, ProxyToolCallEvent)
        ]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call.name, "read_file")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        continuation = decode_continuation_from_tool_call_id(result.tool_calls[0].id)
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation.mode, "orchestrator")

    async def test_stream_turn_resumes_workroom_from_tool_result(self) -> None:
        first_core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="standard-build",
                            message="Build calculator",
                        ),
                    ],
                    "architect": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_arch_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            ],
                        }
                    ],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
            tools=(_host_tool("read_file"),),
        )
        first_stream = first_core.stream_turn(first_request, created_at=1)
        first_events = [event async for event in first_stream]
        await first_stream.get_final_response()
        tool_call = next(
            event.call
            for event in first_events
            if isinstance(event, ProxyToolCallEvent)
        )

        resumed_core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "architect": ["Architecture plan"],
                    "coder": ["Built feature"],
                    "orchestrator": [
                        _internal_action(
                            "continue_workroom",
                            message="Continue from the architecture plan",
                        ),
                        "Workroom final summary",
                    ],
                }
            ),
        )
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

        reasoning = "".join(
            event.delta
            for event in resumed_events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("continuing workroom standard-build with architect", reasoning)
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertEqual(resumed_result.content, "Workroom final summary")
        self.assertEqual(resumed_result.finish_reason, "stop")

    async def test_stream_turn_does_not_resume_stale_tool_loop(self) -> None:
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(
                mode="workroom",
                agent_id="coder",
                workroom_id="__ad_hoc__",
                workroom_participants=("coder",),
            ),
            call_id="call_1",
            name="read_file",
        )
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": ["Fresh reply"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(
                    role="assistant", content="", tool_calls=(tool_call,)
                ),
                ProxyInputMessage(
                    role="tool", content="file contents", tool_call_id=tool_call.id
                ),
                ProxyInputMessage(role="assistant", content="That is done."),
                ProxyInputMessage(role="user", content="Now explain the design"),
            ),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(result.content, "Fresh reply")

    async def test_workroom_continuation_keeps_remaining_agents_in_same_group(
        self,
    ) -> None:
        registry = _staged_workroom_registry()
        first_core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="staged-build",
                            message="Build calculator",
                        ),
                    ],
                    "architect": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_arch_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            ],
                        }
                    ],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build calculator"),),
            tools=(_host_tool("read_file"),),
        )
        first_stream = first_core.stream_turn(first_request, created_at=1)
        first_events = [event async for event in first_stream]
        tool_call = next(
            event.call
            for event in first_events
            if isinstance(event, ProxyToolCallEvent)
        )

        resumed_core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "architect": ["Architecture plan"],
                    "coder": ["Built feature"],
                    "reviewer": ["Reviewed result"],
                    "orchestrator": ["Workroom final summary"],
                }
            ),
        )
        resumed_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Build calculator"),
                ProxyInputMessage(
                    role="assistant", content="", tool_calls=(tool_call,)
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

        reasoning = "".join(
            event.delta
            for event in resumed_events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("coder: Built", reasoning)
        self.assertIn("reviewer: Reviewed", reasoning)
        self.assertEqual(resumed_result.content, "Workroom final summary")

    async def test_discussion_workroom_uses_turns_as_turn_order(self) -> None:
        registry = _advanced_workroom_registry()
        core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="debate",
                            message="Choose an approach",
                        ),
                        "Debate final summary",
                    ],
                    "architect": ["Option A", "Refined option A"],
                    "brainstormer": ["Option B"],
                    "reviewer": ["Decision-ready recommendation"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Choose an approach"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertGreaterEqual(reasoning.count("architect:"), 2)
        self.assertIn("brainstormer: Option", reasoning)
        self.assertIn("reviewer: Decision-ready", reasoning)
        self.assertEqual(result.content, "Debate final summary")

    async def test_dynamic_workroom_runs_as_discussion_room(self) -> None:
        registry = _advanced_workroom_registry()
        core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="dynamic-open-ended",
                            message="Build it",
                        ),
                        "Dynamic final summary",
                    ],
                    "architect": ["Architecture pass"],
                    "reviewer": ["Review pass"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("architect: Architecture", reasoning)
        self.assertIn("reviewer: Review", reasoning)
        self.assertEqual(result.content, "Dynamic final summary")

    async def test_handoff_chain_runs_as_staged_room(self) -> None:
        registry = _advanced_workroom_registry()
        core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_workroom",
                            workroom_id="handoff-chain",
                            message="Research and decide",
                        ),
                        _internal_action(
                            "continue_workroom",
                            message=(
                                "Continue the staged handoff toward a "
                                "recommendation"
                            ),
                        ),
                        "Handoff final summary",
                    ],
                    "researcher": ["Initial direction"],
                    "reviewer": ["Final recommendation"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Research and decide"),),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("researcher: Initial", reasoning)
        self.assertIn("reviewer: Final", reasoning)
        self.assertEqual(result.content, "Handoff final summary")

    async def test_stream_turn_respects_host_tool_policy(self) -> None:
        captured: dict[str, object] = {}
        remaining = {
            "orchestrator": [{"text": "", "tool_calls": []}],
        }

        class _CaptureAgent(_FakeAgent):
            def run(
                self, _messages, *, session, stream: bool = False, tools=None, **kwargs
            ):
                captured["tools"] = tools
                captured["kwargs"] = kwargs
                return super().run(
                    _messages, session=session, stream=stream, tools=tools, **kwargs
                )

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
        self.assertEqual(
            [tool.name for tool in tools],
            ["write_file", "open_workroom"],
        )
        self.assertEqual(
            kwargs["tool_choice"],
            {"mode": "required", "required_function_name": "write_file"},
        )
        self.assertFalse(kwargs["allow_multiple_tool_calls"])

    async def test_stream_turn_strips_optional_tools_when_provider_cannot_call_tools(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        remaining = {
            "orchestrator": ["Done"],
        }

        class _CaptureAgent(_FakeAgent):
            def run(
                self, _messages, *, session, stream: bool = False, tools=None, **kwargs
            ):
                captured["tools"] = tools
                captured["kwargs"] = kwargs
                return super().run(
                    _messages, session=session, stream=stream, tools=tools, **kwargs
                )

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

    async def test_stream_turn_errors_when_required_tool_choice_hits_toolless_provider(
        self,
    ) -> None:
        registry = _provider_registry(tool_calling=False)
        core = ProxyOrchestrationCore(
            registry,
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": ["unused"],
                }
            ),
        )
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
        self.assertTrue(
            any(isinstance(event, ProxyContentDeltaEvent) for event in events)
        )

    async def test_stream_turn_errors_when_model_ignores_required_tool_choice(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "write_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                            ],
                        },
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"), _host_tool("write_file")),
            tool_choice={"type": "function", "function": {"name": "read_file"}},
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn("outside required tool 'read_file'", result.content)
        self.assertTrue(
            any(isinstance(event, ProxyContentDeltaEvent) for event in events)
        )

    async def test_stream_turn_errors_when_model_ignores_parallel_tool_call_limit(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_builder=_fake_agent_builder(
                {
                    "orchestrator": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                },
                                {
                                    "id": "call_2",
                                    "name": "read_file",
                                    "arguments": '{"path":"other.py"}',
                                },
                            ],
                        },
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect the files"),),
            tools=(_host_tool("read_file"),),
            parallel_tool_calls=False,
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn(
            "multiple tool calls despite parallel_tool_calls=false", result.content
        )
        self.assertTrue(
            any(isinstance(event, ProxyContentDeltaEvent) for event in events)
        )

    async def test_stream_turn_rebuilds_structured_tool_history_for_continuations(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(mode="orchestrator", agent_id="orchestrator"),
            call_id="call_1",
            name="read_file",
        )

        class _CaptureAgent(_FakeAgent):
            def run(
                self, messages, *, session, stream: bool = False, tools=None, **kwargs
            ):
                captured["messages"] = messages
                return super().run(
                    messages, session=session, stream=stream, tools=tools, **kwargs
                )

        def _builder(_registry, agent_id: str, **_kwargs):
            if agent_id != "orchestrator":
                raise AssertionError(f"unexpected agent: {agent_id}")
            return _CaptureAgent(["Final answer"])

        core = ProxyOrchestrationCore(
            _provider_registry(tool_calling=True), agent_builder=_builder
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect main.py"),
                ProxyInputMessage(
                    role="assistant", content="", tool_calls=(tool_call,)
                ),
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=tool_call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        messages = captured["messages"]
        self.assertEqual(
            [message.role for message in messages], ["user", "assistant", "tool"]
        )
        self.assertEqual(messages[1].contents[0].type, "function_call")
        self.assertEqual(messages[2].contents[0].type, "function_result")

    async def test_stream_turn_rebuilds_tool_result_without_assistant_call_history(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(mode="orchestrator", agent_id="orchestrator"),
            call_id="call_1",
            name="read_file",
        )

        class _CaptureAgent(_FakeAgent):
            def run(
                self, messages, *, session, stream: bool = False, tools=None, **kwargs
            ):
                captured["messages"] = messages
                return super().run(
                    messages, session=session, stream=stream, tools=tools, **kwargs
                )

        def _builder(_registry, agent_id: str, **_kwargs):
            if agent_id != "orchestrator":
                raise AssertionError(f"unexpected agent: {agent_id}")
            return _CaptureAgent(["Final answer"])

        core = ProxyOrchestrationCore(
            _provider_registry(tool_calling=True), agent_builder=_builder
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect main.py"),
                ProxyInputMessage(
                    role="tool",
                    content="print('current main')",
                    tool_call_id=tool_call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        messages = captured["messages"]
        self.assertEqual(
            [message.role for message in messages], ["user", "assistant", "tool"]
        )
        self.assertEqual(messages[1].contents[0].type, "function_call")
        self.assertEqual(messages[1].contents[0].call_id, tool_call.id)
        self.assertEqual(messages[1].contents[0].name, "read_file")
        self.assertEqual(messages[2].contents[0].type, "function_result")
        self.assertEqual(messages[2].contents[0].call_id, "call_1")


class _FakeRegistry:
    def __init__(self) -> None:
        self.inner = RuntimeRegistry(
            upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
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
            workroom_definitions={
                "standard-build": DefinitionDocument(
                    id="standard-build",
                    path=Path("standard-build.md"),
                    metadata={
                        "id": "standard-build",
                        "stages": ["architect", "coder"],
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


def _fake_agent_builder(mapping: dict[str, list[object]]):
    remaining = {agent_id: list(responses) for agent_id, responses in mapping.items()}

    def _build(_registry, agent_id: str, **_kwargs):
        queue = remaining[agent_id]
        if not queue:
            raise AssertionError(f"no fake responses left for {agent_id}")
        return _FakeAgent([queue.pop(0)])

    return _build


def _fake_registry():
    return _FakeRegistry()


def _staged_workroom_registry():
    registry = _FakeRegistry()
    registry.workroom_definitions["staged-build"] = DefinitionDocument(
        id="staged-build",
        path=Path("staged-build.md"),
        metadata={
            "id": "staged-build",
            "stages": [["architect", "coder", "reviewer"]],
        },
        body="## Purpose\nStaged build.",
        sections={"Purpose": "Staged build."},
    )
    return registry


def _provider_registry(*, tool_calling: bool) -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(
            base_url="http://localhost:8080/v1", tool_calling=tool_calling
        ),
        agent_definitions={
            "orchestrator": DefinitionDocument(
                id="orchestrator",
                path=Path("orchestrator.md"),
                metadata={"id": "orchestrator", "role": "orchestrator"},
                body="## Identity\nLead engineer.",
                sections={"Identity": "Lead engineer."},
            ),
        },
        workroom_definitions={},
    )


def _advanced_workroom_registry() -> RuntimeRegistry:
    return RuntimeRegistry(
        upstream=UpstreamSettings(base_url="http://localhost:8080/v1"),
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
            "researcher": DefinitionDocument(
                id="researcher",
                path=Path("researcher.md"),
                metadata={"id": "researcher", "role": "researcher"},
                body="## Identity\nResearcher.",
                sections={"Identity": "Researcher."},
            ),
            "reviewer": DefinitionDocument(
                id="reviewer",
                path=Path("reviewer.md"),
                metadata={"id": "reviewer", "role": "reviewer"},
                body="## Identity\nReviewer.",
                sections={"Identity": "Reviewer."},
            ),
        },
        workroom_definitions={
            "debate": DefinitionDocument(
                id="debate",
                path=Path("debate.md"),
                metadata={
                    "id": "debate",
                    "turns": [
                        "architect",
                        "brainstormer",
                        "architect",
                        "reviewer",
                    ],
                },
                body="## Purpose\nDebate.",
                sections={"Purpose": "Debate."},
            ),
            "dynamic-open-ended": DefinitionDocument(
                id="dynamic-open-ended",
                path=Path("dynamic-open-ended.md"),
                metadata={
                    "id": "dynamic-open-ended",
                    "turns": ["architect", "reviewer"],
                },
                body="## Purpose\nAdaptive.",
                sections={"Purpose": "Adaptive."},
            ),
            "handoff-chain": DefinitionDocument(
                id="handoff-chain",
                path=Path("handoff-chain.md"),
                metadata={
                    "id": "handoff-chain",
                    "stages": [["researcher"], ["reviewer"]],
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


def _internal_action(name: str, **payload: object) -> dict[str, object]:
    return {
        "text": "",
        "tool_calls": [
            {
                "id": f"internal_{name}",
                "name": name,
                "arguments": json.dumps(payload),
            }
        ],
    }


def _host_continuation_tool_call(
    *, state: ContinuationState, call_id: str, name: str
) -> ProxyToolCall:
    return encode_continuation_tool_call(
        ProxyToolCall(id=call_id, name=name, arguments_json="{}"),
        state=state,
    )
