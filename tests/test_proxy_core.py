from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Callable
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentInvocation, AgentRunResult
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
from ergon_studio.response_stream import ResponseStream
from ergon_studio.upstream import UpstreamSettings


class ProxyCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_replies_directly_when_orchestrator_just_answers(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
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

    async def test_stream_turn_emits_direct_reply_deltas_before_turn_finishes(
        self,
    ) -> None:
        gate = asyncio.Event()

        def _invoker(_invocation: AgentInvocation):
            async def _events():
                yield "Hello"
                await gate.wait()
                yield " world"

            return ResponseStream(
                _events(),
                finalizer=lambda _updates: AgentRunResult(
                    text="Hello world",
                    tool_calls=(),
                ),
            )

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_invoker)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        first_event = await asyncio.wait_for(stream.__anext__(), timeout=0.2)
        self.assertIsInstance(first_event, ProxyContentDeltaEvent)
        self.assertEqual(first_event.delta, "Hello")

        gate.set()
        events = [first_event, *[event async for event in stream]]
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

    async def test_stream_turn_converts_unexpected_exceptions_to_error_results(
        self,
    ) -> None:
        def _invoker(_invocation: AgentInvocation):
            raise RuntimeError("core exploded")

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_invoker)
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

    async def test_stream_turn_passes_requested_model_to_agent_invoker(self) -> None:
        captured: dict[str, object] = {}

        def _capture(invocation: AgentInvocation) -> None:
            captured["agent_id"] = invocation.agent.id
            captured["model_id_override"] = invocation.model

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {"orchestrator": ["Hello world"]},
                capture=_capture,
            ),
        )
        request = ProxyTurnRequest(
            model="gpt-oss-20b",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(captured["agent_id"], "orchestrator")
        self.assertEqual(captured["model_id_override"], "gpt-oss-20b")

    async def test_stream_turn_allows_text_before_workroom_action(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {
                            "text": "Starting now",
                            "tool_calls": [
                                {
                                    "id": "internal_message_workroom",
                                    "name": "message_workroom",
                                    "arguments": json.dumps(
                                        {
                                            "participants": ["coder"],
                                            "message": "Implement it",
                                        }
                                    ),
                                }
                            ],
                        },
                        "Done",
                    ],
                    "coder": ["Here is the code"],
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

        content = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyContentDeltaEvent)
        )
        self.assertIn("Starting now", content)
        self.assertNotEqual(result.finish_reason, "error")

    async def test_stream_turn_opens_single_person_workroom_and_then_replies(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
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

    async def test_stream_turn_keeps_solo_worker_until_it_replies_to_lead_dev(
        self,
    ) -> None:
        agent_order: list[str] = []

        def _capture(invocation: AgentInvocation) -> None:
            agent_order.append(invocation.agent.id)

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            participants=["coder"],
                            message="Update the README.",
                        ),
                        "Final summary",
                    ],
                    "coder": [
                        "Checked README.md and found the current intro.",
                        _internal_action(
                            "reply_lead_dev",
                            message="Updated the README intro and added setup notes.",
                        ),
                    ],
                },
                capture=_capture,
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Update the README"),),
            tools=(_host_tool("read_file"), _host_tool("write_file")),
        )

        stream = core.stream_turn(request, created_at=1)
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("coder: Checked README.md", reasoning)
        self.assertIn("coder: Updated the README intro", reasoning)
        self.assertEqual(
            agent_order,
            ["orchestrator", "coder", "coder", "orchestrator"],
        )
        self.assertEqual(result.content, "Final summary")

    async def test_stream_turn_handles_workroom_rounds(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="standard-build",
                            message="Build calculator",
                        ),
                        _internal_action(
                            "message_workroom",
                            participants=["coder"],
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
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="standard-build",
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
            agent_invoker=_fake_agent_invoker(
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
        self.assertEqual(continuation.agent_id, "orchestrator")
        self.assertIsNone(continuation.workroom_name)

    async def test_stream_turn_resumes_workroom_from_tool_result(self) -> None:
        first_core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="standard-build",
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
            agent_invoker=_fake_agent_invoker(
                {
                    "architect": [
                        _internal_action(
                            "reply_lead_dev",
                            message="Architecture plan",
                        )
                    ],
                    "coder": [
                        _internal_action(
                            "reply_lead_dev",
                            message="Built feature",
                        )
                    ],
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            participants=["coder"],
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
                agent_id="coder",
                workroom_name="ad hoc",
                workroom_participants=("coder",),
            ),
            call_id="call_1",
            name="read_file",
        )
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
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

    async def test_workroom_continuation_keeps_remaining_participants_in_same_round(
        self,
    ) -> None:
        registry = _multi_participant_workroom_registry()
        first_core = ProxyOrchestrationCore(
            registry,
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="build-room",
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
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": ["Workroom final summary"],
                    "architect": [
                        _internal_action(
                            "reply_lead_dev",
                            message="Architecture plan",
                        )
                    ],
                    "coder": [
                        _internal_action(
                            "reply_lead_dev",
                            message="Built feature",
                        )
                    ],
                    "reviewer": ["Reviewed result"],
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

    async def test_workroom_uses_template_participant_order(self) -> None:
        registry = _advanced_workroom_registry()
        core = ProxyOrchestrationCore(
            registry,
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="debate",
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
        self.assertIn("architect[1]: Option", reasoning)
        self.assertIn("architect[2]: Refined", reasoning)
        self.assertIn("brainstormer: Option", reasoning)
        self.assertIn("reviewer: Decision-ready", reasoning)
        self.assertEqual(result.content, "Debate final summary")

    async def test_message_workroom_can_replace_room_staffing(self) -> None:
        registry = _advanced_workroom_registry()
        core = ProxyOrchestrationCore(
            registry,
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "message_workroom",
                            preset="debate",
                            message="Build it",
                        ),
                        _internal_action(
                            "message_workroom",
                            participants=["reviewer"],
                            message="Reviewer, give the final verdict.",
                        ),
                        "Done",
                    ],
                    "architect": ["Architecture pass", "Refined architecture"],
                    "brainstormer": ["Alternative path"],
                    "reviewer": ["Review pass", "Final verdict"],
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
        self.assertEqual(reasoning.count("architect[1]: Architecture"), 1)
        self.assertEqual(reasoning.count("brainstormer: Alternative"), 1)
        self.assertEqual(reasoning.count("reviewer: Review"), 1)
        self.assertEqual(reasoning.count("reviewer: Final"), 1)
        self.assertEqual(result.content, "Done")

    async def test_stream_turn_respects_host_tool_policy(self) -> None:
        captured: dict[str, object] = {}
        
        def _capture(invocation: AgentInvocation) -> None:
            captured["tools"] = invocation.tools
            captured["tool_choice"] = invocation.tool_choice
            captured["parallel_tool_calls"] = invocation.parallel_tool_calls

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {"orchestrator": [{"text": "", "tool_calls": []}]},
                capture=_capture,
            ),
        )
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
        tool_choice = captured["tool_choice"]
        self.assertEqual(
            [tool.name for tool in tools],
            ["write_file", "message_workroom"],
        )
        self.assertEqual(
            tool_choice,
            {"type": "function", "function": {"name": "write_file"}},
        )
        self.assertFalse(captured["parallel_tool_calls"])

    async def test_stream_turn_strips_optional_tools_when_provider_cannot_call_tools(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        
        def _capture(invocation: AgentInvocation) -> None:
            captured["tools"] = invocation.tools
            captured["tool_choice"] = invocation.tool_choice

        registry = _provider_registry(tool_calling=False)
        core = ProxyOrchestrationCore(
            registry,
            agent_invoker=_fake_agent_invoker(
                {"orchestrator": ["Done"]},
                capture=_capture,
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect main.py"),),
            tools=(_host_tool("read_file"),),
            tool_choice="auto",
        )

        stream = core.stream_turn(request, created_at=1)
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(captured["tools"], ())
        self.assertIsNone(captured["tool_choice"])

    async def test_stream_turn_errors_when_required_tool_choice_hits_toolless_provider(
        self,
    ) -> None:
        registry = _provider_registry(tool_calling=False)
        core = ProxyOrchestrationCore(
            registry,
            agent_invoker=_fake_agent_invoker(
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
            agent_invoker=_fake_agent_invoker(
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
            agent_invoker=_fake_agent_invoker(
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
            state=ContinuationState(agent_id="orchestrator"),
            call_id="call_1",
            name="read_file",
        )

        def _capture(invocation: AgentInvocation) -> None:
            captured["messages"] = invocation.messages

        core = ProxyOrchestrationCore(
            _provider_registry(tool_calling=True),
            agent_invoker=_fake_agent_invoker(
                {"orchestrator": ["Final answer"]},
                capture=_capture,
            ),
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
            [message["role"] for message in messages],
            ["system", "assistant", "tool", "user"],
        )
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[2]["tool_call_id"], tool_call.id)

    async def test_stream_turn_rebuilds_tool_result_without_assistant_call_history(
        self,
    ) -> None:
        captured: dict[str, object] = {}
        tool_call = _host_continuation_tool_call(
            state=ContinuationState(agent_id="orchestrator"),
            call_id="call_1",
            name="read_file",
        )

        def _capture(invocation: AgentInvocation) -> None:
            captured["messages"] = invocation.messages

        core = ProxyOrchestrationCore(
            _provider_registry(tool_calling=True),
            agent_invoker=_fake_agent_invoker(
                {"orchestrator": ["Final answer"]},
                capture=_capture,
            ),
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
            [message["role"] for message in messages],
            ["system", "assistant", "tool", "user"],
        )
        self.assertEqual(messages[1]["tool_calls"][0]["id"], tool_call.id)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")


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
                "reviewer": DefinitionDocument(
                    id="reviewer",
                    path=Path("reviewer.md"),
                    metadata={"id": "reviewer", "role": "reviewer"},
                    body="## Identity\nReviewer.",
                    sections={"Identity": "Reviewer."},
                ),
            },
            workroom_definitions={
                "standard-build": DefinitionDocument(
                    id="standard-build",
                    path=Path("standard-build.md"),
                metadata={
                    "id": "standard-build",
                    "participants": ["architect"],
                },
                body="## Purpose\nBuild.",
                sections={"Purpose": "Build."},
            )
            },
        )

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def _fake_agent_invoker(
    mapping: dict[str, list[object]],
    *,
    capture: Callable[[AgentInvocation], None] | None = None,
):
    remaining = {agent_id: list(responses) for agent_id, responses in mapping.items()}

    def _invoke(invocation: AgentInvocation):
        if capture is not None:
            capture(invocation)
        queue = remaining[invocation.agent.id]
        if not queue:
            raise AssertionError(f"no fake responses left for {invocation.agent.id}")
        raw = queue.pop(0)
        if isinstance(raw, str):
            payload = {"text": raw, "tool_calls": []}
        else:
            payload = raw
        text = payload.get("text", "")
        tool_calls = tuple(
            ProxyToolCall(
                id=tool_call["id"],
                name=tool_call["name"],
                arguments_json=tool_call["arguments"],
            )
            for tool_call in payload.get("tool_calls", [])
        )
        parts = [piece for piece in text.split(" ") if piece]

        async def _events():
            for index, part in enumerate(parts):
                suffix = " " if index < len(parts) - 1 else ""
                yield part + suffix

        return ResponseStream(
            _events(),
            finalizer=lambda _updates: AgentRunResult(
                text=text,
                tool_calls=tool_calls,
            ),
        )

    return _invoke


def _fake_registry():
    return _FakeRegistry()


def _multi_participant_workroom_registry():
    registry = _FakeRegistry()
    registry.workroom_definitions["build-room"] = DefinitionDocument(
        id="build-room",
        path=Path("build-room.md"),
        metadata={
            "id": "build-room",
            "participants": ["architect", "coder", "reviewer"],
        },
        body="## Purpose\nBuild room.",
        sections={"Purpose": "Build room."},
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
                    "participants": [
                        "architect",
                        "brainstormer",
                        "architect",
                        "reviewer",
                    ],
                },
                body="## Purpose\nDebate.",
                sections={"Purpose": "Debate."},
            ),
        },
    )



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
