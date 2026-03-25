from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentInvocation, AgentRunResult
from ergon_studio.proxy.continuation import decode_pending_id_from_tool_call_id
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
            agent_invoker=_fake_agent_invoker({"orchestrator": ["Hello world"]}),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )

        stream = core.stream_turn(request, session_id="session_1")
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

    async def test_stream_turn_passes_real_pm_history_to_orchestrator(self) -> None:
        captured_invocations: list[AgentInvocation] = []

        def _capturing_invoker(invocation: AgentInvocation):
            captured_invocations.append(invocation)
            return _response_stream("Done")

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_capturing_invoker,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="First request"),
                ProxyInputMessage(role="assistant", content="First reply"),
                ProxyInputMessage(role="user", content="Second request"),
            ),
        )

        stream = core.stream_turn(request, session_id="session_1")
        [event async for event in stream]
        await stream.get_final_response()

        invocation = captured_invocations[0]
        self.assertEqual(
            [message["role"] for message in invocation.messages[:5]],
            ["system", "system", "user", "assistant", "user"],
        )
        self.assertEqual(invocation.messages[2]["content"], "First request")
        self.assertEqual(invocation.messages[3]["content"], "First reply")
        self.assertEqual(invocation.messages[4]["content"], "Second request")

    async def test_stream_turn_opens_channel_and_returns_participant_reply(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Implement it",
                            recipients=["coder"],
                        ),
                        "Final summary",
                    ],
                    "coder": ["Patch applied"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Implement it"),),
        )

        stream = core.stream_turn(request, session_id="session_1")
        events = [event async for event in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("opening channel channel-1 with coder", reasoning.lower())
        self.assertIn("coder: Patch applied", reasoning)
        self.assertEqual(result.content, "Final summary")

    async def test_stream_turn_rejects_mixed_preset_and_participants(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            preset="standard-build",
                            participants=["coder"],
                            message="Build it",
                            recipients=["coder"],
                        ),
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )

        stream = core.stream_turn(request, session_id="session_1")
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn(
            "open_channel requires either preset or participants, not both",
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
        )

    async def test_stream_turn_persists_channels_by_session_id(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start here",
                            recipients=["coder"],
                        ),
                        "Opened",
                        _internal_action(
                            "message_channel",
                            channel="channel-1",
                            message="Continue",
                            recipients=["coder"],
                        ),
                        "Done",
                    ],
                    "coder": ["First pass", "Second pass"],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Start"),),
        )
        second_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Continue"),),
        )

        first_stream = core.stream_turn(
            first_request,
            session_id="session_1",
        )
        [event async for event in first_stream]
        await first_stream.get_final_response()

        second_stream = core.stream_turn(
            second_request,
            session_id="session_1",
        )
        events = [event async for event in second_stream]
        result = await second_stream.get_final_response()

        reasoning = "".join(
            event.delta
            for event in events
            if isinstance(event, ProxyReasoningDeltaEvent)
        )
        self.assertIn("continuing channel channel-1", reasoning.lower())
        self.assertIn("coder: Second pass", reasoning)
        self.assertEqual(result.content, "Done")

    async def test_close_channel_ends_channel_lifecycle(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start",
                            recipients=["coder"],
                        ),
                        "Opened",
                        _internal_action("close_channel", channel="channel-1"),
                        "Closed",
                        _internal_action(
                            "message_channel",
                            channel="channel-1",
                            message="Continue",
                            recipients=["coder"],
                        ),
                    ],
                    "coder": ["First pass"],
                }
            ),
        )
        open_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Start"),),
        )
        close_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Close it"),),
        )
        reopen_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Continue it"),),
        )

        first_stream = core.stream_turn(
            open_request,
            session_id="session_1",
        )
        [event async for event in first_stream]
        await first_stream.get_final_response()

        close_stream = core.stream_turn(
            close_request,
            session_id="session_1",
        )
        [event async for event in close_stream]
        await close_stream.get_final_response()

        stream = core.stream_turn(
            reopen_request,
            session_id="session_1",
        )
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertIn(
            "unknown channel: channel-1",
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
        )

    async def test_stream_turn_resumes_orchestrator_tool_call_from_pending_store(
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
                            ],
                        },
                        "Done",
                    ],
                }
            ),
        )
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Inspect it"),),
            tools=(_host_tool("read_file"),),
        )

        first_stream = core.stream_turn(
            first_request,
            session_id="session_1",
        )
        first_events = [event async for event in first_stream]
        first_result = await first_stream.get_final_response()

        tool_event = next(
            event for event in first_events if isinstance(event, ProxyToolCallEvent)
        )
        second_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Inspect it"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=first_result.tool_calls,
                ),
                ProxyInputMessage(
                    role="tool",
                    content="file contents",
                    tool_call_id=tool_event.call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )

        second_stream = core.stream_turn(
            second_request,
            session_id="session_1",
        )
        [event async for event in second_stream]
        second_result = await second_stream.get_final_response()

        self.assertEqual(second_result.content, "Done")

    async def test_stream_turn_rejects_pending_channel_without_channel_id(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start",
                            recipients=["coder"],
                        ),
                        "Opened",
                    ],
                    "coder": [
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
        first_request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Start"),),
            tools=(_host_tool("read_file"),),
        )

        first_stream = core.stream_turn(first_request, session_id="session_1")
        first_events = [event async for event in first_stream]
        await first_stream.get_final_response()

        tool_event = next(
            event for event in first_events if isinstance(event, ProxyToolCallEvent)
        )
        pending_id = decode_pending_id_from_tool_call_id(tool_event.call.id)
        assert pending_id is not None
        core._pending_store._records[pending_id] = replace(  # type: ignore[attr-defined]
            core._pending_store._records[pending_id],  # type: ignore[attr-defined]
            active_channel_id=None,
        )

        second_request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(role="user", content="Start"),
                ProxyInputMessage(
                    role="assistant",
                    content="",
                    tool_calls=(tool_event.call,),
                ),
                ProxyInputMessage(
                    role="tool",
                    content="file contents",
                    tool_call_id=tool_event.call.id,
                ),
            ),
            tools=(_host_tool("read_file"),),
        )

        second_stream = core.stream_turn(second_request, session_id="session_1")
        events = [event async for event in second_stream]
        second_result = await second_stream.get_final_response()

        self.assertEqual(second_result.finish_reason, "error")
        self.assertIn(
            "pending channel resume is missing an active channel id",
            "".join(
                event.delta
                for event in events
                if isinstance(event, ProxyContentDeltaEvent)
            ),
        )


    async def test_orchestrator_retries_on_malformed_tool_json(self) -> None:
        malformed = {"id": "tc_bad", "name": "open_channel", "arguments": "{bad json"}
        valid = {
            "id": "tc_good",
            "name": "open_channel",
            "arguments": json.dumps(
                {"participants": ["coder"], "message": "Build it", "recipients": ["coder"]}
            ),
        }
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {"text": "", "tool_calls": [malformed]},
                        {"text": "", "tool_calls": [valid]},
                        "Done",
                    ],
                    "coder": ["Patch applied"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.content, "Done")

    async def test_orchestrator_fails_after_max_retries(self) -> None:
        malformed = {"id": "tc_bad", "name": "open_channel", "arguments": "{bad json"}
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {"text": "", "tool_calls": [malformed]},
                        {"text": "", "tool_calls": [malformed]},
                        {"text": "", "tool_calls": [malformed]},
                    ]
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")

    async def test_orchestrator_does_not_retry_structural_errors(self) -> None:
        invoker_call_count = 0

        def _counting_invoker(invocation: AgentInvocation):
            nonlocal invoker_call_count
            invoker_call_count += 1
            # Valid JSON but missing required 'message' field
            return _response_stream(
                "",
                response=AgentRunResult(
                    text="",
                    tool_calls=(
                        ProxyToolCall(
                            id="tc_1",
                            name="open_channel",
                            arguments_json='{"participants":["coder"],"recipients":["coder"]}',
                        ),
                    ),
                ),
            )

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_counting_invoker)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        self.assertEqual(invoker_call_count, 1)

    async def test_message_channel_before_open_raises_error_without_side_effects(
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
                                    "id": "tc_1",
                                    "name": "open_channel",
                                    "arguments": json.dumps(
                                        {
                                            "participants": ["coder"],
                                            "message": "Start",
                                            "recipients": ["coder"],
                                        }
                                    ),
                                },
                                {
                                    "id": "tc_2",
                                    "name": "message_channel",
                                    "arguments": json.dumps(
                                        {
                                            "channel": "channel-99",
                                            "message": "Continue",
                                            "recipients": ["coder"],
                                        }
                                    ),
                                },
                            ],
                        },
                    ],
                    # 'coder' intentionally absent: if open_channel runs, it would
                    # fail with KeyError rather than the expected "unknown channel" error
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")
        error_text = "".join(
            e.delta for e in events if isinstance(e, ProxyContentDeltaEvent)
        )
        self.assertIn("unknown channel", error_text)
        # "channel-99" must appear — not a KeyError from missing coder
        self.assertIn("channel-99", error_text)

    async def test_final_pass_orchestrator_text_emitted_as_content(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker({"orchestrator": ["Final answer"]}),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="What?"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [event async for event in stream]
        await stream.get_final_response()

        content_text = "".join(
            e.delta for e in events if isinstance(e, ProxyContentDeltaEvent)
        )
        self.assertIn("Final answer", content_text)

    async def test_pre_channel_orchestrator_text_emitted_as_reasoning(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {
                            "text": "Let me check...",
                            "tool_calls": [
                                {
                                    "id": "internal_open",
                                    "name": "open_channel",
                                    "arguments": json.dumps(
                                        {
                                            "participants": ["coder"],
                                            "message": "Implement it",
                                            "recipients": ["coder"],
                                        }
                                    ),
                                }
                            ],
                        },
                        "Done",
                    ],
                    "coder": ["Patch applied"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Do it"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [event async for event in stream]
        await stream.get_final_response()

        reasoning_text = "".join(
            e.delta for e in events if isinstance(e, ProxyReasoningDeltaEvent)
        )
        content_text = "".join(
            e.delta for e in events if isinstance(e, ProxyContentDeltaEvent)
        )
        self.assertIn("Let me check...", reasoning_text)
        self.assertNotIn("Let me check...", content_text)

    async def test_orchestrator_receives_prior_text_in_loop_history(self) -> None:
        captured_invocations: list[AgentInvocation] = []
        base_invoker = _fake_agent_invoker(
            {
                "orchestrator": [
                    {
                        "text": "Checking...",
                        "tool_calls": [
                            {
                                "id": "internal_open",
                                "name": "open_channel",
                                "arguments": json.dumps(
                                    {
                                        "participants": ["coder"],
                                        "message": "Check it",
                                        "recipients": ["coder"],
                                    }
                                ),
                            }
                        ],
                    },
                    "Done",
                ],
                "coder": ["Patch applied"],
            }
        )

        def _capturing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "orchestrator":
                captured_invocations.append(invocation)
            return base_invoker(invocation)

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_capturing_invoker)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Do it"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(len(captured_invocations), 2)
        second_messages = captured_invocations[1].messages
        assistant_msgs = [
            m for m in second_messages
            if m["role"] == "assistant" and m.get("content") == "Checking..."
        ]
        self.assertTrue(
            len(assistant_msgs) >= 1,
            f"Expected assistant 'Checking...' in second invocation. Messages: {second_messages}",
        )

    async def test_orchestrator_receives_channel_results_in_loop_history(self) -> None:
        captured_invocations: list[AgentInvocation] = []
        base_invoker = _fake_agent_invoker(
            {
                "orchestrator": [
                    _internal_action(
                        "open_channel",
                        participants=["coder"],
                        message="Build it",
                        recipients=["coder"],
                    ),
                    "Done",
                ],
                "coder": ["Solution X"],
            }
        )

        def _capturing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "orchestrator":
                captured_invocations.append(invocation)
            return base_invoker(invocation)

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_capturing_invoker)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build it"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(len(captured_invocations), 2)
        second_messages = captured_invocations[1].messages
        user_msgs_with_solution = [
            m for m in second_messages
            if m["role"] == "user" and "Solution X" in str(m.get("content", ""))
        ]
        self.assertTrue(
            len(user_msgs_with_solution) >= 1,
            f"Expected user message with 'Solution X' in second invocation. Messages: {second_messages}",
        )


class RunParallelTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_parallel_invokes_agent_n_times(self) -> None:
        coder_invocations: list[AgentInvocation] = []
        base_invoker = _fake_agent_invoker(
            {
                "orchestrator": [
                    _internal_action(
                        "run_parallel",
                        agent="coder",
                        count=2,
                        task="Build X",
                    ),
                    "Done",
                ],
                "coder": ["Result A", "Result B"],
            }
        )

        def _capturing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "coder":
                coder_invocations.append(invocation)
            return base_invoker(invocation)

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_capturing_invoker,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.content, "Done")
        self.assertEqual(len(coder_invocations), 2)

    async def test_run_parallel_results_injected_into_loop_history(self) -> None:
        orchestrator_invocations: list[AgentInvocation] = []
        base_invoker = _fake_agent_invoker(
            {
                "orchestrator": [
                    _internal_action(
                        "run_parallel",
                        agent="coder",
                        count=1,
                        task="Build X",
                    ),
                    "Done",
                ],
                "coder": ["My unique result"],
            }
        )

        def _capturing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "orchestrator":
                orchestrator_invocations.append(invocation)
            return base_invoker(invocation)

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_capturing_invoker,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        await stream.get_final_response()

        self.assertEqual(len(orchestrator_invocations), 2)
        second_messages = orchestrator_invocations[1].messages
        all_content = " ".join(
            str(m.get("content", "")) for m in second_messages
        )
        self.assertIn("My unique result", all_content)

    async def test_run_parallel_surfaced_as_reasoning_events(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "run_parallel",
                            agent="coder",
                            count=1,
                            task="Task",
                        ),
                        "Done",
                    ],
                    "coder": ["Subsession thinking"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [event async for event in stream]
        await stream.get_final_response()

        reasoning_texts = [
            e.delta for e in events if isinstance(e, ProxyReasoningDeltaEvent)
        ]
        self.assertTrue(
            any("Subsession thinking" in t for t in reasoning_texts),
            f"Expected 'Subsession thinking' in reasoning. Got: {reasoning_texts}",
        )

    async def test_orchestrator_prompt_mentions_run_parallel(self) -> None:
        captured: list[AgentInvocation] = []

        def _capturing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "orchestrator":
                captured.append(invocation)
            return _response_stream("Done")

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_capturing_invoker,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Hi"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        await stream.get_final_response()

        all_system_content = " ".join(
            m["content"]
            for m in captured[0].messages
            if m["role"] in ("system", "user") and m.get("content")
        )
        self.assertIn("run_parallel", all_system_content)

    async def test_run_parallel_with_unknown_agent_fails(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "run_parallel",
                            agent="nonexistent_agent",
                            count=1,
                            task="Task",
                        ),
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")

    async def test_run_parallel_with_invalid_count_fails(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "run_parallel",
                            agent="coder",
                            count=0,
                            task="Task",
                        ),
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "error")

    async def test_run_parallel_one_failing_subsession_does_not_crash_others(
        self,
    ) -> None:
        call_counts: dict[str, int] = {"coder": 0}

        def _partly_failing_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "coder":
                call_counts["coder"] += 1
                if call_counts["coder"] == 1:
                    # First coder call raises
                    async def _fail_events():
                        raise RuntimeError("sub-session failed")
                        yield  # noqa: unreachable

                    return ResponseStream(
                        _fail_events(),
                        finalizer=lambda: AgentRunResult(text="", tool_calls=()),
                    )
            return _response_stream("Success")

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_partly_failing_invoker,
        )
        # orchestrator calls run_parallel with count=2, one sub-session fails
        # The turn should not crash; the orchestrator gets a result (even if partial)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )

        def _orchestrator_invoker(invocation: AgentInvocation):
            if invocation.agent_id == "orchestrator":
                if call_counts.get("orchestrator", 0) == 0:
                    call_counts["orchestrator"] = 1
                    return _response_stream(
                        "",
                        response=AgentRunResult(
                            text="",
                            tool_calls=(
                                ProxyToolCall(
                                    id="rp1",
                                    name="run_parallel",
                                    arguments_json=json.dumps(
                                        {"agent": "coder", "count": 2, "task": "X"}
                                    ),
                                ),
                            ),
                        ),
                    )
                return _response_stream("Done")
            return _partly_failing_invoker(invocation)

        core2 = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_orchestrator_invoker,
        )
        stream = core2.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        # Turn must complete (not crash); orchestrator got something back
        self.assertIn(result.finish_reason, {"stop", "error"})

    async def test_run_parallel_with_count_one_returns_single_result(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _internal_action(
                            "run_parallel",
                            agent="coder",
                            count=1,
                            task="Do it",
                        ),
                        "All done",
                    ],
                    "coder": ["Solo result"],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [event async for event in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.content, "All done")


def _fake_registry() -> RuntimeRegistry:
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
            "coder": DefinitionDocument(
                id="coder",
                path=Path("coder.md"),
                metadata={"id": "coder", "role": "coder"},
                body="## Identity\nCoder.",
                sections={"Identity": "Coder."},
            ),
            "architect": DefinitionDocument(
                id="architect",
                path=Path("architect.md"),
                metadata={"id": "architect", "role": "architect"},
                body="## Identity\nArchitect.",
                sections={"Identity": "Architect."},
            ),
        },
        channel_presets={"standard-build": ("architect", "coder")},
    )


def _host_tool(name: str):
    from ergon_studio.proxy.models import ProxyFunctionTool

    return ProxyFunctionTool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _internal_action(name: str, **payload):
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


def _fake_agent_invoker(
    responses_by_agent: dict[str, list[str | dict[str, object]]],
):
    counters = {agent_id: 0 for agent_id in responses_by_agent}

    def _invoker(invocation: AgentInvocation):
        index = counters[invocation.agent_id]
        counters[invocation.agent_id] += 1
        response = responses_by_agent[invocation.agent_id][index]
        if isinstance(response, str):
            return _response_stream(response)
        return _response_stream(
            str(response.get("text", "")),
            response=AgentRunResult(
                text=str(response.get("text", "")),
                tool_calls=tuple(
                    ProxyToolCall(
                        id=str(tool_call["id"]),
                        name=str(tool_call["name"]),
                        arguments_json=str(tool_call["arguments"]),
                    )
                    for tool_call in response.get("tool_calls", [])
                ),
            ),
        )

    return _invoker


def _response_stream(
    text: str,
    *,
    response: AgentRunResult | None = None,
) -> ResponseStream[str, AgentRunResult]:
    async def _events():
        if text:
            yield text

    final = response or AgentRunResult(text=text, tool_calls=())
    return ResponseStream(_events(), finalizer=lambda: final)
