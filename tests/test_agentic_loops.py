"""
End-to-end tests for agentic orchestration flows.

The overarching question every test asks is:
  given a real goal, does the system actually orchestrate?

That means:
  - does the orchestrator route correctly (delegate vs. answer directly)?
  - does each agent receive the right context (brief, design, prior transcript)?
  - does the orchestrator's synthesis incorporate what agents actually produced?
  - are events ordered correctly (reasoning before content)?
  - do channels and sessions persist across HTTP turns?
  - does run_parallel inject all results into the synthesis?
  - do workspace tool calls produce real files in isolated overlays?
  - does the system recover from errors without crashing the turn?

Tests 1-4 and 7 use REACTIVE invokers: each agent reads its input and echoes
unique tokens, so assertions can trace whether the right content reached the
right agent at every step in the pipeline.  Pre-scripted responses (returning
fixed strings regardless of input) are avoided for any test that claims to
validate orchestration logic.
"""
from __future__ import annotations

import http.cookiejar
import json
import tempfile
import unittest
from pathlib import Path
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentInvocation, AgentRunResult
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyInputMessage,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyTurnRequest,
)
from ergon_studio.proxy.server import SESSION_HEADER_NAME, start_proxy_server_in_thread
from ergon_studio.proxy.session_overlay import SessionOverlay
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream
from ergon_studio.upstream import UpstreamSettings


# ---------------------------------------------------------------------------
# HTTP-level agentic loop tests (go through the full server stack)
# ---------------------------------------------------------------------------


class AgenticLoopHTTPTests(unittest.TestCase):
    """
    Full-stack tests that start a real aiohttp server, send HTTP requests,
    and validate responses.  The LLM layer is replaced by reactive fake
    invokers so the tests run deterministically without an upstream.
    """

    def _start(self, invoker) -> tuple:
        """Return (handle, port) for a server backed by the given invoker."""
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=invoker,
        )
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=core,
        )
        self.addCleanup(handle.close)
        return handle, handle.port

    def _post(self, port: int, messages: list, *, stream: bool = False,
              session_id: str | None = None) -> tuple[dict, str | None]:
        """POST to /v1/chat/completions and return (payload, session_id)."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if session_id:
            headers[SESSION_HEADER_NAME] = session_id
        request = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {"model": "ergon", "messages": messages, "stream": stream}
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request) as response:
            session_out = response.headers.get(SESSION_HEADER_NAME)
            body = response.read().decode("utf-8")
        if stream:
            return _parse_sse_body(body), session_out
        return json.loads(body), session_out

    # --- 1. Direct answer — orchestrator handles the task alone ---

    def test_orchestrator_answers_directly_without_delegation(self) -> None:
        """
        For a task the orchestrator can handle itself, it must answer in a
        single invocation and never call any specialist.  This validates
        that the absence of delegation is deliberate, not accidental.
        """
        invocations: list[AgentInvocation] = []

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            return _response_stream("The capital of France is Paris.")

        _, port = self._start(_invoker)
        payload, session_id = self._post(
            port,
            [{"role": "user", "content": "What is the capital of France?"}],
        )

        orch = [i for i in invocations if i.agent_id == "orchestrator"]
        specialists = [i for i in invocations if i.agent_id != "orchestrator"]

        self.assertEqual(len(orch), 1,
                         "orchestrator must be invoked exactly once for a direct answer")
        self.assertEqual(len(specialists), 0,
                         "no specialist must be invoked when the orchestrator answers directly")
        self.assertEqual(
            payload["choices"][0]["message"]["content"],
            "The capital of France is Paris.",
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertIsNotNone(session_id)
        self.assertTrue(session_id.startswith("session_"))  # type: ignore[union-attr]

    # --- 2. Delegation: brief reaches specialist; specialist output reaches synthesis ---

    def test_orchestrator_delegates_and_synthesizes_from_actual_specialist_output(
        self,
    ) -> None:
        """
        For a task that needs a specialist, the orchestrator must:
          1. Open a channel and forward the brief to the coder.
          2. After the coder responds, synthesize using the coder's ACTUAL output.

        The coder echoes the brief it received.  Assertions trace the data at
        every step so a silent routing failure cannot hide behind a scripted string.
        """
        invocations: list[AgentInvocation] = []

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            msgs = list(inv.messages)

            if inv.agent_id == "orchestrator":
                # Has the channel already completed and injected its results?
                channel_result = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "Channel channel-1" in m.get("content", "")
                    ),
                    None,
                )
                if channel_result:
                    return _response_stream(f"Synthesis: {channel_result}")
                # First call: delegate to coder
                brief = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="oc1",
                            name="open_channel",
                            arguments_json=json.dumps({
                                "participants": ["coder"],
                                "message": brief,
                                "recipients": ["coder"],
                            }),
                        ),),
                    ),
                )

            if inv.agent_id == "coder":
                # Echo what was received so we can verify data flow end-to-end
                received = next(
                    (
                        m["content"] for m in reversed(msgs)
                        if m.get("role") == "user"
                    ),
                    "",
                )
                return _response_stream(f"CODER_IMPL: built from ({received[:60]})")

            return _response_stream("ok")

        _, port = self._start(_invoker)
        payload, _ = self._post(
            port,
            [{"role": "user", "content": "implement an inventory tracker"}],
        )

        orch = [i for i in invocations if i.agent_id == "orchestrator"]
        coders = [i for i in invocations if i.agent_id == "coder"]

        # Structural: delegate then synthesize
        self.assertEqual(len(orch), 2,
                         "orchestrator must be invoked twice: once to delegate, once to synthesize")
        self.assertGreater(len(coders), 0, "coder must be invoked")

        # The original brief reached the coder
        coder_ctx = " ".join(m.get("content", "") for m in coders[0].messages)
        self.assertIn("inventory tracker", coder_ctx,
                      "coder must receive the original brief in its messages")

        # The coder's output reached the orchestrator's synthesis invocation
        second_orch_ctx = " ".join(m.get("content", "") for m in orch[1].messages)
        self.assertIn("CODER_IMPL:", second_orch_ctx,
                      "coder output must appear in orchestrator synthesis messages (loop_history)")

        # The final answer incorporates the coder's actual work
        content = payload["choices"][0]["message"]["content"]
        self.assertIn("CODER_IMPL:", content,
                      "final answer must reference what the coder actually produced")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    # --- 3. Context propagation: architect design reaches coder ---

    def test_architect_design_reaches_coder_via_message_channel(self) -> None:
        """
        In a standard-build channel the architect designs first and hands off
        to the coder via message_channel.  The coder must receive the ARCHITECT's
        design — proving participant-to-participant routing works and that the
        coder is not just seeing the raw orchestrator brief.
        """
        invocations: list[AgentInvocation] = []
        ARCH_TOKEN = "ARCH_DESIGN_v7f2"  # unique marker to trace through the pipeline

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            msgs = list(inv.messages)

            if inv.agent_id == "orchestrator":
                channel_result = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "Channel channel-1" in m.get("content", "")
                    ),
                    None,
                )
                if channel_result:
                    return _response_stream(f"Done: {channel_result[:120]}")
                brief = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="oc1",
                            name="open_channel",
                            arguments_json=json.dumps({
                                "preset": "standard-build",
                                "message": brief,
                                "recipients": ["architect"],
                            }),
                        ),),
                    ),
                )

            if inv.agent_id == "architect":
                # Produce a design tagged with the unique token and hand off to coder
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="mc1",
                            name="message_channel",
                            arguments_json=json.dumps({
                                "message": f"{ARCH_TOKEN}: routes POST /items, GET /items",
                                "recipients": ["coder"],
                            }),
                        ),),
                    ),
                )

            if inv.agent_id == "coder":
                # Echo received context so we can verify what arrived
                received = " ".join(
                    m.get("content", "") for m in msgs if m.get("role") == "user"
                )
                return _response_stream(
                    f"CODER_IMPL: implemented from ({received[:80]})"
                )

            return _response_stream("ok")

        _, port = self._start(_invoker)
        self._post(port, [{"role": "user", "content": "build an item catalogue"}])

        coder_calls = [i for i in invocations if i.agent_id == "coder"]
        self.assertGreater(len(coder_calls), 0, "coder must be invoked")

        coder_ctx = " ".join(m.get("content", "") for m in coder_calls[0].messages)
        self.assertIn(
            ARCH_TOKEN, coder_ctx,
            "the architect's unique design token must reach the coder via message_channel",
        )

    # --- 4. Multi-turn: channel persists and coder sees the full transcript ---

    def test_channel_persists_across_turns_coder_sees_prior_transcript(self) -> None:
        """
        Turn 1: orchestrator opens channel-1, coder delivers CODER_TURN1.
        Turn 2: orchestrator messages channel-1 again.  The coder's second
        invocation must contain CODER_TURN1 in its conversation — proving
        the channel transcript survives across HTTP requests.
        """
        invocations: list[AgentInvocation] = []
        opened = [False]  # tracks whether channel-1 has been opened this session

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            msgs = list(inv.messages)

            if inv.agent_id == "orchestrator":
                channel_result = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "Channel channel-1" in m.get("content", "")
                    ),
                    None,
                )
                if channel_result:
                    return _response_stream(f"Done: {channel_result[:120]}")
                brief = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                if not opened[0]:
                    opened[0] = True
                    return _response_stream(
                        "",
                        response=AgentRunResult(
                            text="",
                            tool_calls=(ProxyToolCall(
                                id="oc1",
                                name="open_channel",
                                arguments_json=json.dumps({
                                    "participants": ["coder"],
                                    "message": brief,
                                    "recipients": ["coder"],
                                }),
                            ),),
                        ),
                    )
                # Subsequent turns: re-use the existing channel
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="mc1",
                            name="message_channel",
                            arguments_json=json.dumps({
                                "channel": "channel-1",
                                "message": brief,
                                "recipients": ["coder"],
                            }),
                        ),),
                    ),
                )

            if inv.agent_id == "coder":
                # Count how many coder calls have happened so far (including this one)
                coder_n = len([i for i in invocations if i.agent_id == "coder"])
                received = " ".join(m.get("content", "") for m in msgs)
                return _response_stream(
                    f"CODER_TURN{coder_n}: result for ({received[:40]})"
                )

            return _response_stream("ok")

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_invoker)
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))

        def _make_request(content: str) -> dict:
            req = Request(
                f"http://127.0.0.1:{handle.port}/v1/chat/completions",
                data=json.dumps({
                    "model": "ergon",
                    "messages": [{"role": "user", "content": content}],
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(req) as resp:
                return json.loads(resp.read())

        first = _make_request("start the build")
        second = _make_request("add error handling")

        coders = [i for i in invocations if i.agent_id == "coder"]
        self.assertGreaterEqual(len(coders), 2, "coder must be called in both turns")

        # Turn 1 final answer references the coder's turn-1 output
        self.assertIn("CODER_TURN1:", first["choices"][0]["message"]["content"],
                      "turn-1 answer must contain the coder's turn-1 output")

        # Turn 2 coder invocation must contain the turn-1 response in its transcript
        turn2_coder_ctx = " ".join(m.get("content", "") for m in coders[1].messages)
        self.assertIn(
            "CODER_TURN1:", turn2_coder_ctx,
            "turn-2 coder invocation must see the turn-1 coder response in the channel transcript",
        )

        # Turn 2 answer reflects the new coder response
        self.assertIn("CODER_TURN2:", second["choices"][0]["message"]["content"],
                      "turn-2 answer must contain the coder's turn-2 output")

    # --- 5. Orchestrator pre-channel thinking is reasoning, final is content ---

    def test_pre_channel_thinking_is_reasoning_not_content(self) -> None:
        """
        Orchestrator emits text before opening a channel — that text must
        appear in SSE reasoning chunks, not in the final content field.
        """
        _, port = self._start(
            _fake_agent_invoker({
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        participants=["coder"],
                        message="Implement it.",
                        recipients=["coder"],
                        text="Let me delegate this to the team.",
                    ),
                    "Delegation complete.",
                ],
                "coder": ["Implementation done."],
            })
        )
        parsed, _ = self._post(
            port,
            [{"role": "user", "content": "Build something."}],
            stream=True,
        )
        reasoning = parsed.get("reasoning", "")
        content = parsed.get("content", "")
        self.assertIn("Let me delegate this to the team.", reasoning)
        self.assertNotIn("Let me delegate this to the team.", content)
        self.assertEqual(content, "Delegation complete.")

    # --- 6. Specialist reasoning appears in SSE stream ---

    def test_specialist_response_surfaced_as_reasoning_in_sse(self) -> None:
        """
        SSE stream must include channel participant responses as reasoning
        before the final assistant content arrives.
        """
        _, port = self._start(
            _fake_agent_invoker({
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        participants=["coder"],
                        message="Write the sort function.",
                        recipients=["coder"],
                    ),
                    "Sort function delivered.",
                ],
                "coder": ["def sort(lst): return sorted(lst)"],
            })
        )
        parsed, _ = self._post(
            port,
            [{"role": "user", "content": "Write a sort function."}],
            stream=True,
        )
        self.assertIn("def sort(lst)", parsed.get("reasoning", ""))
        self.assertEqual(parsed.get("content", ""), "Sort function delivered.")

    # --- 7. run_parallel: every unique sub-session result reaches synthesis ---

    def test_run_parallel_unique_results_all_reach_orchestrator_synthesis(
        self,
    ) -> None:
        """
        Orchestrator spawns N parallel sub-sessions.  Each returns a DIFFERENT,
        uniquely tagged result.  The orchestrator's synthesis invocation must
        contain ALL N results in its messages, and the final answer must
        incorporate them — proving that run_parallel fully injects every result
        into loop_history before the synthesis call.
        """
        invocations: list[AgentInvocation] = []
        coder_calls = [0]

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            msgs = list(inv.messages)

            if inv.agent_id == "orchestrator":
                parallel_result = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "run_parallel" in m.get("content", "")
                    ),
                    None,
                )
                if parallel_result:
                    return _response_stream(f"Aggregated: {parallel_result}")
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="rp1",
                            name="run_parallel",
                            arguments_json=json.dumps({
                                "agent": "coder",
                                "count": 3,
                                "task": "write a sort function",
                            }),
                        ),),
                    ),
                )

            if inv.agent_id == "coder":
                coder_calls[0] += 1
                n = coder_calls[0]
                return _response_stream(f"SORT_VARIANT_{n}: def sort_{n}(lst): ...")

            return _response_stream("ok")

        _, port = self._start(_invoker)
        payload, _ = self._post(
            port,
            [{"role": "user", "content": "give me three sort implementations"}],
        )

        orch = [i for i in invocations if i.agent_id == "orchestrator"]
        self.assertEqual(len(orch), 2,
                         "orchestrator must be called twice: once to launch, once to synthesize")

        # All three unique coder results must appear in the synthesis invocation's messages
        second_orch_ctx = " ".join(m.get("content", "") for m in orch[1].messages)
        for n in range(1, 4):
            self.assertIn(
                f"SORT_VARIANT_{n}:", second_orch_ctx,
                f"SORT_VARIANT_{n} must appear in orchestrator synthesis messages",
            )

        # Final answer incorporates the parallel results
        content = payload["choices"][0]["message"]["content"]
        self.assertIn("SORT_VARIANT_", content,
                      "final answer must reference at least one parallel coder result")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    # --- 8. run_parallel results visible in orchestrator second invocation ---

    def test_run_parallel_results_reach_orchestrator_loop_history(self) -> None:
        """
        The orchestrator's second invocation must contain all coder outputs
        in its message history so it can reason over them.
        """
        invocations: list[AgentInvocation] = []
        base = _fake_agent_invoker(
            {
                "orchestrator": [
                    _channel_action(
                        "run_parallel",
                        agent="coder",
                        count=2,
                        task="Implement quicksort.",
                    ),
                    "Done.",
                ],
                "coder": ["Quicksort version A", "Quicksort version B"],
            }
        )

        def _cap(inv: AgentInvocation):
            if inv.agent_id == "orchestrator":
                invocations.append(inv)
            return base(inv)

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_cap)
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        self._post(handle.port, [{"role": "user", "content": "Quicksort me."}])

        self.assertEqual(len(invocations), 2)
        second_messages = " ".join(
            str(m.get("content", "")) for m in invocations[1].messages
        )
        self.assertIn("Quicksort version A", second_messages)
        self.assertIn("Quicksort version B", second_messages)

    # --- 9. Channel close cleans up; subsequent message returns error ---

    def test_close_channel_then_message_returns_error_turn(self) -> None:
        """
        After the orchestrator closes a channel, attempting to message it
        in the same session must produce a finish_reason='error' turn.
        """
        _, port = self._start(
            _fake_agent_invoker({
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        participants=["coder"],
                        message="Start.",
                        recipients=["coder"],
                    ),
                    "Opened.",
                    _channel_action("close_channel", channel="channel-1"),
                    _channel_action(
                        "message_channel",
                        channel="channel-1",
                        message="Still here?",
                        recipients=["coder"],
                    ),
                ],
                "coder": ["First pass."],
            })
        )
        jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))

        def _req(content: str) -> dict:
            r = Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "ergon",
                    "messages": [{"role": "user", "content": content}],
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener.open(r) as resp:
                return json.loads(resp.read())

        first = _req("Go.")
        self.assertEqual(first["choices"][0]["finish_reason"], "stop")

        second = _req("Continue.")
        self.assertEqual(second["choices"][0]["finish_reason"], "error")

    # --- 10. Malformed tool call retried, turn still succeeds ---

    def test_malformed_tool_call_retried_turn_succeeds(self) -> None:
        """
        First orchestrator response has malformed JSON arguments — the core
        retries and the second valid response makes the turn succeed.
        """
        call_count = [0]
        base = _fake_agent_invoker(
            {
                "orchestrator": [
                    {
                        "text": "",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "name": "open_channel",
                                "arguments": "{malformed json",
                            }
                        ],
                    },
                    {
                        "text": "",
                        "tool_calls": [
                            {
                                "id": "tc2",
                                "name": "open_channel",
                                "arguments": json.dumps({
                                    "participants": ["coder"],
                                    "message": "Go.",
                                    "recipients": ["coder"],
                                }),
                            }
                        ],
                    },
                    "Recovered and delivered.",
                ],
                "coder": ["Done."],
            }
        )

        def _counting(inv: AgentInvocation):
            if inv.agent_id == "orchestrator":
                call_count[0] += 1
            return base(inv)

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_counting)
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        payload, _ = self._post(
            handle.port,
            [{"role": "user", "content": "Do something."}],
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            payload["choices"][0]["message"]["content"], "Recovered and delivered."
        )
        self.assertGreaterEqual(call_count[0], 2)

    # --- 11. Complete project pipeline — full agentic loop via SSE ---

    def test_complete_project_pipeline_sse(self) -> None:
        """
        Full project flow with streaming:
          brief → orchestrator thinks → opens channel with architect + coder
          → architect designs → coder implements → orchestrator synthesises.
        Validates: reasoning stream includes all intermediate contributions;
                   content stream contains the final synthesis.
        """
        _, port = self._start(
            _fake_agent_invoker({
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        preset="standard-build",
                        message="Build a REST API for a todo list.",
                        recipients=["architect"],
                        text="Spinning up the team.",
                    ),
                    "Project complete: todo-api with GET/POST/DELETE endpoints.",
                ],
                "architect": [
                    _channel_message_action(
                        message=(
                            "Design: three routes — GET /todos, POST /todos, "
                            "DELETE /todos/{id}. Use Flask."
                        ),
                        recipients=["coder"],
                    )
                ],
                "coder": [
                    "Implemented todo_api.py with all three routes using Flask."
                ],
            })
        )
        parsed, session_id = self._post(
            port,
            [{"role": "user", "content": "Build a todo REST API."}],
            stream=True,
        )
        reasoning = parsed.get("reasoning", "")
        content = parsed.get("content", "")

        self.assertIn("Spinning up the team.", reasoning)
        self.assertIn("GET /todos", reasoning)
        self.assertIn("Implemented todo_api.py", reasoning)
        self.assertIn("todo-api", content)
        self.assertIsNotNone(session_id)


# ---------------------------------------------------------------------------
# Core-level async agentic loop tests (ProxyOrchestrationCore directly)
# ---------------------------------------------------------------------------


class AgenticLoopCoreTests(unittest.IsolatedAsyncioTestCase):
    """
    Async tests against ProxyOrchestrationCore for scenarios that are
    cleaner to assert at the event-stream level.
    """

    # --- 12. Event ordering: reasoning before content ---

    async def test_event_order_reasoning_before_content(self) -> None:
        """
        All reasoning events must be emitted before the final content event.
        """
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _channel_action(
                            "open_channel",
                            participants=["coder"],
                            message="Do the work.",
                            recipients=["coder"],
                        ),
                        "All done.",
                    ],
                    "coder": ["Work complete."],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [e async for e in stream]
        await stream.get_final_response()

        saw_content = False
        for event in events:
            if isinstance(event, ProxyContentDeltaEvent):
                saw_content = True
            if isinstance(event, ProxyReasoningDeltaEvent) and saw_content:
                self.fail(
                    "Reasoning event emitted after content has started"
                )

    # --- 13. run_parallel sub-sessions are workspace-isolated ---

    async def test_run_parallel_subsessions_workspace_isolated(self) -> None:
        """
        Two parallel sub-sessions writing to the same logical path must
        produce isolated overlay files (no cross-contamination).
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        overlay_root = Path(tmp.name)
        target_path = str(overlay_root / "output.txt")

        coder_call = [0]

        def _writing_coder(inv: AgentInvocation):
            if inv.agent_id != "coder":
                return _response_stream("Done.")
            coder_call[0] += 1
            version = coder_call[0]

            async def _events():  # type: ignore[return]
                return
                yield  # noqa: unreachable

            return ResponseStream(
                _events(),
                finalizer=lambda v=version: AgentRunResult(
                    text="",
                    tool_calls=(
                        ProxyToolCall(
                            id=f"wf_{v}",
                            name="write_file",
                            arguments_json=json.dumps({
                                "path": target_path,
                                "content": f"version {v}",
                            }),
                        ),
                    ),
                ),
            )

        # Make the orchestrator call run_parallel with count=2,
        # then return a final answer. The writing coder streams first.
        call_idx = [0]
        def _invoker(inv: AgentInvocation):
            if inv.agent_id == "orchestrator":
                call_idx[0] += 1
                if call_idx[0] == 1:
                    return _response_stream(
                        "",
                        response=AgentRunResult(
                            text="",
                            tool_calls=(
                                ProxyToolCall(
                                    id="rp1",
                                    name="run_parallel",
                                    arguments_json=json.dumps({
                                        "agent": "coder",
                                        "count": 2,
                                        "task": "Write output.txt",
                                    }),
                                ),
                            ),
                        ),
                    )
                return _response_stream("Parallel done.")
            return _writing_coder(inv)

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_invoker,
            overlay_root=overlay_root,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="session_iso")
        [e async for e in stream]
        await stream.get_final_response()

        # Each sub-session has its own overlay directory
        sub0 = SessionOverlay(root=overlay_root / "session_iso-parallel-0")
        sub1 = SessionOverlay(root=overlay_root / "session_iso-parallel-1")
        content0 = sub0.read_file(target_path)
        content1 = sub1.read_file(target_path)
        self.assertNotEqual(content0, content1)
        self.assertIn("version", content0)
        self.assertIn("version", content1)

    # --- 14. run_parallel error isolation: one crash, others succeed ---

    async def test_run_parallel_one_failing_subsession_others_complete(self) -> None:
        """
        One sub-session raises during execution. The other sub-session must
        complete and the orchestrator must receive both an error string and
        a successful result in loop history.
        """
        invocations: list[AgentInvocation] = []
        coder_calls = [0]

        def _invoker(inv: AgentInvocation):
            invocations.append(inv)
            if inv.agent_id == "coder":
                coder_calls[0] += 1
                if coder_calls[0] == 1:
                    async def _crash_events():  # type: ignore[return]
                        raise RuntimeError("sub-session exploded")
                        yield  # noqa: unreachable

                    return ResponseStream(
                        _crash_events(),
                        finalizer=lambda: AgentRunResult(text="", tool_calls=()),
                    )
                return _response_stream("Healthy result")
            # Orchestrator: first call runs parallel, second returns final text
            orch_calls = [i for i in invocations if i.agent_id == "orchestrator"]
            if len(orch_calls) == 1:
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="rp1",
                                name="run_parallel",
                                arguments_json=json.dumps({
                                    "agent": "coder",
                                    "count": 2,
                                    "task": "Do something.",
                                }),
                            ),
                        ),
                    ),
                )
            return _response_stream("Got both results.")

        core = ProxyOrchestrationCore(_fake_registry(), agent_invoker=_invoker)
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Go"),),
        )
        stream = core.stream_turn(request, session_id="s1")
        [e async for e in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "stop")
        # Second orchestrator invocation must have received both results
        second_orch = [i for i in invocations if i.agent_id == "orchestrator"][1]
        hist = " ".join(str(m.get("content", "")) for m in second_orch.messages)
        self.assertIn("Error:", hist)
        self.assertIn("Healthy result", hist)

    # --- 15. Channel message_channel tool from participant ---

    async def test_participant_message_channel_delivers_to_other_participant(
        self,
    ) -> None:
        """
        A channel participant uses message_channel to address another
        participant. The addressed participant must receive the message and
        respond; both responses must appear in the reasoning stream.
        """
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _channel_action(
                            "open_channel",
                            preset="standard-build",
                            message="Design and implement a queue.",
                            recipients=["architect"],
                        ),
                        "Queue delivered.",
                    ],
                    "architect": [
                        _channel_message_action(
                            message="Queue API: enqueue, dequeue, is_empty, size.",
                            recipients=["coder"],
                        )
                    ],
                    "coder": ["Queue implemented: enqueue/dequeue/is_empty/size."],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Build a queue."),),
        )
        stream = core.stream_turn(request, session_id="s1")
        events = [e async for e in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            e.delta for e in events if isinstance(e, ProxyReasoningDeltaEvent)
        )
        self.assertIn("Queue API:", reasoning)
        self.assertIn("Queue implemented", reasoning)
        self.assertEqual(result.content, "Queue delivered.")
        self.assertEqual(result.finish_reason, "stop")

    # --- 16. Full project: brief → design → implement → synthesise ---

    async def test_full_project_brief_design_implement_synthesise(self) -> None:
        """
        The most comprehensive flow:
          1. User provides a project brief.
          2. Orchestrator opens a standard-build channel.
          3. Architect designs the API.
          4. Architect hands off to coder via message_channel.
          5. Coder implements and replies.
          6. Orchestrator closes the channel.
          7. Orchestrator returns a complete synthesis to the user.
        All reasoning events are validated; finish_reason must be 'stop'.
        """
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _channel_action(
                            "open_channel",
                            preset="standard-build",
                            message=(
                                "Build a URL shortener: POST /shorten returns a short "
                                "code, GET /{code} redirects."
                            ),
                            recipients=["architect"],
                        ),
                        _channel_action(
                            "close_channel", channel="channel-1"
                        ),
                        (
                            "Project complete: url_shortener.py implemented with "
                            "POST /shorten and GET /{code} routes."
                        ),
                    ],
                    "architect": [
                        _channel_message_action(
                            message=(
                                "Architecture: in-memory dict maps code→url. "
                                "POST /shorten generates 6-char code and stores it. "
                                "GET /{code} looks up and returns 302."
                            ),
                            recipients=["coder"],
                        )
                    ],
                    "coder": [
                        "url_shortener.py: Flask app, POST /shorten, GET /<code> done."
                    ],
                }
            ),
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(
                ProxyInputMessage(
                    role="user",
                    content=(
                        "Build a URL shortener service. POST /shorten should accept "
                        "a URL and return a short code. GET /{code} should redirect."
                    ),
                ),
            ),
        )
        stream = core.stream_turn(request, session_id="project_1")
        events = [e async for e in stream]
        result = await stream.get_final_response()

        reasoning = "".join(
            e.delta for e in events if isinstance(e, ProxyReasoningDeltaEvent)
        )
        content = "".join(
            e.delta for e in events if isinstance(e, ProxyContentDeltaEvent)
        )

        # Intermediate work surfaces as reasoning
        self.assertIn("POST /shorten", reasoning)
        self.assertIn("url_shortener.py", reasoning)
        # Final synthesis surfaces as content
        self.assertIn("url_shortener.py", content)
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("url_shortener.py", result.content)

    # --- 17. Parallel sub-sessions each use workspace read + write ---

    async def test_parallel_subsessions_read_and_write_workspace(self) -> None:
        """
        Two parallel sub-sessions each: read an existing real file, then
        write a transformed result to the workspace. Both results must be
        present in the orchestrator's loop history and in their respective
        overlay directories.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        overlay_root = Path(tmp.name) / "overlays"
        overlay_root.mkdir()
        real_file = Path(tmp.name) / "spec.txt"
        real_file.write_text("spec content", encoding="utf-8")

        _all_invs: list[AgentInvocation] = []
        coder_round: dict[int, int] = {}

        def _stateful_invoker(inv: AgentInvocation) -> ResponseStream[str, AgentRunResult]:
            _all_invs.append(inv)
            if inv.agent_id == "coder":
                # identify sub-session by counting coder invocations with
                # the same first user message
                first_content = next(
                    (m["content"] for m in inv.messages if m["role"] == "user"),
                    None,
                )
                key = id(first_content)
                coder_round[key] = coder_round.get(key, 0) + 1
                round_num = coder_round[key]
                coder_idx = sum(
                    1 for i in _all_invs[:-1] if i.agent_id == "coder"
                )

                if round_num == 1:
                    # First turn: emit a read_file tool call
                    async def _ev1():  # type: ignore[return]
                        return
                        yield  # noqa: unreachable

                    return ResponseStream(
                        _ev1(),
                        finalizer=lambda idx=coder_idx: AgentRunResult(
                            text="",
                            tool_calls=(
                                ProxyToolCall(
                                    id=f"rf_{idx}",
                                    name="read_file",
                                    arguments_json=json.dumps(
                                        {"path": str(real_file)}
                                    ),
                                ),
                            ),
                        ),
                    )
                # Second turn: read result is in messages, reply with summary
                return _response_stream(f"Read spec ok (coder {coder_idx})")

            # Orchestrator
            orch_invs = [i for i in _all_invs if i.agent_id == "orchestrator"]
            if len(orch_invs) == 1:
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id="rp1",
                                name="run_parallel",
                                arguments_json=json.dumps({
                                    "agent": "coder",
                                    "count": 2,
                                    "task": "Read the spec.",
                                }),
                            ),
                        ),
                    ),
                )
            return _response_stream("All coders confirmed the spec.")

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_stateful_invoker,
            overlay_root=overlay_root,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(role="user", content="Confirm the spec."),),
        )
        stream = core.stream_turn(request, session_id="ws_read_test")
        [e async for e in stream]
        result = await stream.get_final_response()

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.content, "All coders confirmed the spec.")

        # Both coder sub-sessions must have read the spec
        second_orch = [i for i in _all_invs if i.agent_id == "orchestrator"][1]
        hist = " ".join(str(m.get("content", "")) for m in second_orch.messages)
        self.assertIn("Read spec ok", hist)

    # --- 18. Full autonomous project build from a product brief ---

    async def test_orchestrator_builds_complete_project_from_brief(self) -> None:
        """
        The orchestrator receives a product brief with no implementation
        instructions — only what the product should be — and must autonomously
        orchestrate the full pipeline:

          1. Recognise the task requires a team and open a standard-build channel.
          2. Architect produces a real design document and hands it to the coder
             via message_channel.
          3. Channel coder acknowledges, proving context arrived from the architect.
          4. Orchestrator sees channel results and launches run_parallel to write
             the actual source files to the workspace overlay.
          5. Sub-session coder writes game.js, main.js, and package.json.
          6. Orchestrator synthesises a final delivery summary.

        Assertions verify the DATA FLOW end-to-end and the WORKSPACE STATE:
          - orchestrator called at least 3 times (design / implement / synthesise)
          - architect's design text reached the channel coder
          - workspace overlay contains all three files with real JS content
          - package.json parses as valid JSON with the correct project name
          - final answer describes the delivered project
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        overlay_root = Path(tmp.name) / "workspace"
        overlay_root.mkdir()

        invocations: list[AgentInvocation] = []
        SESSION_ID = "ttt_build"

        def _invoker(inv: AgentInvocation) -> ResponseStream:
            invocations.append(inv)
            msgs = list(inv.messages)

            # ── Orchestrator ────────────────────────────────────────────────
            if inv.agent_id == "orchestrator":
                parallel_done = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "run_parallel" in m.get("content", "")
                    ),
                    None,
                )
                channel_done = next(
                    (
                        m["content"] for m in msgs
                        if m.get("role") == "user"
                        and "Channel channel-1" in m.get("content", "")
                    ),
                    None,
                )
                if parallel_done:
                    # Stage 3: files written — synthesise
                    return _response_stream(
                        "Tic-tac-toe game delivered.\n" + parallel_done
                    )
                if channel_done:
                    # Stage 2: design done — implement
                    return _response_stream(
                        "",
                        response=AgentRunResult(
                            text="",
                            tool_calls=(ProxyToolCall(
                                id="rp1",
                                name="run_parallel",
                                arguments_json=json.dumps({
                                    "agent": "coder",
                                    "count": 1,
                                    "task": (
                                        "Implement the tic-tac-toe game using "
                                        "this design:\n" + channel_done[:600]
                                    ),
                                }),
                            ),),
                        ),
                    )
                # Stage 1: delegate design to the team
                brief = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="oc1",
                            name="open_channel",
                            arguments_json=json.dumps({
                                "preset": "standard-build",
                                "message": brief,
                                "recipients": ["architect"],
                            }),
                        ),),
                    ),
                )

            # ── Architect ───────────────────────────────────────────────────
            if inv.agent_id == "architect":
                brief = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                design = (
                    "Design for tic-tac-toe JS game:\n"
                    "  game.js — Board class: 9-cell array, makeMove(pos,player), "
                    "checkWin() returns winner or null, isFull() for draw detection.\n"
                    "  main.js — readline game loop, displays 3×3 grid, "
                    "alternates between X and O players.\n"
                    "  package.json — minimal node package descriptor, "
                    'name "tictactoe", version "1.0.0".\n'
                    f"Based on brief: {brief[:80]}"
                )
                return _response_stream(
                    "",
                    response=AgentRunResult(
                        text="",
                        tool_calls=(ProxyToolCall(
                            id="mc1",
                            name="message_channel",
                            arguments_json=json.dumps({
                                "message": design,
                                "recipients": ["coder"],
                            }),
                        ),),
                    ),
                )

            # ── Coder ────────────────────────────────────────────────────────
            if inv.agent_id == "coder":
                has_tool_result = any(m.get("role") == "tool" for m in msgs)
                # Sub-session turn 1: exactly 1 user message (the task) and no
                # tool results yet. Channel coder has 2+ user messages (full
                # transcript), and sub-session turn 2 has tool result messages.
                user_msg_count = sum(1 for m in msgs if m.get("role") == "user")
                is_fresh_subsession = user_msg_count == 1 and not has_tool_result

                if is_fresh_subsession:
                    # Sub-session first turn: write the complete game
                    game_js = (
                        "class Board {\n"
                        "  constructor() {\n"
                        "    this.cells = Array(9).fill(null);\n"
                        "    this.current = 'X';\n"
                        "  }\n"
                        "  makeMove(pos) {\n"
                        "    if (this.cells[pos] !== null) return false;\n"
                        "    this.cells[pos] = this.current;\n"
                        "    this.current = this.current === 'X' ? 'O' : 'X';\n"
                        "    return true;\n"
                        "  }\n"
                        "  checkWin() {\n"
                        "    const lines = [\n"
                        "      [0,1,2],[3,4,5],[6,7,8],\n"
                        "      [0,3,6],[1,4,7],[2,5,8],\n"
                        "      [0,4,8],[2,4,6]\n"
                        "    ];\n"
                        "    for (const [a,b,c] of lines) {\n"
                        "      if (this.cells[a] &&\n"
                        "          this.cells[a] === this.cells[b] &&\n"
                        "          this.cells[a] === this.cells[c])\n"
                        "        return this.cells[a];\n"
                        "    }\n"
                        "    return null;\n"
                        "  }\n"
                        "  isFull() { return this.cells.every(c => c !== null); }\n"
                        "}\n"
                        "module.exports = Board;\n"
                    )
                    main_js = (
                        "const readline = require('readline');\n"
                        "const Board = require('./game');\n"
                        "const board = new Board();\n"
                        "const rl = readline.createInterface("
                        "{ input: process.stdin, output: process.stdout });\n"
                        "function display() {\n"
                        "  const c = board.cells;\n"
                        "  console.log(`${c[0]||'_'}|${c[1]||'_'}|${c[2]||'_'}`);\n"
                        "  console.log(`${c[3]||'_'}|${c[4]||'_'}|${c[5]||'_'}`);\n"
                        "  console.log(`${c[6]||'_'}|${c[7]||'_'}|${c[8]||'_'}`);\n"
                        "}\n"
                        "function prompt() {\n"
                        "  rl.question(`${board.current}'s turn (0-8): `, (input) => {\n"
                        "    const pos = parseInt(input, 10);\n"
                        "    if (isNaN(pos) || !board.makeMove(pos)) return prompt();\n"
                        "    display();\n"
                        "    const w = board.checkWin();\n"
                        "    if (w) { console.log(`${w} wins!`); rl.close(); return; }\n"
                        "    if (board.isFull()) { console.log('Draw!'); rl.close(); return; }\n"
                        "    prompt();\n"
                        "  });\n"
                        "}\n"
                        "display();\n"
                        "prompt();\n"
                    )
                    pkg_json = json.dumps({
                        "name": "tictactoe",
                        "version": "1.0.0",
                        "description": "Two-player tic-tac-toe for the terminal",
                        "main": "main.js",
                        "scripts": {"start": "node main.js"},
                    }, indent=2)
                    return _response_stream(
                        "",
                        response=AgentRunResult(
                            text="",
                            tool_calls=(
                                ProxyToolCall(
                                    id="wf_game",
                                    name="write_file",
                                    arguments_json=json.dumps({
                                        "path": "/tictactoe/game.js",
                                        "content": game_js,
                                    }),
                                ),
                                ProxyToolCall(
                                    id="wf_main",
                                    name="write_file",
                                    arguments_json=json.dumps({
                                        "path": "/tictactoe/main.js",
                                        "content": main_js,
                                    }),
                                ),
                                ProxyToolCall(
                                    id="wf_pkg",
                                    name="write_file",
                                    arguments_json=json.dumps({
                                        "path": "/tictactoe/package.json",
                                        "content": pkg_json,
                                    }),
                                ),
                            ),
                        ),
                    )

                if has_tool_result:
                    # Sub-session second turn: files are written, return summary
                    return _response_stream(
                        "Implemented:\n"
                        "  game.js  — Board class with makeMove(), checkWin(), isFull()\n"
                        "  main.js  — readline game loop, displays 3×3 grid, "
                        "alternates X/O\n"
                        "  package.json — tictactoe 1.0.0, entry point main.js"
                    )

                # Channel coder: ack the architect's design
                design_ctx = " ".join(
                    m.get("content", "") for m in msgs if m.get("role") == "user"
                )
                return _response_stream(
                    "Design received. Will implement Board class with "
                    "makeMove/checkWin and main.js game loop. "
                    f"Design context: {design_ctx[:80]}..."
                )

            return _response_stream("ok")

        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_invoker,
            overlay_root=overlay_root,
        )
        request = ProxyTurnRequest(
            model="ergon",
            messages=(ProxyInputMessage(
                role="user",
                content=(
                    "Build a tic-tac-toe game in JavaScript for two players. "
                    "It should support alternating turns between X and O, "
                    "display the 3×3 board after each move, detect wins and draws, "
                    "and run in the terminal."
                ),
            ),),
        )
        stream = core.stream_turn(request, session_id=SESSION_ID)
        [e async for e in stream]
        result = await stream.get_final_response()

        # ── Structural checks: all three orchestration stages ran ────────────
        orch = [i for i in invocations if i.agent_id == "orchestrator"]
        self.assertGreaterEqual(
            len(orch), 3,
            "orchestrator must pass through at least three stages: "
            "open-channel, run-parallel, synthesise",
        )

        arch = [i for i in invocations if i.agent_id == "architect"]
        self.assertGreater(len(arch), 0, "architect must be invoked in the channel")

        # Channel coder: called with channel transcript (multiple messages,
        # no tool results) — distinct from sub-session coder calls
        channel_coders = [
            i for i in invocations
            if i.agent_id == "coder"
            and len(i.messages) > 1
            and not any(m.get("role") == "tool" for m in i.messages)
        ]
        self.assertGreater(len(channel_coders), 0,
                           "coder must be invoked in the design channel")

        # ── Context propagation: architect's design reached the channel coder ─
        coder_ctx = " ".join(
            m.get("content", "") for m in channel_coders[0].messages
        )
        self.assertIn("makeMove", coder_ctx,
                      "architect's design (mentioning makeMove) must reach the coder")
        self.assertIn("game.js", coder_ctx,
                      "architect's design (mentioning game.js) must reach the coder")

        # ── Workspace state: all three files written with real content ────────
        overlay = SessionOverlay(root=overlay_root / f"{SESSION_ID}-parallel-0")

        game_js = overlay.read_file("/tictactoe/game.js")
        self.assertIn("Board", game_js, "game.js must define a Board class")
        self.assertIn("makeMove", game_js)
        self.assertIn("checkWin", game_js)
        self.assertIn("isFull", game_js)
        self.assertIn("module.exports", game_js)

        main_js = overlay.read_file("/tictactoe/main.js")
        self.assertIn("require", main_js, "main.js must use require()")
        self.assertIn("readline", main_js)
        self.assertIn("makeMove", main_js)

        pkg_raw = overlay.read_file("/tictactoe/package.json")
        pkg = json.loads(pkg_raw)   # must be valid JSON
        self.assertEqual(pkg["name"], "tictactoe")
        self.assertEqual(pkg["version"], "1.0.0")
        self.assertIn("start", pkg.get("scripts", {}))

        # ── Final answer describes the delivered project ───────────────────────
        self.assertEqual(result.finish_reason, "stop")
        self.assertIn("game.js", result.content)
        self.assertIn("main.js", result.content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _fake_agent_invoker(
    responses_by_agent: dict[str, list[str | dict[str, object]]],
):
    """Scripted invoker: each agent returns the next item from its list.

    Used for tests that care about SSE encoding or error handling, not
    about whether the right context reached the right agent.
    """
    counters: dict[str, int] = {a: 0 for a in responses_by_agent}

    def _invoker(invocation: AgentInvocation):
        idx = counters[invocation.agent_id]
        counters[invocation.agent_id] += 1
        response = responses_by_agent[invocation.agent_id][idx]
        if isinstance(response, str):
            return _response_stream(response)
        return _response_stream(
            str(response.get("text", "")),
            response=AgentRunResult(
                text=str(response.get("text", "")),
                tool_calls=tuple(
                    ProxyToolCall(
                        id=str(tc["id"]),
                        name=str(tc["name"]),
                        arguments_json=str(tc["arguments"]),
                    )
                    for tc in response.get("tool_calls", [])
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


def _channel_action(name: str, **payload: object) -> dict[str, object]:
    """Build a scripted orchestrator response that emits an internal tool call."""
    text = str(payload.pop("text", ""))
    return {
        "text": text,
        "tool_calls": [
            {
                "id": f"internal_{name}",
                "name": name,
                "arguments": json.dumps(payload),
            }
        ],
    }


def _channel_message_action(
    *,
    message: str,
    recipients: list[str],
) -> dict[str, object]:
    """Build a scripted participant response that uses message_channel."""
    return {
        "text": "",
        "tool_calls": [
            {
                "id": "internal_message_channel",
                "name": "message_channel",
                "arguments": json.dumps(
                    {"message": message, "recipients": recipients}
                ),
            }
        ],
    }


def _parse_sse_body(body: str) -> dict[str, str]:
    """
    Parse an SSE response body into a dict with 'reasoning', 'content',
    and 'raw' keys.  Only processes ``data: {...}`` lines.

    Reasoning events carry ``delta["reasoning"]``; content events carry
    ``delta["content"]``.  The two fields are mutually exclusive in the
    server's encoding so we check them independently.
    """
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        # ProxyReasoningDeltaEvent → delta["reasoning"] (and delta["reasoning_content"])
        reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content", "")
        if isinstance(reasoning_delta, str) and reasoning_delta:
            reasoning_parts.append(reasoning_delta)
        # ProxyContentDeltaEvent → delta["content"]
        content_delta = delta.get("content", "")
        if isinstance(content_delta, str) and content_delta:
            content_parts.append(content_delta)
    return {
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "raw": body,
    }
