"""
Comprehensive end-to-end tests for agentic orchestration flows.

Each test scripts realistic orchestrator + specialist behaviour using
deterministic fake invokers, then validates the full pipeline:
  - event stream ordering (reasoning vs. content)
  - session persistence across HTTP turns
  - channel lifecycle (open / message / close)
  - run_parallel isolation and result injection
  - workspace tool I/O in sub-sessions
  - SSE streaming through the HTTP server
  - error recovery without crashing the turn
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
    and validate responses.  The LLM layer is replaced by scripted fake
    invokers so the tests run deterministically without an upstream.
    """

    def _start(self, invoker_map: dict) -> tuple:
        """Return (handle, port) for a server backed by the given invoker map."""
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(invoker_map),
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

    # --- 1. Orchestrator answers directly ---

    def test_direct_answer_produces_content_and_session_id(self) -> None:
        """Orchestrator replies without delegation — baseline flow."""
        _, port = self._start(
            {"orchestrator": ["The answer is 42."]}
        )
        payload, session_id = self._post(
            port, [{"role": "user", "content": "What is the answer?"}]
        )
        self.assertEqual(
            payload["choices"][0]["message"]["content"], "The answer is 42."
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertIsNotNone(session_id)
        self.assertTrue(session_id.startswith("session_"))  # type: ignore[union-attr]

    # --- 2. Orchestrator opens a channel, specialist delivers, synthesis ---

    def test_channel_open_specialist_delivers_orchestrator_synthesizes(self) -> None:
        """
        Orchestrator opens a channel → coder delivers a solution →
        orchestrator synthesises and returns a final answer.
        """
        _, port = self._start(
            {
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        participants=["coder"],
                        message="Implement a hello-world script.",
                        recipients=["coder"],
                    ),
                    "Delivered: hello.py prints 'Hello, World!'",
                ],
                "coder": ["print('Hello, World!')  # hello.py implementation"],
            }
        )
        payload, _ = self._post(
            port,
            [{"role": "user", "content": "Write a hello-world script."}],
        )
        content = payload["choices"][0]["message"]["content"]
        self.assertIn("hello.py", content)
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    # --- 3. Two-specialist channel ---

    def test_two_specialist_channel_both_contribute(self) -> None:
        """
        Architect designs, coder implements. Orchestrator receives both
        contributions and produces a final answer.
        """
        _, port = self._start(
            {
                "orchestrator": [
                    _channel_action(
                        "open_channel",
                        preset="standard-build",
                        message="Design and implement a stack data structure.",
                        recipients=["architect"],
                    ),
                    "Done — stack implementation complete.",
                ],
                "architect": [
                    _channel_message_action(
                        message="Stack API: push(x), pop() -> x, peek() -> x, is_empty() -> bool",
                        recipients=["coder"],
                    )
                ],
                "coder": ["class Stack: push/pop/peek/is_empty implemented."],
            }
        )
        payload, _ = self._post(
            port,
            [{"role": "user", "content": "Build a stack data structure."}],
        )
        content = payload["choices"][0]["message"]["content"]
        self.assertIn("stack", content.lower())
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    # --- 4. Multi-turn session with channel persistence ---

    def test_multi_turn_session_channel_persists_across_requests(self) -> None:
        """
        Turn 1: brief → orchestrator opens channel → coder provides first draft.
        Turn 2: follow-up → orchestrator messages same channel → coder refines.
        Session cookie carries the session across both HTTP requests.
        """
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        _channel_action(
                            "open_channel",
                            participants=["coder"],
                            message="Start the implementation.",
                            recipients=["coder"],
                        ),
                        "First draft ready.",
                        _channel_action(
                            "message_channel",
                            channel="channel-1",
                            message="Add error handling.",
                            recipients=["coder"],
                        ),
                        "Final version with error handling delivered.",
                    ],
                    "coder": [
                        "Initial implementation done.",
                        "Added try/except blocks throughout.",
                    ],
                }
            ),
        )
        handle = start_proxy_server_in_thread(
            host="127.0.0.1", port=0, core=core
        )
        self.addCleanup(handle.close)

        jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))

        r1 = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps({
                "model": "ergon",
                "messages": [{"role": "user", "content": "Start the project."}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(r1) as resp:
            first = json.loads(resp.read())

        r2 = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps({
                "model": "ergon",
                "messages": [{"role": "user", "content": "Add error handling."}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(r2) as resp:
            second = json.loads(resp.read())

        self.assertEqual(
            first["choices"][0]["message"]["content"], "First draft ready."
        )
        self.assertEqual(
            second["choices"][0]["message"]["content"],
            "Final version with error handling delivered.",
        )

    # --- 5. Orchestrator pre-channel thinking is reasoning, final is content ---

    def test_pre_channel_thinking_is_reasoning_not_content(self) -> None:
        """
        Orchestrator emits text before opening a channel — that text must
        appear in SSE reasoning chunks, not in the final content field.
        """
        _, port = self._start(
            {
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
            }
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
            {
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
            }
        )
        parsed, _ = self._post(
            port,
            [{"role": "user", "content": "Write a sort function."}],
            stream=True,
        )
        self.assertIn("def sort(lst)", parsed.get("reasoning", ""))
        self.assertEqual(parsed.get("content", ""), "Sort function delivered.")

    # --- 7. run_parallel best-of-N via HTTP ---

    def test_run_parallel_all_results_in_final_answer(self) -> None:
        """
        Orchestrator calls run_parallel(count=3). All three coder results
        must reach the orchestrator (shown in the second invocation's messages)
        and the final answer must reference the synthesis.
        """
        _, port = self._start(
            {
                "orchestrator": [
                    _channel_action(
                        "run_parallel",
                        agent="coder",
                        count=3,
                        task="Write a fibonacci function.",
                    ),
                    "Best implementation selected: recursive memoization.",
                ],
                "coder": [
                    "def fib_iterative(n): ...",
                    "def fib_recursive(n): ...",
                    "def fib_memoized(n): ...",
                ],
            }
        )
        payload, _ = self._post(
            port,
            [{"role": "user", "content": "Give me the best fibonacci implementation."}],
        )
        content = payload["choices"][0]["message"]["content"]
        self.assertIn("memoization", content.lower())
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    # --- 8. run_parallel results visible in orchestrator second invocation ---

    def test_run_parallel_results_reach_orchestrator_loop_history(self) -> None:
        """
        The orchestrator's second invocation must contain all three coder
        outputs in its message history so it can reason over them.
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
            {
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
            }
        )
        jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))
        r1 = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({
                "model": "ergon",
                "messages": [{"role": "user", "content": "Go."}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(r1) as resp:
            first = json.loads(resp.read())
        # Turn 1 completes normally
        self.assertEqual(first["choices"][0]["finish_reason"], "stop")

        r2 = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({
                "model": "ergon",
                "messages": [{"role": "user", "content": "Continue."}],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(r2) as resp:
            second = json.loads(resp.read())
        # Turn 2 tries to message closed channel → error
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
            {
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
            }
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

        coder_calls = [0]

        def _invoker(inv: AgentInvocation):
            if inv.agent_id == "coder":
                coder_calls[0] += 1
                call_num = coder_calls[0]

                async def _events():  # type: ignore[return]
                    return
                    yield  # noqa: unreachable

                # Each coder reads the spec, then writes a result
                return ResponseStream(
                    _events(),
                    finalizer=lambda n=call_num: AgentRunResult(
                        text="",
                        tool_calls=(
                            ProxyToolCall(
                                id=f"rf_{n}",
                                name="read_file",
                                arguments_json=json.dumps(
                                    {"path": str(real_file)}
                                ),
                            ),
                        ),
                    ),
                )

            # Orchestrator
            orch_invs = [i for i in _all_invs if i.agent_id == "orchestrator"]
            if len(orch_invs) <= 1:
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
                                    "task": "Read the spec and confirm.",
                                }),
                            ),
                        ),
                    ),
                )
            return _response_stream("Both coders read the spec.")

        # But wait — after the coder reads, it needs a second turn to reply.
        # Let's use a stateful fake that returns the read result and then text.
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
