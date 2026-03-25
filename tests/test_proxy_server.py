from __future__ import annotations

import http.cookiejar
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.agent_runner import AgentInvocation, AgentRunResult
from ergon_studio.proxy.core import ProxyOrchestrationCore, ProxyTurnResult
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyOutputItemRef,
    ProxyToolCall,
    ProxyToolCallEvent,
)
from ergon_studio.proxy.server import (
    MAX_REQUEST_BODY_BYTES,
    SESSION_HEADER_NAME,
    serve_proxy,
    start_proxy_server_in_thread,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.response_stream import ResponseStream
from ergon_studio.upstream import UpstreamSettings


class ProxyServerTests(unittest.TestCase):
    def test_serve_proxy_exits_cleanly_on_keyboard_interrupt(self) -> None:
        cleaned_up = [False]

        class _MockRunner:
            async def setup(self) -> None:
                raise KeyboardInterrupt

            async def cleanup(self) -> None:
                cleaned_up[0] = True

            @property
            def addresses(self) -> list[tuple[str, int]]:
                return [("127.0.0.1", 0)]

        with patch(
            "ergon_studio.proxy.server.web.AppRunner",
            return_value=_MockRunner(),
        ):
            serve_proxy(host="127.0.0.1", port=0, core=_FakeCore([]))

        self.assertTrue(cleaned_up[0])

    def test_models_endpoint_proxies_upstream_models(self) -> None:
        with patch(
            "ergon_studio.proxy.server.probe_upstream_models",
            return_value=[{"id": "gpt-oss-20b"}],
        ):
            handle = start_proxy_server_in_thread(
                host="127.0.0.1",
                port=0,
                core=_FakeCore([]),
            )
            self.addCleanup(handle.close)

            with urlopen(f"http://127.0.0.1:{handle.port}/v1/models") as response:
                payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "gpt-oss-20b")

    def test_chat_completions_return_session_header(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
            session_id = response.headers.get(SESSION_HEADER_NAME)

        self.assertEqual(payload["choices"][0]["message"]["content"], "Done.")
        self.assertIsNotNone(session_id)
        self.assertTrue(session_id.startswith("session_"))

    def test_chat_completions_streams_sse_chunks(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            body = response.read().decode("utf-8")

        self.assertIn('"chat.completion.chunk"', body)
        self.assertIn('"content":"Done."', body)
        self.assertIn("data: [DONE]", body)

    def test_chat_completions_respect_request_body_limit(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        oversized_message = "x" * (MAX_REQUEST_BODY_BYTES + 1)
        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": oversized_message}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises((HTTPError, URLError)) as exc:
            urlopen(request)
        if isinstance(exc.exception, HTTPError):
            self.assertEqual(exc.exception.code, 413)

    def test_session_cookie_is_reused_across_requests(self) -> None:
        core = ProxyOrchestrationCore(
            _fake_registry(),
            agent_invoker=_fake_agent_invoker(
                {
                    "orchestrator": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "internal_open",
                                    "name": "open_channel",
                                    "arguments": json.dumps(
                                        {
                                            "participants": ["coder"],
                                            "message": "Start",
                                            "recipients": ["coder"],
                                        }
                                    ),
                                }
                            ],
                        },
                        "Opened",
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "internal_message",
                                    "name": "message_channel",
                                    "arguments": json.dumps(
                                        {
                                            "channel": "channel-1",
                                            "message": "Continue",
                                            "recipients": ["coder"],
                                        }
                                    ),
                                }
                            ],
                        },
                        "Done",
                    ],
                    "coder": ["First pass", "Second pass"],
                }
            ),
        )
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=core,
        )
        self.addCleanup(handle.close)

        cookie_jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(cookie_jar))

        first_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Start"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(first_request) as response:
            first_payload = json.loads(response.read().decode("utf-8"))

        second_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Continue"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(second_request) as response:
            second_payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(first_payload["choices"][0]["message"]["content"], "Opened")
        self.assertEqual(second_payload["choices"][0]["message"]["content"], "Done")


class _FakeCore:
    def __init__(self, events):
        self.events = list(events)
        self.registry = _fake_registry()

    def stream_turn(self, _request, *, session_id=None):
        async def _events():
            for event in self.events:
                yield event

        final = ProxyTurnResult(
            finish_reason="stop",
            content="Done.",
            reasoning="",
            tool_calls=tuple(
                event.call
                for event in self.events
                if isinstance(event, ProxyToolCallEvent)
            ),
            output_items=(ProxyOutputItemRef(kind="content"),),
        )
        return ResponseStream(_events(), finalizer=lambda: final)


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
        },
        channel_presets={},
    )


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
