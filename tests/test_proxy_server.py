from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_framework import ResponseStream

from ergon_studio.definitions import DefinitionDocument
from ergon_studio.proxy.continuation import (
    ContinuationState,
    encode_continuation_tool_call,
)
from ergon_studio.proxy.core import ProxyOrchestrationCore, ProxyTurnResult
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyOutputItemRef,
    ProxyReasoningDeltaEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
)
from ergon_studio.proxy.server import (
    MAX_REQUEST_BODY_BYTES,
    start_proxy_server_in_thread,
)
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import UpstreamSettings


class ProxyServerTests(unittest.TestCase):
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

    def test_models_endpoint_reuses_cached_upstream_listing(self) -> None:
        with patch(
            "ergon_studio.proxy.server.probe_upstream_models",
            return_value=[{"id": "gpt-oss-20b"}],
        ) as probe:
            handle = start_proxy_server_in_thread(
                host="127.0.0.1",
                port=0,
                core=_FakeCore([]),
            )
            self.addCleanup(handle.close)

            for _ in range(2):
                with urlopen(f"http://127.0.0.1:{handle.port}/v1/models") as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["data"][0]["id"], "gpt-oss-20b")

        self.assertEqual(probe.call_count, 1)

    def test_models_endpoint_uses_stale_cache_when_upstream_probe_fails(self) -> None:
        with (
            patch(
                "ergon_studio.proxy.server.MODEL_CACHE_TTL_SECONDS",
                0.0,
            ),
            patch(
                "ergon_studio.proxy.server.probe_upstream_models",
                side_effect=[
                    [{"id": "gpt-oss-20b"}],
                    RuntimeError("offline"),
                ],
            ) as probe,
        ):
            handle = start_proxy_server_in_thread(
                host="127.0.0.1",
                port=0,
                core=_FakeCore([]),
            )
            self.addCleanup(handle.close)

            with urlopen(f"http://127.0.0.1:{handle.port}/v1/models") as response:
                first_payload = json.loads(response.read().decode("utf-8"))
            with urlopen(f"http://127.0.0.1:{handle.port}/v1/models") as response:
                second_payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(first_payload["data"][0]["id"], "gpt-oss-20b")
        self.assertEqual(second_payload["data"][0]["id"], "gpt-oss-20b")
        self.assertEqual(probe.call_count, 2)

    def test_models_endpoint_returns_bad_gateway_when_upstream_listing_fails(
        self,
    ) -> None:
        with patch(
            "ergon_studio.proxy.server.probe_upstream_models",
            side_effect=RuntimeError("offline"),
        ):
            handle = start_proxy_server_in_thread(
                host="127.0.0.1",
                port=0,
                core=_FakeCore([]),
            )
            self.addCleanup(handle.close)

            with self.assertRaises(HTTPError) as exc:
                urlopen(f"http://127.0.0.1:{handle.port}/v1/models")

        self.assertEqual(exc.exception.code, 502)

    def test_chat_completions_echo_requested_model(self) -> None:
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
                    "model": "host-selected-name",
                    "messages": [{"role": "user", "content": "Hi"}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["model"], "host-selected-name")

    def test_responses_echo_requested_model(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "gpt-oss-20b",
                    "input": "Hi",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["model"], "gpt-oss-20b")

    def test_chat_completions_returns_non_stream_response(self) -> None:
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
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Done.")

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

    def test_chat_completions_streams_error_chunks_for_unexpected_failures(
        self,
    ) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FailingCore(RuntimeError("planner exploded")),
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

        self.assertIn('RuntimeError: planner exploded', body)
        self.assertIn('"finish_reason":"error"', body)
        self.assertIn("data: [DONE]", body)

    def test_responses_returns_non_stream_response(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["output_text"], "Done.")

    def test_chat_completions_returns_server_error_for_failed_turn(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyContentDeltaEvent("provider exploded"), ProxyFinishEvent("error")]
            ),
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

        with self.assertRaises(HTTPError) as exc:
            urlopen(request)

        payload = json.loads(exc.exception.read().decode("utf-8"))
        self.assertEqual(exc.exception.code, 500)
        self.assertEqual(payload["error"]["message"], "provider exploded")

    def test_responses_returns_server_error_for_failed_turn(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyContentDeltaEvent("provider exploded"), ProxyFinishEvent("error")]
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(HTTPError) as exc:
            urlopen(request)

        payload = json.loads(exc.exception.read().decode("utf-8"))
        self.assertEqual(exc.exception.code, 500)
        self.assertEqual(payload["error"]["message"], "provider exploded")

    def test_responses_streams_response_events(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [
                    ProxyReasoningDeltaEvent("Plan."),
                    ProxyContentDeltaEvent("Done."),
                    ProxyFinishEvent("stop"),
                ]
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            body = response.read().decode("utf-8")

        self.assertIn('"type":"response.created"', body)
        self.assertIn('"type":"response.reasoning_text.delta"', body)
        self.assertIn('"type":"response.output_text.delta"', body)
        self.assertIn('"type":"response.reasoning_text.done"', body)
        self.assertIn('"type":"response.output_item.done"', body)
        self.assertIn('"type":"response.completed"', body)

    def test_responses_stream_uses_failed_terminal_event_for_errors(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyContentDeltaEvent("provider exploded"), ProxyFinishEvent("error")]
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            body = response.read().decode("utf-8")

        self.assertIn('"type":"response.failed"', body)
        self.assertNotIn('"type":"response.completed"', body)

    def test_responses_stream_uses_consistent_output_indexes(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [
                    ProxyReasoningDeltaEvent("Plan."),
                    ProxyToolCallEvent(
                        ProxyToolCall(
                            id="call_1",
                            name="read_file",
                            arguments_json='{"path":"main.py"}',
                        )
                    ),
                    ProxyFinishEvent("tool_calls"),
                ],
                tool_calls=(
                    ProxyToolCall(
                        id="call_1",
                        name="read_file",
                        arguments_json='{"path":"main.py"}',
                    ),
                ),
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Inspect main.py",
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payloads = [
                json.loads(line[6:])
                for line in response.read().decode("utf-8").splitlines()
                if line.startswith("data: {")
            ]

        reasoning = next(
            item for item in payloads if item["type"] == "response.reasoning_text.delta"
        )
        tool = next(
            item for item in payloads if item["type"] == "response.output_item.added"
        )
        tool_arguments_done = next(
            item
            for item in payloads
            if item["type"] == "response.function_call_arguments.done"
        )
        tool_done = next(
            item
            for item in payloads
            if item["type"] == "response.output_item.done"
            and item["item"]["type"] == "function_call"
        )
        self.assertEqual(reasoning["output_index"], 0)
        self.assertEqual(tool["output_index"], 1)
        self.assertEqual(tool_arguments_done["output_index"], 1)
        self.assertEqual(tool_done["output_index"], 1)
        self.assertEqual(tool["item"]["id"], tool_done["item"]["id"])
        self.assertEqual(tool_arguments_done["item_id"], tool_done["item"]["id"])

    def test_chat_completions_returns_tool_calls(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyFinishEvent("tool_calls")],
                tool_calls=(
                    ProxyToolCall(
                        id="call_1",
                        name="read_file",
                        arguments_json='{"path":"main.py"}',
                    ),
                ),
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Inspect main.py"}],
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload["choices"][0]["message"]
        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")

    def test_responses_returns_function_calls(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyFinishEvent("tool_calls")],
                tool_calls=(
                    ProxyToolCall(
                        id="call_1",
                        name="read_file",
                        arguments_json='{"path":"main.py"}',
                    ),
                ),
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Inspect main.py",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["call_id"], "call_1")

    def test_responses_keep_content_before_tool_calls(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [
                    ProxyContentDeltaEvent("Draft first."),
                    ProxyFinishEvent("tool_calls"),
                ],
                tool_calls=(
                    ProxyToolCall(
                        id="call_1",
                        name="read_file",
                        arguments_json='{"path":"main.py"}',
                    ),
                ),
                output_items=(
                    ProxyOutputItemRef(kind="content"),
                    ProxyOutputItemRef(kind="tool_call", call_id="call_1"),
                ),
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Inspect main.py",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(
            [item["type"] for item in payload["output"]], ["message", "function_call"]
        )

    def test_chat_completions_resume_full_tool_loop(self) -> None:
        core = ProxyOrchestrationCore(
            _proxy_registry(),
            agent_builder=_proxy_agent_builder(
                {
                    "orchestrator": [
                        (
                            '{"mode":"workflow","workflow_id":"standard-build",'
                            '"goal":"Build calculator"}'
                        ),
                        '{"mode":"workflow","workflow_id":"standard-build"}',
                        '{"mode":"finish"}',
                        "Workflow final summary",
                    ],
                    "architect": [
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_arch_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                }
                            ],
                        },
                        "Architecture plan",
                    ],
                    "coder": ["Built feature"],
                }
            ),
        )
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        first_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Build calculator"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(first_request) as response:
            first_payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(first_payload["choices"][0]["finish_reason"], "tool_calls")
        tool_call = first_payload["choices"][0]["message"]["tool_calls"][0]

        second_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [
                        {"role": "user", "content": "Build calculator"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool_call],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "print('current main')",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(second_request) as response:
            second_payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(second_payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            second_payload["choices"][0]["message"]["content"], "Workflow final summary"
        )

    def test_chat_completions_streams_tool_calls_with_separate_finish_chunk(
        self,
    ) -> None:
        core = ProxyOrchestrationCore(
            _proxy_registry(),
            agent_builder=_proxy_agent_builder(
                {
                    "orchestrator": [
                        '{"mode":"act"}',
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                }
                            ],
                        },
                    ],
                    "architect": [],
                    "coder": [],
                }
            ),
        )
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": "Inspect main.py"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            chunks = [
                json.loads(line[6:])
                for line in response.read().decode("utf-8").splitlines()
                if line.startswith("data: {")
            ]

        tool_chunk = next(
            chunk for chunk in chunks if chunk["choices"][0]["delta"].get("tool_calls")
        )
        finish_chunk = chunks[-1]
        self.assertIsNone(tool_chunk["choices"][0]["finish_reason"])
        self.assertEqual(finish_chunk["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_resume_full_tool_loop(self) -> None:
        core = ProxyOrchestrationCore(
            _proxy_registry(),
            agent_builder=_proxy_agent_builder(
                {
                    "orchestrator": [
                        '{"mode":"act"}',
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                }
                            ],
                        },
                        "Final answer",
                    ],
                    "architect": [],
                    "coder": [],
                }
            ),
        )
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        first_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Inspect main.py",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(first_request) as response:
            first_payload = json.loads(response.read().decode("utf-8"))

        function_call = next(
            item for item in first_payload["output"] if item["type"] == "function_call"
        )

        second_request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "Inspect main.py",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": function_call["call_id"],
                            "output": "print('current main')",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(second_request) as response:
            second_payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(second_payload["output_text"], "Final answer")

    def test_responses_accepts_function_call_history_on_resume(self) -> None:
        core = ProxyOrchestrationCore(
            _proxy_registry(),
            agent_builder=_proxy_agent_builder(
                {
                    "orchestrator": [
                        "Final answer",
                    ],
                    "architect": [],
                    "coder": [],
                }
            ),
        )
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        tool_call_id = encode_continuation_tool_call(
            ProxyToolCall(
                id="call_1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
            state=ContinuationState(mode="act", agent_id="orchestrator"),
        ).id
        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "Inspect main.py",
                        },
                        {
                            "type": "function_call",
                            "call_id": tool_call_id,
                            "name": "read_file",
                            "arguments": '{"path":"main.py"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": "print('current main')",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["output_text"], "Final answer")

    def test_responses_stream_tool_calls_do_not_emit_empty_output_done(self) -> None:
        core = ProxyOrchestrationCore(
            _proxy_registry(),
            agent_builder=_proxy_agent_builder(
                {
                    "orchestrator": [
                        '{"mode":"act"}',
                        {
                            "text": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "read_file",
                                    "arguments": '{"path":"main.py"}',
                                }
                            ],
                        },
                    ],
                    "architect": [],
                    "coder": [],
                }
            ),
        )
        handle = start_proxy_server_in_thread(host="127.0.0.1", port=0, core=core)
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Inspect main.py",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payloads = [
                json.loads(line[6:])
                for line in response.read().decode("utf-8").splitlines()
                if line.startswith("data: {")
            ]

        event_types = [payload["type"] for payload in payloads]
        self.assertIn("response.output_item.added", event_types)
        self.assertNotIn("response.output_text.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")

    def test_responses_streams_failure_event_for_unexpected_failures(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FailingCore(RuntimeError("planner exploded")),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payloads = [
                json.loads(line[6:])
                for line in response.read().decode("utf-8").splitlines()
                if line.startswith("data: {")
            ]

        self.assertEqual(payloads[-1]["type"], "response.failed")
        self.assertEqual(
            payloads[-1]["response"]["error"]["message"],
            "RuntimeError: planner exploded",
        )

    def test_responses_stream_failure_preserves_sequence_order(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_LateFailingCore(
                [ProxyReasoningDeltaEvent("Plan.")],
                RuntimeError("planner exploded"),
            ),
        )
        self.addCleanup(handle.close)

        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/responses",
            data=json.dumps(
                {
                    "model": "ergon",
                    "input": "Hi",
                    "stream": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            payloads = [
                json.loads(line[6:])
                for line in response.read().decode("utf-8").splitlines()
                if line.startswith("data: {")
            ]

        self.assertEqual(payloads[0]["sequence_number"], 1)
        self.assertEqual(payloads[1]["sequence_number"], 2)
        self.assertEqual(payloads[-1]["type"], "response.failed")
        self.assertEqual(payloads[-1]["sequence_number"], 3)

    def test_chat_completions_reject_large_request_bodies(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)

        oversized_content = "x" * MAX_REQUEST_BODY_BYTES
        request = Request(
            f"http://127.0.0.1:{handle.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "ergon",
                    "messages": [{"role": "user", "content": oversized_content}],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises((HTTPError, URLError)) as exc:
            urlopen(request)

        error = exc.exception
        if isinstance(error, HTTPError):
            payload = json.loads(error.read().decode("utf-8"))
            self.assertEqual(error.code, 413)
            self.assertIn("request body exceeds", payload["error"]["message"])
            return
        self.assertIn("Broken pipe", str(error))


class _FakeCore:
    def __init__(self, events, *, tool_calls=(), output_items=()):
        self._events = list(events)
        self._tool_calls = tuple(tool_calls)
        self._output_items = tuple(output_items)
        self.registry = type(
            "Registry",
            (),
            {
                "upstream": UpstreamSettings(base_url="http://localhost:8080/v1"),
                "agent_definitions": {},
                "workflow_definitions": {},
            },
        )()

    def stream_turn(self, request, *, created_at: int | None = None):
        events = list(self._events)
        content = "".join(
            event.delta for event in events if isinstance(event, ProxyContentDeltaEvent)
        )
        finish_reason = "stop"
        for event in events:
            if isinstance(event, ProxyFinishEvent):
                finish_reason = event.reason

        async def _event_iter():
            for event in events:
                yield event

        return ResponseStream(
            _event_iter(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason=finish_reason,
                content=content,
                reasoning="",
                mode="act",
                tool_calls=self._tool_calls,
                output_items=self._output_items,
            ),
        )


class _FailingCore:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.registry = type(
            "Registry",
            (),
            {
                "upstream": UpstreamSettings(base_url="http://localhost:8080/v1"),
                "agent_definitions": {},
                "workflow_definitions": {},
            },
        )()

    def stream_turn(self, request, *, created_at: int | None = None):
        del request, created_at

        async def _event_iter():
            raise self._exc
            yield None  # pragma: no cover

        return ResponseStream(
            _event_iter(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason="error",
                content=str(self._exc),
                reasoning="",
                mode="act",
            ),
        )


class _LateFailingCore:
    def __init__(self, events, exc: Exception) -> None:
        self._events = list(events)
        self._exc = exc
        self.registry = type(
            "Registry",
            (),
            {
                "upstream": UpstreamSettings(base_url="http://localhost:8080/v1"),
                "agent_definitions": {},
                "workflow_definitions": {},
            },
        )()

    def stream_turn(self, request, *, created_at: int | None = None):
        del request, created_at

        async def _event_iter():
            for event in self._events:
                yield event
            raise self._exc
            yield None  # pragma: no cover

        return ResponseStream(
            _event_iter(),
            finalizer=lambda _updates: ProxyTurnResult(
                finish_reason="error",
                content=str(self._exc),
                reasoning="",
                mode="act",
            ),
        )


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


def _proxy_agent_builder(mapping: dict[str, list[object]]):
    remaining = {agent_id: list(responses) for agent_id, responses in mapping.items()}

    def _build(_registry, agent_id: str, **_kwargs):
        queue = remaining[agent_id]
        if not queue:
            raise AssertionError(f"no fake responses left for {agent_id}")
        return _FakeAgent([queue.pop(0)])

    return _build


def _proxy_registry() -> RuntimeRegistry:
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
