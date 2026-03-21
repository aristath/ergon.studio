from __future__ import annotations

import unittest
from unittest.mock import patch

from agent_framework import ResponseStream
from openai import OpenAI

from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyToolCall,
    ProxyToolCallEvent,
    ProxyTurnResult,
)
from ergon_studio.proxy.server import start_proxy_server_in_thread
from ergon_studio.upstream import UpstreamSettings


class ProxyOpenAISDKTests(unittest.TestCase):
    def test_chat_completions_create_returns_parsed_content(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        response = client.chat.completions.create(
            model="ergon",
            messages=[{"role": "user", "content": "Hi"}],
        )

        self.assertEqual(response.choices[0].message.content, "Done.")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(response.model, "ergon")

    def test_chat_completions_create_returns_parsed_tool_calls(self) -> None:
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
        client = _client(handle.port)

        response = client.chat.completions.create(
            model="ergon",
            messages=[{"role": "user", "content": "Inspect main.py"}],
        )

        tool_call = response.choices[0].message.tool_calls[0]
        self.assertEqual(response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(tool_call.function.name, "read_file")
        self.assertEqual(tool_call.id, "call_1")

    def test_chat_completions_stream_returns_content_then_finish(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        stream = client.chat.completions.create(
            model="ergon",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
        chunks = list(stream)

        self.assertEqual(chunks[0].choices[0].delta.content, "Done.")
        self.assertEqual(chunks[-1].choices[0].finish_reason, "stop")

    def test_chat_completions_stream_returns_tool_call_deltas(self) -> None:
        call = ProxyToolCall(
            id="call_1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyToolCallEvent(call), ProxyFinishEvent("tool_calls")],
                tool_calls=(call,),
            ),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        chunks = list(
            client.chat.completions.create(
                model="ergon",
                messages=[{"role": "user", "content": "Inspect main.py"}],
                stream=True,
            )
        )

        self.assertEqual(
            chunks[0].choices[0].delta.tool_calls[0].function.name, "read_file"
        )
        self.assertEqual(chunks[-1].choices[0].finish_reason, "tool_calls")

    def test_responses_create_returns_parsed_content(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        response = client.responses.create(
            model="ergon",
            input="Hi",
        )

        self.assertEqual(response.output_text, "Done.")

    def test_responses_create_returns_parsed_function_calls(self) -> None:
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
        client = _client(handle.port)

        response = client.responses.create(
            model="ergon",
            input="Inspect main.py",
        )

        item = response.output[0]
        self.assertEqual(item.type, "function_call")
        self.assertEqual(item.name, "read_file")
        self.assertEqual(item.call_id, "call_1")

    def test_responses_stream_returns_expected_event_types(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        stream = client.responses.create(
            model="ergon",
            input="Hi",
            stream=True,
        )
        event_types = [event.type for event in stream]

        self.assertEqual(
            event_types,
            [
                "response.created",
                "response.output_text.delta",
                "response.output_text.done",
                "response.output_item.done",
                "response.completed",
            ],
        )

    def test_responses_stream_returns_function_call_added_events(self) -> None:
        call = ProxyToolCall(
            id="call_1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore(
                [ProxyToolCallEvent(call), ProxyFinishEvent("tool_calls")],
                tool_calls=(call,),
            ),
        )
        self.addCleanup(handle.close)
        client = _client(handle.port)

        events = list(
            client.responses.create(
                model="ergon",
                input="Inspect main.py",
                stream=True,
            )
        )

        self.assertEqual(events[1].type, "response.output_item.added")
        self.assertEqual(events[1].item.type, "function_call")
        self.assertEqual(events[1].item.name, "read_file")
        self.assertEqual(events[2].type, "response.function_call_arguments.delta")
        self.assertEqual(events[3].type, "response.function_call_arguments.done")
        self.assertEqual(events[4].type, "response.output_item.done")
        self.assertEqual(events[-1].type, "response.completed")

    def test_models_list_returns_upstream_model_ids(self) -> None:
        with patch(
            "ergon_studio.proxy.server.probe_upstream_models",
            return_value=[{"id": "gpt-oss-20b"}],
        ):
            handle = start_proxy_server_in_thread(
                host="127.0.0.1",
                port=0,
                core=_FakeCore(
                    [ProxyContentDeltaEvent("Done."), ProxyFinishEvent("stop")]
                ),
            )
            self.addCleanup(handle.close)
            client = _client(handle.port)

            models = client.models.list()

        self.assertEqual(models.data[0].id, "gpt-oss-20b")


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


def _client(port: int) -> OpenAI:
    return OpenAI(
        api_key="test",
        base_url=f"http://127.0.0.1:{port}/v1",
    )
