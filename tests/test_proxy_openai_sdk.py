from __future__ import annotations

import unittest

from agent_framework import ResponseStream
from openai import OpenAI

from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent, ProxyToolCall
from ergon_studio.proxy.models import ProxyTurnResult
from ergon_studio.proxy.server import start_proxy_server_in_thread


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
                        arguments_json="{\"path\":\"main.py\"}",
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
                        arguments_json="{\"path\":\"main.py\"}",
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
                "response.completed",
            ],
        )


class _FakeCore:
    def __init__(self, events, *, tool_calls=()):
        self._events = list(events)
        self._tool_calls = tuple(tool_calls)
        self.registry = type("Registry", (), {"config": {}, "agent_definitions": {}, "workflow_definitions": {}})()

    def stream_turn(self, request, *, created_at: int | None = None):
        events = list(self._events)
        content = "".join(event.delta for event in events if isinstance(event, ProxyContentDeltaEvent))
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
            ),
        )


def _client(port: int) -> OpenAI:
    return OpenAI(
        api_key="test",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


if __name__ == "__main__":
    unittest.main()
