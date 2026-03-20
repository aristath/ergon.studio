from __future__ import annotations

import json
import unittest
from urllib.request import Request, urlopen

from agent_framework import ResponseStream

from ergon_studio.proxy.core import ProxyTurnResult
from ergon_studio.proxy.models import ProxyContentDeltaEvent, ProxyFinishEvent
from ergon_studio.proxy.server import start_proxy_server_in_thread


class ProxyServerTests(unittest.TestCase):
    def test_models_endpoint_lists_ergon_model(self) -> None:
        handle = start_proxy_server_in_thread(
            host="127.0.0.1",
            port=0,
            core=_FakeCore([]),
        )
        self.addCleanup(handle.close)

        with urlopen(f"http://127.0.0.1:{handle.port}/v1/models") as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "ergon")

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

        self.assertIn("\"chat.completion.chunk\"", body)
        self.assertIn("\"content\":\"Done.\"", body)
        self.assertIn("data: [DONE]", body)


class _FakeCore:
    def __init__(self, events):
        self._events = list(events)

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
            ),
        )


if __name__ == "__main__":
    unittest.main()
