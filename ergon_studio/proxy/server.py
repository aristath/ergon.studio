from __future__ import annotations

import asyncio
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any
from uuid import uuid4

from ergon_studio.proxy import build_chat_completion_response, encode_chat_stream_done, encode_chat_stream_sse, parse_chat_completion_request


class ProxyHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, *, core, model_id: str = "ergon"):
        super().__init__(server_address, RequestHandlerClass)
        self.core = core
        self.model_id = model_id


class ProxyRequestHandler(BaseHTTPRequestHandler):
    server_version = "ergon-studio-proxy/0.1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "ergon-studio",
                        }
                    ],
                },
            )
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {self.path}")

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {self.path}")

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_chat_completions(self) -> None:
        try:
            payload = self._read_json_body()
            request = parse_chat_completion_request(payload)
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        completion_id = f"chatcmpl_{uuid4().hex}"
        created_at = int(time.time())
        if request.stream:
            self._send_sse_headers()
            try:
                asyncio.run(
                    self._stream_chat_completion(
                        request=request,
                        completion_id=completion_id,
                        created_at=created_at,
                    )
                )
            except BrokenPipeError:
                return
            return

        try:
            result = asyncio.run(self._run_chat_completion(request))
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")
            return
        self._send_json(
            HTTPStatus.OK,
            build_chat_completion_response(
                completion_id=completion_id,
                model=request.model,
                created_at=created_at,
                content=result.content,
                reasoning=result.reasoning,
                finish_reason=result.finish_reason,
            ),
        )

    async def _run_chat_completion(self, request):
        stream = self.server.core.stream_turn(request, created_at=int(time.time()))
        async for _event in stream:
            pass
        return await stream.get_final_response()

    async def _stream_chat_completion(self, *, request, completion_id: str, created_at: int) -> None:
        stream = self.server.core.stream_turn(request, created_at=created_at)
        async for event in stream:
            chunk = encode_chat_stream_sse(
                event,
                completion_id=completion_id,
                model=request.model,
                created_at=created_at,
            )
            self.wfile.write(chunk)
            self.wfile.flush()
        self.wfile.write(encode_chat_stream_done())
        self.wfile.flush()

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("missing Content-Length header")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if status < HTTPStatus.INTERNAL_SERVER_ERROR else "server_error",
                }
            },
        )


def serve_proxy(*, host: str, port: int, core, model_id: str = "ergon") -> None:
    server = ProxyHTTPServer((host, port), ProxyRequestHandler, core=core, model_id=model_id)
    try:
        server.serve_forever()
    finally:
        server.server_close()


class ProxyServerHandle:
    def __init__(self, server: ProxyHTTPServer, thread: threading.Thread) -> None:
        self.server = server
        self.thread = thread

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_proxy_server_in_thread(*, host: str, port: int, core, model_id: str = "ergon") -> ProxyServerHandle:
    server = ProxyHTTPServer((host, port), ProxyRequestHandler, core=core, model_id=model_id)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return ProxyServerHandle(server, thread)
