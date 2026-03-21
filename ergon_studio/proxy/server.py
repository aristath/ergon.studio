from __future__ import annotations

import asyncio
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from uuid import uuid4

from ergon_studio.proxy.chat_adapter import (
    build_chat_completion_response,
    encode_chat_stream_done,
    encode_chat_stream_sse,
)
from ergon_studio.proxy.chat_bridge import parse_chat_completion_request
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.models import (
    ProxyContentDeltaEvent,
    ProxyFinishEvent,
    ProxyOutputItemRef,
    ProxyReasoningDeltaEvent,
    ProxyToolCallEvent,
    ProxyTurnRequest,
    ProxyTurnResult,
)
from ergon_studio.proxy.responses_adapter import (
    build_responses_response,
    encode_responses_stream_events,
    encode_responses_stream_sse,
)
from ergon_studio.proxy.responses_bridge import parse_responses_request
from ergon_studio.registry import RuntimeRegistry
from ergon_studio.upstream import probe_upstream_models

MAX_REQUEST_BODY_BYTES = 5 * 1024 * 1024


class RequestBodyTooLargeError(ValueError):
    """Raised when a request exceeds the accepted body size."""


class ProxyHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        core: ProxyOrchestrationCore,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.core = core


class ProxyRequestHandler(BaseHTTPRequestHandler):
    server_version = "ergon-studio-proxy/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def proxy_server(self) -> ProxyHTTPServer:
        return cast(ProxyHTTPServer, self.server)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            try:
                data = _available_models(self.proxy_server.core.registry)
            except Exception as exc:
                self._send_error_json(
                    HTTPStatus.BAD_GATEWAY, f"failed to load upstream models: {exc}"
                )
                return
            self._send_json(HTTPStatus.OK, {"object": "list", "data": data})
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {self.path}")

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        if self.path == "/v1/responses":
            self._handle_responses()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"unknown route: {self.path}")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_chat_completions(self) -> None:
        try:
            payload = self._read_json_body()
            request = parse_chat_completion_request(payload)
        except RequestBodyTooLargeError as exc:
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            return
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
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            except Exception as exc:
                self._stream_chat_failure(
                    completion_id=completion_id,
                    model=request.model,
                    created_at=created_at,
                    error_text=f"{type(exc).__name__}: {exc}",
                )
            return

        try:
            result = asyncio.run(self._run_chat_completion(request))
        except Exception as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
            )
            return
        if result.finish_reason == "error":
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, result.content or "proxy turn failed"
            )
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
                tool_calls=result.tool_calls,
            ),
        )

    async def _run_chat_completion(self, request: ProxyTurnRequest) -> ProxyTurnResult:
        stream = self.proxy_server.core.stream_turn(
            request,
            created_at=int(time.time()),
        )
        async for _event in stream:
            pass
        return await stream.get_final_response()

    async def _stream_chat_completion(
        self,
        *,
        request: ProxyTurnRequest,
        completion_id: str,
        created_at: int,
    ) -> None:
        stream = self.proxy_server.core.stream_turn(request, created_at=created_at)
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

    def _handle_responses(self) -> None:
        try:
            payload = self._read_json_body()
            request = parse_responses_request(payload)
        except RequestBodyTooLargeError as exc:
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            return
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        response_id = f"resp_{uuid4().hex}"
        created_at = int(time.time())
        if request.stream:
            self._send_sse_headers()
            try:
                asyncio.run(
                    self._stream_responses(
                        request=request,
                        response_id=response_id,
                        created_at=created_at,
                    )
                )
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            except Exception as exc:
                self._stream_responses_failure(
                    response_id=response_id,
                    model=request.model,
                    created_at=created_at,
                    error_text=f"{type(exc).__name__}: {exc}",
                )
            return

        try:
            result = asyncio.run(self._run_chat_completion(request))
        except Exception as exc:
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
            )
            return
        if result.finish_reason == "error":
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, result.content or "proxy turn failed"
            )
            return
        self._send_json(
            HTTPStatus.OK,
            build_responses_response(
                response_id=response_id,
                model=request.model,
                created_at=created_at,
                content=result.content,
                reasoning=result.reasoning,
                tool_calls=result.tool_calls,
                output_items=result.output_items,
            ),
        )

    async def _stream_responses(
        self,
        *,
        request: ProxyTurnRequest,
        response_id: str,
        created_at: int,
    ) -> None:
        reasoning_item_id = f"rs_{uuid4().hex}"
        message_item_id = f"msg_{uuid4().hex}"
        content_emitted = False
        reasoning_text = ""
        message_text = ""
        output_items: list[ProxyOutputItemRef] = []
        tool_item_ids: dict[str, str] = {}
        created_payload = {
            "type": "response.created",
            "event_id": f"event_{uuid4().hex}",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": request.model,
                "status": "in_progress",
            },
            "sequence_number": 1,
        }
        self.wfile.write(encode_responses_stream_sse(created_payload))
        self.wfile.flush()
        sequence_number = 2
        stream = self.proxy_server.core.stream_turn(request, created_at=created_at)
        try:
            async for event in stream:
                if isinstance(event, ProxyContentDeltaEvent) and event.delta:
                    content_emitted = True
                    message_text += event.delta
                if isinstance(event, ProxyReasoningDeltaEvent) and event.delta:
                    reasoning_text += event.delta
                reasoning_output_index = (
                    _response_output_index(
                        output_items, ProxyOutputItemRef(kind="reasoning")
                    )
                    or 0
                )
                if isinstance(event, ProxyReasoningDeltaEvent):
                    reasoning_output_index = _ensure_response_output_item(
                        output_items, ProxyOutputItemRef(kind="reasoning")
                    )
                message_output_index = (
                    _response_output_index(
                        output_items, ProxyOutputItemRef(kind="content")
                    )
                    or 0
                )
                if isinstance(event, ProxyContentDeltaEvent) or (
                    isinstance(event, ProxyFinishEvent) and content_emitted
                ):
                    message_output_index = _ensure_response_output_item(
                        output_items, ProxyOutputItemRef(kind="content")
                    )
                tool_output_index = 0
                if isinstance(event, ProxyToolCallEvent):
                    tool_output_index = _ensure_response_output_item(
                        output_items,
                        ProxyOutputItemRef(kind="tool_call", call_id=event.call.id),
                    )
                    tool_item_ids.setdefault(event.call.id, f"fc_{uuid4().hex}")
                for payload in encode_responses_stream_events(
                    event=event,
                    response_id=response_id,
                    model=request.model,
                    created_at=created_at,
                    sequence_number=sequence_number,
                    reasoning_item_id=reasoning_item_id,
                    message_item_id=message_item_id,
                    reasoning_output_index=reasoning_output_index,
                    message_output_index=message_output_index,
                    tool_output_index=tool_output_index,
                    tool_item_id=tool_item_ids.get(event.call.id)
                    if isinstance(event, ProxyToolCallEvent)
                    else None,
                    reasoning_text=reasoning_text,
                    message_text=message_text,
                    include_output_done=content_emitted,
                ):
                    self.wfile.write(encode_responses_stream_sse(payload))
                    self.wfile.flush()
                    sequence_number += 1
        except Exception as exc:
            self._stream_responses_failure(
                response_id=response_id,
                model=request.model,
                created_at=created_at,
                error_text=f"{type(exc).__name__}: {exc}",
                sequence_number=sequence_number,
            )

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("missing Content-Length header")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        if length > MAX_REQUEST_BODY_BYTES:
            raise RequestBodyTooLargeError(
                f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"
            )
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
                    "type": "invalid_request_error"
                    if status < HTTPStatus.INTERNAL_SERVER_ERROR
                    else "server_error",
                }
            },
        )

    def _stream_chat_failure(
        self,
        *,
        completion_id: str,
        model: str,
        created_at: int,
        error_text: str,
    ) -> None:
        try:
            self.wfile.write(
                encode_chat_stream_sse(
                    ProxyContentDeltaEvent(error_text),
                    completion_id=completion_id,
                    model=model,
                    created_at=created_at,
                )
            )
            self.wfile.write(
                encode_chat_stream_sse(
                    ProxyFinishEvent("error"),
                    completion_id=completion_id,
                    model=model,
                    created_at=created_at,
                )
            )
            self.wfile.write(encode_chat_stream_done())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _stream_responses_failure(
        self,
        *,
        response_id: str,
        model: str,
        created_at: int,
        error_text: str,
        sequence_number: int = 1,
    ) -> None:
        payload = {
            "type": "response.failed",
            "event_id": f"event_{uuid4().hex}",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "failed",
                "error": {
                    "message": error_text,
                    "type": "server_error",
                },
            },
            "sequence_number": sequence_number,
        }
        try:
            self.wfile.write(encode_responses_stream_sse(payload))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def serve_proxy(*, host: str, port: int, core: ProxyOrchestrationCore) -> None:
    server = ProxyHTTPServer((host, port), ProxyRequestHandler, core=core)
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


def start_proxy_server_in_thread(
    *,
    host: str,
    port: int,
    core: ProxyOrchestrationCore,
) -> ProxyServerHandle:
    server = ProxyHTTPServer((host, port), ProxyRequestHandler, core=core)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return ProxyServerHandle(server, thread)


def _available_models(registry: RuntimeRegistry) -> list[dict[str, Any]]:
    now = int(time.time())
    models = probe_upstream_models(registry.upstream, timeout=5)
    normalized: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue
        normalized.append(
            {
                "id": model_id,
                "object": "model",
                "created": int(model.get("created", now))
                if isinstance(model.get("created"), int)
                else now,
                "owned_by": str(model.get("owned_by", "upstream")),
            }
        )
    return normalized


def _ensure_response_output_item(
    output_items: list[ProxyOutputItemRef], item: ProxyOutputItemRef
) -> int:
    existing = _response_output_index(output_items, item)
    if existing is not None:
        return existing
    output_items.append(item)
    return len(output_items) - 1


def _response_output_index(
    output_items: list[ProxyOutputItemRef], item: ProxyOutputItemRef
) -> int | None:
    try:
        return output_items.index(item)
    except ValueError:
        return None
