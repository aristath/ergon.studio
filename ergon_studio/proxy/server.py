from __future__ import annotations

import asyncio
import json
import threading
import time
from http import HTTPStatus
from typing import Any
from uuid import uuid4

from aiohttp import web

from ergon_studio.debug_log import log_event
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
MODEL_CACHE_TTL_SECONDS = 30.0
SESSION_COOKIE_NAME = "ergon_session"
SESSION_HEADER_NAME = "X-Ergon-Session"


class RequestBodyTooLargeError(ValueError):
    """Raised when a request exceeds the accepted body size."""


class _ModelState:
    def __init__(self) -> None:
        self._cache: list[dict[str, Any]] | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def available_models(self, registry: RuntimeRegistry) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if self._cache is not None and now < self._expires_at:
                return [dict(item) for item in self._cache]
            try:
                models = _available_models(registry)
            except Exception:
                if self._cache is not None:
                    return [dict(item) for item in self._cache]
                raise
            self._cache = [dict(item) for item in models]
            self._expires_at = now + MODEL_CACHE_TTL_SECONDS
            return [dict(item) for item in models]


async def _handle_models(request: web.Request) -> web.Response:
    core = request.app[_CORE_KEY]
    model_state = request.app[_MODEL_STATE_KEY]
    try:
        data = model_state.available_models(core.registry)
    except Exception as exc:
        return _error_response(
            HTTPStatus.BAD_GATEWAY, f"failed to load upstream models: {exc}"
        )
    return web.Response(
        status=200,
        content_type="application/json",
        text=json.dumps({"object": "list", "data": data}),
    )


async def _handle_chat_completions(request: web.Request) -> web.StreamResponse:
    core = request.app[_CORE_KEY]
    try:
        payload = await _read_json_body(request)
        turn_request = parse_chat_completion_request(payload)
    except RequestBodyTooLargeError as exc:
        log_event("http_request_rejected", route=request.path, error=str(exc))
        return _error_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
    except ValueError as exc:
        log_event("http_request_rejected", route=request.path, error=str(exc))
        return _error_response(HTTPStatus.BAD_REQUEST, str(exc))

    completion_id = f"chatcmpl_{uuid4().hex}"
    created_at = int(time.time())
    session_id = _parse_session_id(request)
    log_event(
        "http_request",
        route=request.path,
        session_id=session_id,
        stream=turn_request.stream,
        request=turn_request,
    )

    if turn_request.stream:
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "close",
                SESSION_HEADER_NAME: session_id,
            },
        )
        _apply_session_cookie(resp, session_id)
        resp.force_close()
        await resp.prepare(request)
        try:
            turn_stream = core.stream_turn(turn_request, session_id=session_id)
            async for event in turn_stream:
                await resp.write(
                    encode_chat_stream_sse(
                        event,
                        completion_id=completion_id,
                        model=turn_request.model,
                        created_at=created_at,
                    )
                )
            await resp.write(encode_chat_stream_done())
        except (BrokenPipeError, ConnectionResetError, OSError):
            log_event(
                "http_stream_disconnected",
                route=request.path,
                session_id=session_id,
            )
        except Exception as exc:
            log_event(
                "http_stream_failure",
                route=request.path,
                session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            try:
                await resp.write(
                    encode_chat_stream_sse(
                        ProxyContentDeltaEvent(f"{type(exc).__name__}: {exc}"),
                        completion_id=completion_id,
                        model=turn_request.model,
                        created_at=created_at,
                    )
                )
                await resp.write(
                    encode_chat_stream_sse(
                        ProxyFinishEvent("error"),
                        completion_id=completion_id,
                        model=turn_request.model,
                        created_at=created_at,
                    )
                )
                await resp.write(encode_chat_stream_done())
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        return resp

    try:
        turn_stream = core.stream_turn(turn_request, session_id=session_id)
        async for _ in turn_stream:
            pass
        result: ProxyTurnResult = await turn_stream.get_final_response()
    except Exception as exc:
        log_event(
            "http_request_failure",
            route=request.path,
            session_id=session_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
        )
    log_event(
        "http_request_failure" if result.finish_reason == "error" else "http_request_complete",
        route=request.path,
        session_id=session_id,
        result=result,
    )
    json_resp = web.Response(
        status=200,
        content_type="application/json",
        text=json.dumps(
            build_chat_completion_response(
                completion_id=completion_id,
                model=turn_request.model,
                created_at=created_at,
                content=result.content,
                reasoning=result.reasoning,
                finish_reason=result.finish_reason,
                tool_calls=result.tool_calls,
            )
        ),
        headers={SESSION_HEADER_NAME: session_id},
    )
    _apply_session_cookie(json_resp, session_id)
    return json_resp


async def _handle_responses(request: web.Request) -> web.StreamResponse:
    core = request.app[_CORE_KEY]
    try:
        payload = await _read_json_body(request)
        turn_request = parse_responses_request(payload)
    except RequestBodyTooLargeError as exc:
        log_event("http_request_rejected", route=request.path, error=str(exc))
        return _error_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
    except ValueError as exc:
        log_event("http_request_rejected", route=request.path, error=str(exc))
        return _error_response(HTTPStatus.BAD_REQUEST, str(exc))

    response_id = f"resp_{uuid4().hex}"
    created_at = int(time.time())
    session_id = _parse_session_id(request)
    log_event(
        "http_request",
        route=request.path,
        session_id=session_id,
        stream=turn_request.stream,
        request=turn_request,
    )

    if turn_request.stream:
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "Connection": "close",
                SESSION_HEADER_NAME: session_id,
            },
        )
        _apply_session_cookie(resp, session_id)
        resp.force_close()
        await resp.prepare(request)

        reasoning_item_id = f"rs_{uuid4().hex}"
        message_item_id = f"msg_{uuid4().hex}"
        content_emitted = False
        reasoning_text = ""
        message_text = ""
        output_items: list[ProxyOutputItemRef] = []
        tool_item_ids: dict[str, str] = {}
        created_payload: dict[str, Any] = {
            "type": "response.created",
            "event_id": f"event_{uuid4().hex}",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": turn_request.model,
                "status": "in_progress",
            },
            "sequence_number": 1,
        }
        await resp.write(encode_responses_stream_sse(created_payload))
        sequence_number = 2
        turn_stream = core.stream_turn(turn_request, session_id=session_id)
        try:
            async for event in turn_stream:
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
                for sse_payload in encode_responses_stream_events(
                    event=event,
                    response_id=response_id,
                    model=turn_request.model,
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
                    await resp.write(encode_responses_stream_sse(sse_payload))
                    sequence_number += 1
        except Exception as exc:
            failure_payload: dict[str, Any] = {
                "type": "response.failed",
                "event_id": f"event_{uuid4().hex}",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "model": turn_request.model,
                    "status": "failed",
                    "error": {
                        "message": f"{type(exc).__name__}: {exc}",
                        "type": "server_error",
                    },
                },
                "sequence_number": sequence_number,
            }
            try:
                await resp.write(encode_responses_stream_sse(failure_payload))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        return resp

    try:
        turn_stream = core.stream_turn(turn_request, session_id=session_id)
        async for _ in turn_stream:
            pass
        result = await turn_stream.get_final_response()
    except Exception as exc:
        log_event(
            "http_request_failure",
            route=request.path,
            session_id=session_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
        )
    log_event(
        "http_request_failure" if result.finish_reason == "error" else "http_request_complete",
        route=request.path,
        session_id=session_id,
        result=result,
    )
    json_resp = web.Response(
        status=200,
        content_type="application/json",
        text=json.dumps(
            build_responses_response(
                response_id=response_id,
                model=turn_request.model,
                created_at=created_at,
                content=result.content,
                reasoning=result.reasoning,
                tool_calls=result.tool_calls,
                output_items=result.output_items,
            )
        ),
        headers={SESSION_HEADER_NAME: session_id},
    )
    _apply_session_cookie(json_resp, session_id)
    return json_resp


_CORE_KEY: web.AppKey[ProxyOrchestrationCore] = web.AppKey("core")
_MODEL_STATE_KEY: web.AppKey[_ModelState] = web.AppKey("model_state")


def _create_app(core: ProxyOrchestrationCore) -> web.Application:
    app = web.Application()
    app[_CORE_KEY] = core
    app[_MODEL_STATE_KEY] = _ModelState()
    app.router.add_get("/v1/models", _handle_models)
    app.router.add_post("/v1/chat/completions", _handle_chat_completions)
    app.router.add_post("/v1/responses", _handle_responses)
    return app


def serve_proxy(*, host: str, port: int, core: ProxyOrchestrationCore) -> None:
    app = _create_app(core)

    async def _run() -> None:
        runner = web.AppRunner(app)
        log_event("proxy_server_start", host=host, port=port)
        try:
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            event = asyncio.Event()
            await event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            log_event("proxy_server_interrupt", host=host, port=port)
        finally:
            log_event("proxy_server_stop", host=host, port=port)
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


class ProxyServerHandle:
    def __init__(
        self,
        *,
        runner: web.AppRunner,
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
    ) -> None:
        self._runner = runner
        self._loop = loop
        self._thread = thread

    @property
    def port(self) -> int:
        return int(self._runner.addresses[0][1])

    def close(self) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._runner.cleanup(), self._loop
        )
        try:
            future.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def start_proxy_server_in_thread(
    *,
    host: str,
    port: int,
    core: ProxyOrchestrationCore,
) -> ProxyServerHandle:
    app = _create_app(core)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)

    async def _setup() -> None:
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

    loop.run_until_complete(_setup())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    log_event("proxy_server_thread_start", host=host, port=port)
    return ProxyServerHandle(runner=runner, loop=loop, thread=thread)


def _parse_session_id(request: web.Request) -> str:
    header_value = request.headers.get(SESSION_HEADER_NAME, "")
    if header_value.strip():
        return header_value.strip()
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME, "")
    if cookie_value.strip():
        return cookie_value.strip()
    return f"session_{uuid4().hex}"


def _apply_session_cookie(response: web.StreamResponse, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        path="/",
        httponly=True,
        samesite="Lax",
    )


def _error_response(status: HTTPStatus, message: str) -> web.Response:
    return web.Response(
        status=status.value,
        content_type="application/json",
        text=json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error"
                    if status < HTTPStatus.INTERNAL_SERVER_ERROR
                    else "server_error",
                }
            }
        ),
    )


async def _read_json_body(request: web.Request) -> dict[str, Any]:
    content_length_str = request.headers.get("Content-Length")
    if content_length_str is None:
        raise ValueError("missing Content-Length header")
    try:
        length = int(content_length_str)
    except ValueError as exc:
        raise ValueError("invalid Content-Length header") from exc
    if length > MAX_REQUEST_BODY_BYTES:
        raise RequestBodyTooLargeError(
            f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"
        )
    raw = await request.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


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
