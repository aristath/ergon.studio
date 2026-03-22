from __future__ import annotations

import json
import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any

_LOGGER = logging.getLogger("ergon_studio.debug")
_LOCK = Lock()
_HANDLER: logging.FileHandler | None = None


def default_debug_log_path() -> Path:
    return Path.home() / ".ergon-studio.log"


def configure_debug_logging(path: Path | None = None) -> Path:
    resolved = (path or default_debug_log_path()).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    with _LOCK:
        global _HANDLER
        if _HANDLER is not None:
            _LOGGER.removeHandler(_HANDLER)
            _HANDLER.close()
        _LOGGER.handlers.clear()
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
        _LOGGER.addHandler(handler)
        _HANDLER = handler

    log_event("logging_enabled", path=str(resolved))
    return resolved


def disable_debug_logging() -> None:
    with _LOCK:
        global _HANDLER
        if _HANDLER is None:
            return
        _LOGGER.removeHandler(_HANDLER)
        _HANDLER.close()
        _HANDLER = None


def debug_logging_enabled() -> bool:
    return _HANDLER is not None


def log_event(event: str, **fields: Any) -> None:
    if _HANDLER is None:
        return
    payload = {"event": event}
    payload.update({key: _to_jsonable(value) for key, value in fields.items()})
    _LOGGER.info("%s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    return str(value)
