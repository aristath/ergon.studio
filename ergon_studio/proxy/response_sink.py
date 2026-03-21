from __future__ import annotations

from collections.abc import Callable
from typing import Any


def response_holder_sink(scope: dict[str, Any]) -> Callable[[Any], None]:
    return lambda value: _set_response_holder(scope, value)


def _set_response_holder(scope: dict[str, Any], value: Any) -> None:
    scope["response"] = value
