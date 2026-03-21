from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class UpstreamSettings:
    base_url: str
    api_key: str | None = None
    instruction_role: str | None = None
    tool_calling: bool = True


def validate_upstream_base_url(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("missing upstream base URL")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("upstream base URL must be a valid http(s) URL")
    return stripped


def probe_upstream_models(
    settings: UpstreamSettings, *, timeout: int = 10
) -> list[dict[str, Any]]:
    url = settings.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url)
    if settings.api_key:
        request.add_header("Authorization", f"Bearer {settings.api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models: list[dict[str, Any]] = []
    for entry in payload.get("data", []):
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue
        models.append(entry)
    return models
