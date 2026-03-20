from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any


@dataclass(frozen=True)
class ProviderHealthResult:
    name: str
    ok: bool
    base_url: str
    model: str
    model_count: int = 0
    error: str | None = None


def probe_endpoint_models(base_url: str, api_key: str | None, *, timeout: int = 10) -> list[dict[str, Any]]:
    """Probe an OpenAI-compatible endpoint for available models."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models: list[dict[str, Any]] = []
    for model in data.get("data", []):
        if not isinstance(model, dict) or "id" not in model:
            continue
        entry: dict[str, Any] = {"id": model["id"]}
        for key in ("context_length", "max_model_len", "context_window"):
            if isinstance(model.get(key), int):
                entry["context_length"] = model[key]
                break
        models.append(entry)
    return sorted(models, key=lambda item: str(item["id"]))


def probe_provider(name: str, provider_config: dict[str, Any], *, timeout: int = 10) -> ProviderHealthResult:
    provider_type = str(provider_config.get("type", "openai_chat"))
    base_url = str(provider_config.get("base_url", "")).strip()
    model = str(provider_config.get("model", "")).strip()
    api_key = str(provider_config.get("api_key", "")).strip() or None

    if provider_type != "openai_chat":
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error=f"Unsupported provider type: {provider_type}",
        )
    if not base_url:
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error="Missing base_url",
        )
    if not model:
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error="Missing model",
        )

    try:
        models = probe_endpoint_models(base_url, api_key, timeout=timeout)
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}"
        try:
            payload = exc.read().decode("utf-8").strip()
        except Exception:
            payload = ""
        if payload:
            error = f"{error}: {payload}"
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error=error,
        )
    except urllib.error.URLError as exc:
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error=str(exc.reason),
        )
    except Exception as exc:
        return ProviderHealthResult(
            name=name,
            ok=False,
            base_url=base_url,
            model=model,
            error=str(exc),
        )

    model_ids = {str(entry.get("id", "")) for entry in models}
    configured_model_found = model in model_ids
    return ProviderHealthResult(
        name=name,
        ok=configured_model_found or not model_ids,
        base_url=base_url,
        model=model,
        model_count=len(models),
        error=None if configured_model_found or not model_ids else f"Configured model '{model}' was not returned by /models",
    )


def probe_all_providers(config: dict[str, Any], *, timeout: int = 10) -> list[ProviderHealthResult]:
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        return []
    return [
        probe_provider(name, provider_config, timeout=timeout)
        for name, provider_config in sorted(providers.items())
        if isinstance(provider_config, dict)
    ]
