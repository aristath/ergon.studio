from __future__ import annotations

from typing import Any

from ergon_studio.agent_factory import build_agent
from ergon_studio.provider_health import assess_agent_readiness, probe_all_providers


def assigned_provider_name(registry, agent_id: str) -> str | None:
    definition = registry.agent_definitions.get(agent_id)
    if definition is None:
        return None
    role = str(definition.metadata.get("role", definition.id))
    role_assignments = registry.config.get("role_assignments", {})
    provider_name = role_assignments.get(role) or role_assignments.get(agent_id)
    if not provider_name:
        return None
    if provider_name not in registry.config.get("providers", {}):
        return None
    return provider_name


def provider_details(registry, provider_name: str) -> dict[str, Any] | None:
    providers = registry.config.get("providers", {})
    if not isinstance(providers, dict):
        return None
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return None
    return provider


def agent_unavailable_reason(registry, agent_id: str) -> str | None:
    try:
        build_agent(
            registry,
            agent_id,
            tool_registry={},
            ignore_missing_tools=True,
            include_mcp_servers=False,
        )
    except (KeyError, ValueError) as exc:
        return str(exc)
    return None


def agent_status_summary(registry, agent_id: str) -> str:
    provider_name = assigned_provider_name(registry, agent_id)
    if provider_name is None:
        return "not configured"
    provider = provider_details(registry, provider_name) or {}
    model_name = provider.get("model", "unknown-model")
    return f"ready via {provider_name} ({model_name})"


def build_proxy_health_snapshot(registry, *, timeout: int = 10) -> dict[str, Any]:
    provider_results = probe_all_providers(registry.config, timeout=timeout)
    agent_ids = sorted(registry.agent_definitions)
    agent_results = assess_agent_readiness(
        agent_ids=agent_ids,
        assigned_provider_name=lambda agent_id: assigned_provider_name(registry, agent_id),
        agent_unavailable_reason=lambda agent_id: agent_unavailable_reason(registry, agent_id),
        agent_status_summary=lambda agent_id: agent_status_summary(registry, agent_id),
        provider_health=provider_results,
        provider_details=lambda provider_name: provider_details(registry, provider_name),
    )
    orchestrator = next((result for result in agent_results if result.name == "orchestrator"), None)
    ok = bool(orchestrator and orchestrator.ok)
    return {
        "ok": ok,
        "providers": [
            {
                "name": result.name,
                "ok": result.ok,
                "base_url": result.base_url,
                "model": result.model,
                "model_count": result.model_count,
                "error": result.error,
            }
            for result in provider_results
        ],
        "agents": [
            {
                "name": result.name,
                "ok": result.ok,
                "provider_name": result.provider_name,
                "model": result.model,
                "summary": result.summary,
                "error": result.error,
            }
            for result in agent_results
        ],
    }
