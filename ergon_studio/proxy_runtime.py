from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.app_config import (
    ProxyAppConfig,
    validate_proxy_host,
    validate_proxy_port,
)
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.upstream import (
    UpstreamSettings,
    ensure_upstream_reachable,
    validate_upstream_base_url,
)


@dataclass(frozen=True)
class PreparedProxyRuntime:
    host: str
    port: int
    registry: RuntimeRegistry
    core: ProxyOrchestrationCore


def prepare_proxy_runtime(
    *,
    definitions_dir: Path,
    config: ProxyAppConfig,
    verify_upstream: bool = True,
) -> PreparedProxyRuntime:
    upstream_base_url = validate_upstream_base_url(config.upstream_base_url)
    host = validate_proxy_host(config.host)
    port = validate_proxy_port(config.port)
    upstream = UpstreamSettings(
        base_url=upstream_base_url,
        api_key=config.upstream_api_key.strip() or None,
        instruction_role=config.instruction_role.strip() or None,
        tool_calling=not config.disable_tool_calling,
    )
    if verify_upstream:
        ensure_upstream_reachable(upstream)
    registry = load_registry(definitions_dir, upstream=upstream)
    return PreparedProxyRuntime(
        host=host,
        port=port,
        registry=registry,
        core=ProxyOrchestrationCore(registry),
    )
