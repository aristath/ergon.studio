from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.app_config import ProxyAppConfig
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.server import ProxyServerHandle, start_proxy_server_in_thread
from ergon_studio.registry import RuntimeRegistry, load_registry
from ergon_studio.upstream import UpstreamSettings, validate_upstream_base_url


@dataclass(frozen=True)
class ProxyServerStatus:
    running: bool
    message: str
    url: str | None = None


class ProxyServerController:
    def __init__(self) -> None:
        self._handle: ProxyServerHandle | None = None
        self._host: str = "127.0.0.1"
        self._registry: RuntimeRegistry | None = None

    @property
    def status(self) -> ProxyServerStatus:
        if self._handle is None:
            return ProxyServerStatus(running=False, message="server stopped")
        return ProxyServerStatus(
            running=True,
            message="server running",
            url=f"http://{_display_host(self._host)}:{self._handle.port}/v1",
        )

    def start(
        self,
        *,
        config: ProxyAppConfig,
        definitions_dir: Path,
    ) -> ProxyServerStatus:
        upstream_base_url = validate_upstream_base_url(config.upstream_base_url)
        registry = load_registry(
            definitions_dir,
            upstream=UpstreamSettings(
                base_url=upstream_base_url,
                api_key=config.upstream_api_key.strip() or None,
                instruction_role=config.instruction_role.strip() or None,
                tool_calling=not config.disable_tool_calling,
            ),
        )
        core = ProxyOrchestrationCore(registry)
        old_handle = self._handle
        old_host = self._host
        old_registry = self._registry
        replacing_same_bind = (
            old_handle is not None
            and old_host == config.host
            and old_handle.port == config.port
        )

        if replacing_same_bind:
            self.stop()
            try:
                self._handle = start_proxy_server_in_thread(
                    host=config.host,
                    port=config.port,
                    core=core,
                )
            except Exception:
                if old_registry is not None and old_handle is not None:
                    self._handle = start_proxy_server_in_thread(
                        host=old_host,
                        port=old_handle.port,
                        core=ProxyOrchestrationCore(old_registry),
                    )
                    self._host = old_host
                    self._registry = old_registry
                raise
        else:
            new_handle = start_proxy_server_in_thread(
                host=config.host,
                port=config.port,
                core=core,
            )
            if old_handle is not None:
                old_handle.close()
            self._handle = new_handle

        self._host = config.host
        self._registry = registry
        return self.status

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self._registry = None


def _display_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host
