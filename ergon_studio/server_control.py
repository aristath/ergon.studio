from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.app_config import ProxyAppConfig
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.server import ProxyServerHandle, start_proxy_server_in_thread
from ergon_studio.registry import load_registry
from ergon_studio.upstream import UpstreamSettings


@dataclass(frozen=True)
class ProxyServerStatus:
    running: bool
    message: str
    url: str | None = None


class ProxyServerController:
    def __init__(self) -> None:
        self._handle: ProxyServerHandle | None = None

    @property
    def status(self) -> ProxyServerStatus:
        if self._handle is None:
            return ProxyServerStatus(running=False, message="server stopped")
        return ProxyServerStatus(
            running=True,
            message="server running",
            url=f"http://127.0.0.1:{self._handle.port}/v1",
        )

    def start(
        self,
        *,
        config: ProxyAppConfig,
        definitions_dir: Path,
    ) -> ProxyServerStatus:
        self.stop()
        if not config.upstream_base_url.strip():
            return ProxyServerStatus(
                running=False,
                message="set the upstream URL to start the proxy",
            )
        registry = load_registry(
            definitions_dir,
            upstream=UpstreamSettings(
                base_url=config.upstream_base_url.strip(),
                api_key=config.upstream_api_key.strip() or None,
                instruction_role=config.instruction_role.strip() or None,
                tool_calling=not config.disable_tool_calling,
            ),
        )
        self._handle = start_proxy_server_in_thread(
            host=config.host,
            port=config.port,
            core=ProxyOrchestrationCore(registry),
        )
        return self.status

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
