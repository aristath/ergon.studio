from __future__ import annotations

import argparse
from pathlib import Path

from ergon_studio.app_config import (
    ProxyAppConfig,
    config_path,
    default_app_dir,
    load_app_config,
    validate_proxy_host,
    validate_proxy_port,
)
from ergon_studio.app_config import (
    definitions_dir as default_definitions_dir,
)
from ergon_studio.proxy.config_tui import run_config_tui
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.server import serve_proxy
from ergon_studio.registry import load_registry
from ergon_studio.upstream import (
    UpstreamSettings,
    ensure_upstream_reachable,
    validate_upstream_base_url,
)
from ergon_studio.workspace import ensure_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergon")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument(
        "--definitions-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--upstream-base-url", type=str, default=None)
    parser.add_argument("--upstream-api-key", type=str, default=None)
    parser.add_argument("--instruction-role", type=str, default=None)
    parser.add_argument("--disable-tool-calling", action="store_true")
    return parser


def run_proxy_server(*, definitions_dir: Path, config: ProxyAppConfig) -> int:
    upstream_base_url = validate_upstream_base_url(config.upstream_base_url)
    host = validate_proxy_host(config.host)
    port = validate_proxy_port(config.port)
    upstream = UpstreamSettings(
        base_url=upstream_base_url,
        api_key=config.upstream_api_key.strip() or None,
        instruction_role=config.instruction_role.strip() or None,
        tool_calling=not config.disable_tool_calling,
    )
    ensure_upstream_reachable(upstream)
    registry = load_registry(
        definitions_dir,
        upstream=upstream,
    )
    serve_proxy(
        host=host,
        port=port,
        core=ProxyOrchestrationCore(registry),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        app_dir = args.app_dir or default_app_dir()
        if args.serve:
            config = _resolve_config(
                load_app_config(config_path(app_dir)),
                host=args.host,
                port=args.port,
                upstream_base_url=args.upstream_base_url,
                upstream_api_key=args.upstream_api_key,
                instruction_role=args.instruction_role,
                disable_tool_calling=args.disable_tool_calling,
            )
            definitions_dir = args.definitions_dir or default_definitions_dir(app_dir)
            return run_proxy_server(definitions_dir=definitions_dir, config=config)

        workspace = ensure_workspace(app_dir)
        config = _resolve_config(
            load_app_config(workspace.config_path),
            host=args.host,
            port=args.port,
            upstream_base_url=args.upstream_base_url,
            upstream_api_key=args.upstream_api_key,
            instruction_role=args.instruction_role,
            disable_tool_calling=args.disable_tool_calling,
        )
        definitions_dir = args.definitions_dir or workspace.definitions_dir
        return run_config_tui(
            app_dir=workspace.app_dir,
            definitions_dir=definitions_dir,
            initial_config=config,
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


def _resolve_config(
    config: ProxyAppConfig,
    *,
    host: str | None,
    port: int | None,
    upstream_base_url: str | None,
    upstream_api_key: str | None,
    instruction_role: str | None,
    disable_tool_calling: bool,
) -> ProxyAppConfig:
    return ProxyAppConfig(
        upstream_base_url=(
            config.upstream_base_url if upstream_base_url is None else upstream_base_url
        ),
        upstream_api_key=(
            config.upstream_api_key if upstream_api_key is None else upstream_api_key
        ),
        host=config.host if host is None else host,
        port=config.port if port is None else port,
        instruction_role=(
            config.instruction_role
            if instruction_role is None
            else instruction_role
        ),
        disable_tool_calling=disable_tool_calling or config.disable_tool_calling,
    )


if __name__ == "__main__":
    raise SystemExit(main())
