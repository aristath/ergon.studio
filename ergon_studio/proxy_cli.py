from __future__ import annotations

import argparse
import os
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_definition_home
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.server import serve_proxy
from ergon_studio.registry import load_registry
from ergon_studio.upstream import UpstreamSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergon-studio")
    parser.add_argument("--home-dir", type=Path, default=Path.home())
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--upstream-base-url", type=str, default=os.environ.get("ERGON_UPSTREAM_BASE_URL"))
    parser.add_argument("--upstream-api-key", type=str, default=os.environ.get("ERGON_UPSTREAM_API_KEY"))
    parser.add_argument("--instruction-role", type=str, default=os.environ.get("ERGON_INSTRUCTION_ROLE"))
    parser.add_argument("--disable-tool-calling", action="store_true")
    return parser


def run_proxy_server(
    *,
    home_dir: Path,
    host: str,
    port: int,
    upstream_base_url: str | None,
    upstream_api_key: str | None,
    instruction_role: str | None,
    disable_tool_calling: bool,
) -> int:
    if not upstream_base_url or not upstream_base_url.strip():
        raise ValueError("missing upstream base URL; pass --upstream-base-url or set ERGON_UPSTREAM_BASE_URL")
    proxy_paths = bootstrap_definition_home(home_dir)
    registry = load_registry(
        proxy_paths,
        upstream=UpstreamSettings(
            base_url=upstream_base_url.strip(),
            api_key=upstream_api_key.strip() if upstream_api_key else None,
            instruction_role=instruction_role.strip() if instruction_role else None,
            tool_calling=not disable_tool_calling,
        ),
    )
    serve_proxy(
        host=host,
        port=port,
        core=ProxyOrchestrationCore(registry),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_proxy_server(
        home_dir=args.home_dir,
        host=args.host,
        port=args.port,
        upstream_base_url=args.upstream_base_url,
        upstream_api_key=args.upstream_api_key,
        instruction_role=args.instruction_role,
        disable_tool_calling=args.disable_tool_calling,
    )


if __name__ == "__main__":
    raise SystemExit(main())
