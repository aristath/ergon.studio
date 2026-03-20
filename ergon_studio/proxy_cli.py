from __future__ import annotations

import argparse
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_proxy_home
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.health import build_proxy_health_snapshot
from ergon_studio.proxy.server import serve_proxy
from ergon_studio.registry import load_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergon-studio-proxy")
    parser.add_argument("--home-dir", type=Path, default=Path.home())
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--model-id", type=str, default="ergon")
    parser.add_argument("--check", action="store_true")
    return parser


def run_proxy_server(*, home_dir: Path, host: str, port: int, model_id: str, check: bool) -> int:
    proxy_paths = bootstrap_proxy_home(home_dir)
    registry = load_registry(proxy_paths)
    if check:
        health = build_proxy_health_snapshot(registry)
        print(f"ok={str(health['ok']).lower()}")
        for provider in health["providers"]:
            status = "ok" if provider["ok"] else "error"
            detail = provider["error"] or provider["model"]
            print(f"provider[{provider['name']}]={status}:{detail}")
        for agent in health["agents"]:
            status = "ok" if agent["ok"] else "error"
            detail = agent["error"] or agent["summary"]
            print(f"agent[{agent['name']}]={status}:{detail}")
        if not health["ok"]:
            return 1

    serve_proxy(
        host=host,
        port=port,
        core=ProxyOrchestrationCore(registry),
        model_id=model_id,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_proxy_server(
        home_dir=args.home_dir,
        host=args.host,
        port=args.port,
        model_id=args.model_id,
        check=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
