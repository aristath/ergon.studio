from __future__ import annotations

import argparse
from pathlib import Path

from ergon_studio.bootstrap import bootstrap_workspace
from ergon_studio.runtime import load_runtime
from ergon_studio.tui.app import ErgonStudioApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergon-studio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    bootstrap_parser.add_argument(
        "--home-dir",
        type=Path,
        default=Path.home(),
    )

    tui_parser = subparsers.add_parser("tui")
    tui_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    tui_parser.add_argument(
        "--home-dir",
        type=Path,
        default=Path.home(),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        paths = bootstrap_workspace(
            project_root=args.project_root,
            home_dir=args.home_dir,
        )
        print(f"project_uuid={paths.project_uuid}")
        print(f"project_data_dir={paths.project_data_dir}")
        return 0

    if args.command == "tui":
        runtime = load_runtime(
            project_root=args.project_root,
            home_dir=args.home_dir,
        )
        app = ErgonStudioApp(runtime)
        app.run()
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
