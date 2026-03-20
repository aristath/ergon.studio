from __future__ import annotations

import argparse
from pathlib import Path
import time

from ergon_studio.bootstrap import bootstrap_proxy_home, bootstrap_workspace
from ergon_studio.proxy.core import ProxyOrchestrationCore
from ergon_studio.proxy.health import build_proxy_health_snapshot
from ergon_studio.proxy.server import serve_proxy
from ergon_studio.registry import load_registry_from_global_paths


def _load_runtime_loader():
    from ergon_studio.runtime import load_runtime

    return load_runtime


def _load_session_store_class():
    from ergon_studio.session_store import SessionStore

    return SessionStore


def _load_tui_app_class():
    from ergon_studio.tui.app import ErgonStudioApp

    return ErgonStudioApp


def _load_eval_functions():
    from ergon_studio.evals import run_builtin_evals, summarize_results, write_eval_report

    return run_builtin_evals, summarize_results, write_eval_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ergon-studio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap")
    _add_project_args(bootstrap_parser)

    tui_parser = subparsers.add_parser("tui")
    _add_project_args(tui_parser)
    _add_session_args(tui_parser)
    _add_pick_session_arg(tui_parser)

    eval_parser = subparsers.add_parser("eval")
    _add_project_args(eval_parser)
    _add_session_args(eval_parser)

    serve_parser = subparsers.add_parser("serve")
    _add_project_args(serve_parser)
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=4000)
    serve_parser.add_argument("--model-id", type=str, default="ergon")
    serve_parser.add_argument("--check", action="store_true")

    sessions_parser = subparsers.add_parser("sessions")
    session_subparsers = sessions_parser.add_subparsers(dest="sessions_command", required=True)

    sessions_list_parser = session_subparsers.add_parser("list")
    _add_project_args(sessions_list_parser)
    sessions_list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include archived sessions",
    )

    sessions_new_parser = session_subparsers.add_parser("new")
    _add_project_args(sessions_new_parser)
    sessions_new_parser.add_argument(
        "--title",
        type=str,
        default=None,
    )

    sessions_rename_parser = session_subparsers.add_parser("rename")
    _add_project_args(sessions_rename_parser)
    sessions_rename_parser.add_argument("session_id", type=str)
    sessions_rename_parser.add_argument("--title", type=str, required=True)

    sessions_archive_parser = session_subparsers.add_parser("archive")
    _add_project_args(sessions_archive_parser)
    sessions_archive_parser.add_argument("session_id", type=str)
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
        SessionStore = _load_session_store_class()
        load_runtime = _load_runtime_loader()
        ErgonStudioApp = _load_tui_app_class()

        open_session_picker_on_mount = False
        if args.pick_session and args.session_id is None and not args.new_session:
            paths = bootstrap_workspace(
                project_root=args.project_root,
                home_dir=args.home_dir,
            )
            store = SessionStore(paths)
            open_session_picker_on_mount = len(store.list_sessions()) > 1
        runtime = load_runtime(
            project_root=args.project_root,
            home_dir=args.home_dir,
            session_id=args.session_id,
            create_session=args.new_session,
            session_title=args.title,
        )
        open_config_wizard_on_mount = _should_open_config_wizard_on_mount(
            runtime,
            open_session_picker_on_mount=open_session_picker_on_mount,
        )
        app = ErgonStudioApp(
            runtime,
            open_session_picker_on_mount=open_session_picker_on_mount,
            open_config_wizard_on_mount=open_config_wizard_on_mount,
        )
        app.run()
        return 0

    if args.command == "eval":
        run_builtin_evals, summarize_results, write_eval_report = _load_eval_functions()
        load_runtime = _load_runtime_loader()

        runtime = load_runtime(
            project_root=args.project_root,
            home_dir=args.home_dir,
            session_id=args.session_id,
            create_session=args.new_session,
            session_title=args.title,
        )
        results = run_builtin_evals(runtime)
        report_path = write_eval_report(runtime, results)
        print(summarize_results(results))
        print(f"session_id={runtime.main_session_id}")
        print(f"report={report_path}")
        for result in results:
            print(f"{result.name}:{result.status}:{result.details}")
        return 0

    if args.command == "serve":
        proxy_paths = bootstrap_proxy_home(args.home_dir)
        registry = load_registry_from_global_paths(proxy_paths)
        if args.check:
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
        core = ProxyOrchestrationCore(registry)
        serve_proxy(
            host=args.host,
            port=args.port,
            core=core,
            model_id=args.model_id,
        )
        return 0

    if args.command == "sessions":
        SessionStore = _load_session_store_class()

        paths = bootstrap_workspace(
            project_root=args.project_root,
            home_dir=args.home_dir,
        )
        store = SessionStore(paths)
        if args.sessions_command == "list":
            for session in store.list_sessions(include_archived=args.all):
                archived = " archived" if session.archived_at is not None else ""
                print(f"{session.id}\t{session.title}\t{session.updated_at}{archived}")
            return 0
        if args.sessions_command == "new":
            session = store.create_session(
                title=args.title,
                created_at=int(time.time()),
            )
            print(f"session_id={session.id}")
            print(f"title={session.title}")
            return 0
        if args.sessions_command == "rename":
            session = store.rename_session(
                session_id=args.session_id,
                title=args.title,
                updated_at=int(time.time()),
            )
            print(f"session_id={session.id}")
            print(f"title={session.title}")
            return 0
        if args.sessions_command == "archive":
            session = store.archive_session(
                session_id=args.session_id,
                archived_at=int(time.time()),
            )
            print(f"session_id={session.id}")
            print("archived=true")
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=Path.home(),
    )


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--session",
        dest="session_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create and attach to a new session",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title when creating a new session",
    )


def _add_pick_session_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pick-session",
        action="store_true",
        help="Open the TUI with the session picker when multiple sessions exist",
    )


def _should_open_config_wizard_on_mount(
    runtime,
    *,
    open_session_picker_on_mount: bool,
) -> bool:
    if open_session_picker_on_mount:
        return False
    if runtime.list_main_messages():
        return False
    return runtime.agent_unavailable_reason("orchestrator") is not None
