from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path

from ergon_studio.context_providers import AgentProfileContextProvider, ArtifactContextProvider, ConversationHistoryProvider, ProjectMemoryContextProvider, RetrievalContextProvider, TaskWhiteboardContextProvider
from ergon_studio.runtime import RuntimeContext
from ergon_studio.workflow_compiler import compile_workflow_definition


@dataclass(frozen=True)
class EvalResult:
    name: str
    status: str
    details: str


def run_builtin_evals(runtime: RuntimeContext) -> list[EvalResult]:
    return [
        _definitions_eval(runtime),
        _workflow_compilation_eval(runtime),
        _provider_registry_eval(runtime),
        _context_provider_wiring_eval(runtime),
    ]


def write_eval_report(runtime: RuntimeContext, results: list[EvalResult], *, created_at: int | None = None) -> Path:
    if created_at is None:
        created_at = int(time.time())
    report_dir = runtime.paths.exports_dir / "evals"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval-{created_at}.md"
    report_path.write_text(_render_report(results, created_at=created_at), encoding="utf-8")

    json_path = report_dir / f"eval-{created_at}.json"
    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def summarize_results(results: list[EvalResult]) -> str:
    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    return f"passed={passed} failed={failed} skipped={skipped}"


def _definitions_eval(runtime: RuntimeContext) -> EvalResult:
    required_agents = {"orchestrator", "architect", "coder", "reviewer", "fixer", "researcher", "tester", "documenter", "brainstormer", "designer"}
    missing = sorted(required_agents - set(runtime.list_agent_ids()))
    if missing:
        return EvalResult(
            name="definitions",
            status="failed",
            details=f"missing agent definitions: {', '.join(missing)}",
        )
    return EvalResult(
        name="definitions",
        status="passed",
        details=f"{len(runtime.list_agent_ids())} agent definitions and {len(runtime.list_workflow_ids())} workflow definitions loaded",
    )


def _workflow_compilation_eval(runtime: RuntimeContext) -> EvalResult:
    compiled: list[str] = []
    for definition in runtime.registry.workflow_definitions.values():
        compile_workflow_definition(definition)
        compiled.append(definition.id)
    return EvalResult(
        name="workflow_compilation",
        status="passed",
        details=f"compiled workflows: {', '.join(sorted(compiled))}",
    )


def _provider_registry_eval(runtime: RuntimeContext) -> EvalResult:
    providers = runtime.list_provider_ids()
    if not providers:
        return EvalResult(
            name="provider_registry",
            status="skipped",
            details="no providers configured",
        )

    invalid = [
        provider_name
        for provider_name in providers
        if not isinstance(runtime.provider_details(provider_name), dict)
    ]
    if invalid:
        return EvalResult(
            name="provider_registry",
            status="failed",
            details=f"invalid providers: {', '.join(invalid)}",
        )
    return EvalResult(
        name="provider_registry",
        status="passed",
        details=f"configured providers: {', '.join(providers)}",
    )


def _context_provider_wiring_eval(runtime: RuntimeContext) -> EvalResult:
    if not runtime.can_build_agent("orchestrator"):
        return EvalResult(
            name="context_provider_wiring",
            status="skipped",
            details="orchestrator provider not configured",
        )

    agent = runtime.build_agent("orchestrator")
    provider_types = {type(provider) for provider in agent.context_providers}
    expected = {
        AgentProfileContextProvider,
        ConversationHistoryProvider,
        TaskWhiteboardContextProvider,
        ProjectMemoryContextProvider,
        ArtifactContextProvider,
        RetrievalContextProvider,
    }
    missing = [provider_type.__name__ for provider_type in expected if provider_type not in provider_types]
    if missing:
        return EvalResult(
            name="context_provider_wiring",
            status="failed",
            details=f"missing providers: {', '.join(sorted(missing))}",
        )
    return EvalResult(
        name="context_provider_wiring",
        status="passed",
        details="orchestrator includes the full framework-native context provider stack",
    )


def _render_report(results: list[EvalResult], *, created_at: int) -> str:
    lines = [
        "# ergon.studio Eval Report",
        "",
        f"- created_at: {created_at}",
        f"- summary: {summarize_results(results)}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.name}",
                "",
                f"- status: {result.status}",
                f"- details: {result.details}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
