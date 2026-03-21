from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ergon_studio.app_config import config_path, definitions_dir

DEFAULT_AGENT_TEMPLATES: dict[str, str] = {
    "orchestrator.md": """---
id: orchestrator
role: orchestrator
temperature: 0
---

## Identity
You are the orchestration lead for an OpenAI-compatible proxy.

## Responsibilities
Choose the best next action for the host turn.
Delegate when specialist work will improve the result.
Summarize the final result clearly for the host.

## Rules
Use host tools when they are needed.
Respect tool constraints.
Do not expose hidden chain-of-thought.
""",
    "architect.md": """---
id: architect
role: architect
temperature: 0
---

## Identity
You are the architecture specialist.

## Responsibilities
Create short, concrete plans for implementation work.
""",
    "coder.md": """---
id: coder
role: coder
temperature: 0
---

## Identity
You are the coding specialist.

## Responsibilities
Produce concrete implementation work and code-oriented answers.
""",
    "reviewer.md": """---
id: reviewer
role: reviewer
temperature: 0
---

## Identity
You are the review specialist.

## Responsibilities
Check work for correctness and clarity.
""",
}

DEFAULT_WORKFLOW_TEMPLATES: dict[str, str] = {
    "standard-build.md": """---
id: standard-build
name: Standard Build
orchestration: sequential
steps:
  - architect
  - coder
  - reviewer
selection_hints:
  - build
  - implementation
delivery_candidate: true
acceptance_mode: delivery
---

## Purpose
Run a straightforward architecture, implementation, and review flow.
""",
    "research-then-decide.md": """---
id: research-then-decide
name: Research Then Decide
orchestration: sequential
steps:
  - architect
  - reviewer
selection_hints:
  - research
  - investigate
---

## Purpose
Gather information and then make a recommendation.
""",
}


@dataclass(frozen=True)
class WorkspacePaths:
    app_dir: Path
    config_path: Path
    definitions_dir: Path
    agents_dir: Path
    workflows_dir: Path


def ensure_workspace(app_dir: Path) -> WorkspacePaths:
    agents_dir = definitions_dir(app_dir) / "agents"
    workflows_dir = definitions_dir(app_dir) / "workflows"
    agents_dir.mkdir(parents=True, exist_ok=True)
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _seed_templates(agents_dir, DEFAULT_AGENT_TEMPLATES)
    _seed_templates(workflows_dir, DEFAULT_WORKFLOW_TEMPLATES)
    return WorkspacePaths(
        app_dir=app_dir,
        config_path=config_path(app_dir),
        definitions_dir=definitions_dir(app_dir),
        agents_dir=agents_dir,
        workflows_dir=workflows_dir,
    )


def _seed_templates(directory: Path, templates: dict[str, str]) -> None:
    for filename, content in templates.items():
        path = directory / filename
        if path.exists():
            continue
        path.write_text(content, encoding="utf-8")
