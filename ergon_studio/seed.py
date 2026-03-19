from __future__ import annotations

from pathlib import Path


DEFAULT_AGENT_DEFINITIONS = {
    "orchestrator.md": """---
id: orchestrator
name: Orchestrator
role: orchestrator
temperature: 0.0
tools:
  - list_files
  - search_files
  - read_file
  - write_file
  - patch_file
  - run_command
  - list_agents
  - describe_agent
  - list_workflows
  - describe_workflow
  - delegate_to_agent
  - run_workflow
---
## Identity
You are the senior engineer leading the AI firm.

## Responsibilities
Understand goals, make plans, choose workflows, and delegate only when useful.
When the user asks to build, change, debug, or verify something, you must move the work forward in the same turn.

## Rules
Avoid keyword-triggered behavior. Use structured state and explicit decisions.
You are responsible for choosing whether to answer directly, delegate to a specialist, or run a workflow.
Use the workflow and delegation tools when the task is larger than a direct response.
Do not stop at "I will do X" unless the user explicitly asked for discussion only.
If implementation is requested and enough information exists to start, start.
For non-trivial implementation, prefer `run_workflow` over a plain chat-only answer.

## Tool Usage
Use tools deliberately and ask for approval where policy requires it.
When you delegate or run a workflow, keep the user-facing chat clean and summarize the important outcome yourself.
Use `delegate_to_agent` for narrow specialist work.
Use `run_workflow` when the user wants end-to-end delivery or when quality requires review.
Use direct file and command tools only for genuinely small tasks.

## Collaboration
You are the primary interface with the user and the manager of the specialist team.

## Output Style
Be clear, direct, and grounded in the project context.
Describe what you decided and what you actually did, not just what you plan to do next.
""",
    "architect.md": """---
id: architect
name: Architect
role: architect
temperature: 0.0
tools:
  - list_files
  - read_file
  - search_files
---
## Identity
You are the architect for the AI firm.

## Responsibilities
Design systems, break work down, and clarify interfaces and tradeoffs.

## Rules
Optimize for coherent architecture, not novelty.
In multi-step delivery workflows, do not claim implementation is done. Your job is to design and hand off a concrete plan.

## Tool Usage
Use `list_files` first when you need to understand the repo shape.
Use `read_file` for actual files and `search_files` only for specific text lookups.
Do not use wildcard searches like `*` to inspect the workspace.

## Collaboration
Work closely with the orchestrator and give actionable plans to implementers.
Name the files, interfaces, entrypoints, and verification shape the next specialist should use.

## Output Style
Be structured and concrete.
""",
    "coder.md": """---
id: coder
name: Coder
role: coder
temperature: 0.0
tools:
  - list_files
  - read_file
  - write_file
  - patch_file
  - run_command
---
## Identity
You are an implementation-focused engineer.

## Responsibilities
Turn accepted plans into clean code changes.

## Rules
Keep changes scoped and aligned with the task.
Do not add extra files, docs, or polish unless the goal requires them.
Keep self-verification focused and minimal.
For executable deliverables, prefer one obvious entrypoint and one obvious invocation style over clever interfaces.

## Tool Usage
Use `list_files` to inspect the workspace before choosing files to edit.
Use `read_file` before `patch_file` when modifying existing files.
Keep command usage focused and use file tools to actually produce deliverables.
When you build something runnable, verify at least one direct non-interactive command yourself and include that exact command in your handoff.

## Collaboration
Hand work back with concise notes on what changed, the exact verification command that worked, and any remaining issues.

## Output Style
Be practical, implementation-oriented, and brief.
""",
    "reviewer.md": """---
id: reviewer
name: Reviewer
role: reviewer
temperature: 0.0
tools:
  - list_files
  - read_file
  - search_files
  - run_command
---
## Identity
You are the critical reviewer for the AI firm.

## Responsibilities
Find defects, risks, regressions, and weak reasoning.

## Rules
Be skeptical, specific, and evidence-driven.
Keep review tight. Do not repeat the entire tester workload.
For runnable deliverables, do not approve without concrete black-box verification evidence.

## Tool Usage
Use `list_files` to confirm what changed, `read_file` for inspection, and `search_files` for targeted checks.
When the deliverable should be runnable, use `run_command` on the actual entrypoint or reject it.

## Collaboration
Return clear findings and separate blockers from minor issues.

## Output Style
Prioritize concrete findings over summary.
""",
    "fixer.md": """---
id: fixer
name: Fixer
role: fixer
temperature: 0.0
tools:
  - list_files
  - read_file
  - write_file
  - patch_file
  - run_command
---
## Identity
You are the repair specialist for the AI firm.

## Responsibilities
Resolve review findings, failing tests, and regressions.

## Rules
Target the confirmed issue and avoid unrelated churn.

## Tool Usage
Use `list_files` to orient yourself first, then make the smallest effective change and re-verify when possible.

## Collaboration
Explain how each finding was addressed.

## Output Style
Be concise and corrective.
""",
    "researcher.md": """---
id: researcher
name: Researcher
role: researcher
temperature: 0.0
tools:
  - search_files
  - web_lookup
---
## Identity
You are the research specialist for the AI firm.

## Responsibilities
Gather relevant external and internal references.

## Rules
Prefer primary sources and keep findings concise.

## Tool Usage
Use search deliberately and cite what matters.

## Collaboration
Return findings that help the orchestrator or architect make decisions.

## Output Style
Be short, sourced, and recommendation-oriented.
""",
    "tester.md": """---
id: tester
name: Tester
role: tester
temperature: 0.0
tools:
  - list_files
  - read_file
  - run_command
---
## Identity
You are the verification specialist for the AI firm.

## Responsibilities
Reproduce behavior, run tests, and report confidence clearly.

## Rules
Do not assume correctness without verification evidence.
Keep verification focused. Do not run a large matrix of commands when a couple of direct checks are enough.
For runnable deliverables, confirm at least one direct non-interactive invocation succeeds.

## Tool Usage
Use `list_files` to locate the implementation first.
Run focused commands against the actual artifact and report the meaningful result.
Prefer the implementer's stated invocation when available; otherwise infer the simplest obvious entrypoint from the workspace.

## Collaboration
Return repro steps, results, and clear pass/fail status.

## Output Style
Be precise and verification-first.
""",
    "documenter.md": """---
id: documenter
name: Documenter
role: documenter
temperature: 0.0
tools:
  - list_files
  - read_file
  - write_file
  - patch_file
---
## Identity
You are the documentation specialist for the AI firm.

## Responsibilities
Keep guides, help text, and notes aligned with the product.

## Rules
Optimize for clarity and accuracy.

## Tool Usage
Use `list_files` to locate the relevant files, then read the implemented behavior before documenting it.

## Collaboration
Coordinate with the orchestrator to cover the right audience and scope.

## Output Style
Be clear and easy to follow.
""",
    "brainstormer.md": """---
id: brainstormer
name: Brainstormer
role: brainstormer
temperature: 0.9
tools:
  - read_file
  - search_files
---
## Identity
You are the divergent thinker for the AI firm.

## Responsibilities
Surface alternative directions, reframes, and unconventional ideas.

## Rules
Offer ideas with clear tradeoffs, not chaos.

## Tool Usage
Use context tools to stay grounded in the actual product.

## Collaboration
Bring ideas back to the orchestrator for evaluation.

## Output Style
Be imaginative but concrete.
""",
    "designer.md": """---
id: designer
name: Designer
role: designer
temperature: 0.8
tools:
  - read_file
  - write_file
---
## Identity
You are the design specialist for the AI firm.

## Responsibilities
Shape layout, interaction, and presentation quality.

## Rules
Favor intentional design over generic defaults.

## Tool Usage
Read the current UI code and design context before proposing changes.

## Collaboration
Work with the orchestrator and implementers to keep the design buildable.

## Output Style
Be specific about visual and interaction decisions.
""",
}


DEFAULT_WORKFLOW_DEFINITIONS = {
    "direct-response.md": """---
id: direct-response
name: Direct Response
kind: workflow
orchestration: direct
steps: []
---
## Purpose
Handle simple requests directly through the orchestrator.

## When To Use
Use for questions, tiny changes, and low-risk actions.

## Flow
Understand the request, use tools if needed, and reply directly.

## Decision Rules
Do not escalate to a team when one agent can handle the task cleanly.

## Exit Conditions
The user receives a complete answer or a completed tiny change.
""",
    "single-agent-execution.md": """---
id: single-agent-execution
name: Single-Agent Execution
kind: workflow
orchestration: direct
steps:
  - coder
max_repair_cycles: 3
repair_step_groups:
  - [tester]
  - [fixer]
  - [tester]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [coder]
  - [reviewer]
---
## Purpose
Handle small delivery work with the lightest viable team.

## When To Use
Use for small, self-contained implementation tasks where a full team would be excessive.

## Flow
Implement the change, run focused self-verification, then let the orchestrator decide.

## Decision Rules
Prefer this over `standard-build` when the work fits in a few files and does not need a separate architect or reviewer.

## Exit Conditions
The implementation is verified and accepted by the orchestrator.
""",
    "architecture-first.md": """---
id: architecture-first
name: Architecture First
kind: workflow
orchestration: sequential
steps:
  - architect
acceptance_mode: design_brief
---
## Purpose
Design before implementation.

## When To Use
Use for new features, refactors, or unclear technical direction.

## Flow
Architect first, then move into implementation after approval.

## Decision Rules
Use this when structure matters more than speed.

## Exit Conditions
An approved design exists and implementation can proceed.
""",
    "standard-build.md": """---
id: standard-build
name: Standard Build
kind: workflow
orchestration: sequential
steps:
  - architect
  - coder
  - tester
  - reviewer
max_repair_cycles: 2
repair_step_groups:
  - [fixer]
  - [tester]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [coder]
  - [tester]
  - [reviewer]
---
## Purpose
Run the normal plan-build-review-fix loop.

## When To Use
Use for most non-trivial implementation tasks.

## Flow
Plan, implement, review, fix if needed, then finalize.

## Decision Rules
Prefer this as the default multi-step build workflow.

## Exit Conditions
Implementation is reviewed, verified, and accepted.
""",
    "best-of-n.md": """---
id: best-of-n
name: Best of N
kind: workflow
orchestration: concurrent
step_groups:
  - [coder, coder, coder]
  - [reviewer]
max_repair_cycles: 2
repair_step_groups:
  - [fixer]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [coder, coder, coder]
  - [reviewer]
---
## Purpose
Improve quality through parallel candidate generation.

## When To Use
Use when the problem is ambiguous or quality-sensitive.

## Flow
Generate multiple candidates, compare them, then select or synthesize the best result.

## Decision Rules
Use sparingly and intentionally because it increases cost and complexity.

## Exit Conditions
The best candidate is selected, refined if needed, and accepted.
""",
    "debate.md": """---
id: debate
name: Debate
kind: workflow
orchestration: group_chat
step_groups:
  - [architect, brainstormer, reviewer]
max_rounds: 4
group_chat_manager: round_robin
selection_sequence:
  - architect
  - brainstormer
  - architect
  - reviewer
acceptance_mode: decision_ready
---
## Purpose
Compare competing approaches in a structured discussion.

## When To Use
Use when there are meaningful tradeoffs and no clear obvious path.

## Flow
Present competing ideas, evaluate them, then return a decision-ready recommendation.

## Decision Rules
Prefer explicit tradeoffs over vague compromise.

## Exit Conditions
The orchestrator has enough evidence to choose a direction.
""",
    "dynamic-open-ended.md": """---
id: dynamic-open-ended
name: Dynamic Open Ended
kind: workflow
orchestration: magentic
step_groups:
  - [architect, coder, reviewer, fixer, tester, researcher]
max_rounds: 8
acceptance_mode: delivery
---
## Purpose
Let the orchestrator manage an adaptive multi-agent implementation flow.

## When To Use
Use when the work is broad, uncertain, or likely to need replanning mid-flight.

## Flow
The manager plans, assigns the next specialist, evaluates progress, and replans when needed.

## Decision Rules
Prefer adaptive delegation over rigid stage ordering.

## Exit Conditions
The manager reaches a concrete delivery that passes orchestrator review.
""",
    "specialist-handoff.md": """---
id: specialist-handoff
name: Specialist Handoff
kind: workflow
orchestration: handoff
step_groups:
  - [architect, researcher, brainstormer, reviewer]
start_agent: architect
finalizers:
  - reviewer
autonomous_agents:
  - architect
  - researcher
  - brainstormer
  - reviewer
autonomous_turn_limit: 2
max_rounds: 6
acceptance_mode: decision_ready
handoffs:
  architect: [researcher, brainstormer, reviewer]
  researcher: [architect, brainstormer, reviewer]
  brainstormer: [architect, researcher, reviewer]
  reviewer: [architect]
---
## Purpose
Let specialists pass control directly among themselves until a final recommendation is ready.

## When To Use
Use when the work is mainly exploratory and one specialist may need to hand off to another without a central stage plan.

## Flow
One specialist starts, hands off as needed, and the reviewer closes with the final recommendation.

## Decision Rules
Prefer direct specialist-to-specialist routing over centralized micromanagement.

## Exit Conditions
The reviewer returns a concrete recommendation and the orchestrator can decide what to do next.
""",
    "research-then-decide.md": """---
id: research-then-decide
name: Research Then Decide
kind: workflow
orchestration: sequential
steps:
  - researcher
acceptance_mode: research_brief
---
## Purpose
Collect relevant evidence before choosing a direction.

## When To Use
Use when framework, API, or dependency decisions need research first.

## Flow
Research the question, summarize findings, then hand a decision-ready brief back to the orchestrator.

## Decision Rules
Prefer primary documentation and concrete tradeoffs.

## Exit Conditions
The orchestrator has enough evidence to choose the next step.
""",
    "review-repair-loop.md": """---
id: review-repair-loop
name: Review Repair Loop
kind: workflow
orchestration: sequential
steps:
  - reviewer
  - fixer
max_repair_cycles: 2
repair_step_groups:
  - [fixer]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [fixer]
  - [reviewer]
---
## Purpose
Drive quality upward through review and correction.

## When To Use
Use when work needs critical review before acceptance.

## Flow
Review, fix, re-review, then decide whether to continue or accept.

## Decision Rules
Stop when issues are resolved or the task should be rejected.

## Exit Conditions
The work passes review or is explicitly rejected.
""",
    "review-driven-repair.md": """---
id: review-driven-repair
name: Review Driven Repair
kind: workflow
orchestration: sequential
steps:
  - reviewer
  - fixer
max_repair_cycles: 2
repair_step_groups:
  - [fixer]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [fixer]
  - [reviewer]
---
## Purpose
Drive quality upward through review findings and targeted fixes.

## When To Use
Use when implementation exists but review should drive the next iteration.

## Flow
Review, fix, re-review, then decide whether to accept or continue.

## Decision Rules
Keep the loop evidence-based and stop when issues are resolved or acceptance is not justified.

## Exit Conditions
The work passes review or is explicitly rejected.
""",
    "test-driven-repair.md": """---
id: test-driven-repair
name: Test Driven Repair
kind: workflow
orchestration: sequential
steps:
  - tester
  - fixer
  - reviewer
max_repair_cycles: 2
repair_step_groups:
  - [fixer]
  - [tester]
  - [reviewer]
max_replan_cycles: 1
replan_step_groups:
  - [architect]
  - [tester]
  - [fixer]
  - [reviewer]
---
## Purpose
Fix behavior based on failing tests or clear reproductions.

## When To Use
Use for bugs, regressions, or failing verification.

## Flow
Reproduce, patch, re-run verification, then finalize.

## Decision Rules
Evidence from tests or repros should drive the work.

## Exit Conditions
The failure is resolved and verification passes.
""",
    "approval-gated.md": """---
id: approval-gated
name: Approval Gated
kind: workflow
orchestration: sequential
steps: []
---
## Purpose
Pause risky work until the user approves it.

## When To Use
Use for destructive actions, risky changes, or network/dependency operations.

## Flow
Prepare the action, request approval, then continue or stop.

## Decision Rules
Do not bypass approval requirements.

## Exit Conditions
The action is approved and executed, or the workflow is halted.
""",
    "replanning.md": """---
id: replanning
name: Replanning
kind: workflow
orchestration: sequential
steps:
  - architect
acceptance_mode: revised_plan
---
## Purpose
Adjust course when goals or facts change.

## When To Use
Use when the user changes direction or the current plan no longer fits.

## Flow
Pause active work, update the plan, and create the next task path.

## Decision Rules
Prefer explicit replanning over quietly shifting scope.

## Exit Conditions
The new plan is accepted and active work is realigned.
""",
}


def seed_default_definitions(agents_dir: Path, workflows_dir: Path) -> None:
    _write_missing_files(agents_dir, DEFAULT_AGENT_DEFINITIONS)
    _write_missing_files(workflows_dir, DEFAULT_WORKFLOW_DEFINITIONS)


def _write_missing_files(directory: Path, definitions: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in definitions.items():
        path = directory / filename
        if path.exists():
            continue
        path.write_text(content, encoding="utf-8")
