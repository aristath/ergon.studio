# @ergon.studio/pi-orchestrator

Self-contained Pi extension that adds the Ergon legacy orchestrator as an
interactive mode and bundles the legacy specialist agents it depends on.

The quality gate is agent-owned, exactly like the legacy implementation. The
extension does not parse `APPROVED` or `REJECTED`, does not track quality state,
and does not decide whether work is complete.

## What It Provides

- `/orchestrator` mode command
- faithful legacy `orchestrator` prompt injection
- bundled legacy agent prompts:
  - `quality_controller`
  - `reviewer`
  - `design_reviewer`
  - `coder`
  - `architect`
  - `critic`
  - `researcher`
  - `tester`
- legacy-compatible `task` tool for single specialist delegation
- legacy-compatible `run_parallel` tool for independent parallel specialist work

The delegation tools are infrastructure only. The quality workflow remains in
`agents/orchestrator.md` and `agents/quality_controller.md`.

## Delegation Boundary

The `task` and `run_parallel` tools are registered as Pi tools so the
orchestrator prompt and `quality_controller` can invoke bundled specialists, but
they are mode-scoped:

- Outside `/orchestrator`, direct calls to `task` and `run_parallel` are blocked.
- Inside active `/orchestrator`, both tools are available to the orchestrator.
- In orchestrator-spawned child agents, nested `task` calls remain available so
  `quality_controller` can invoke `reviewer` and `design_reviewer`.

This keeps delegation as infrastructure for the legacy workflow without making
specialist spawning part of normal Pi sessions.

## Command

| Command         | Behavior                                                |
| --------------- | ------------------------------------------------------- |
| `/orchestrator` | Starts orchestrator mode, or opens the active mode menu |

Active menu options:

- Continue orchestrating
- Finish orchestrating
- Cancel orchestrating

In a non-UI Pi session, a second `/orchestrator` exits as finished.

## Mode Coordination

Orchestrator mode is mutually exclusive with `/brainstorm` and `/plan`.

- If `/plan` or `/brainstorm` starts after `/orchestrator`, orchestrator mode
  gets out of the way.
- If `/plan` or `/brainstorm` is already active, `/orchestrator` refuses to
  start until that mode is finished or cancelled.

## Quality Gate

The legacy gate is preserved:

1. The orchestrator invokes `quality_controller` after code work.
2. `quality_controller` invokes `reviewer`.
3. If accepted, `quality_controller` invokes `design_reviewer`.
4. If approved, `quality_controller` verifies `.ergon.studio/COMPLETION.md`.
5. It returns `APPROVED` or `REJECTED`.
6. The orchestrator fixes rejected issues and invokes it again.
7. After 3 rejections, the orchestrator asks the user.

Nothing in the extension replaces that judgment.

## Development

```bash
cd pi
npm test
npm run build

cd packages/orchestrator-mode
npm pack --dry-run
```

Optional live smoke test after installing the package locally:

```bash
cd /path/to/ergon.studio
pi install ./pi/packages/orchestrator-mode
pi
```

In Pi:

```text
/orchestrator
Ask the reviewer agent to say exactly: orchestrator smoke ok
```

Expected result: `/orchestrator` starts, the orchestrator can invoke `task`, and
the delegated reviewer response is returned. A direct `task` call outside
`/orchestrator` should be blocked.
