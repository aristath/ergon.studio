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

## Delegation Tools

The `task` and `run_parallel` tools are registered as Pi tools so the
orchestrator prompt and `quality_controller` can invoke bundled specialists, but
the extension does not mode-scope or block them:

- Their parent-session availability follows Pi's active tool selection.
- Entering or leaving `/orchestrator` does not change that selection.
- Bundled specialist subprocesses still receive explicit tool allowlists. For
  example, the coder receives edit and write tools while planning and design
  specialists do not. Reviewer and tester roles retain their existing shell
  access for inspection and verification.
- `quality_controller` receives `task` so it can invoke `reviewer` and
  `design_reviewer`.

This keeps parent tool ownership with Pi while preserving narrow, isolated
specialist capabilities.

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

The agent-owned gate applies to behavior-changing work:

1. The orchestrator briefs `quality_controller` with the request, changes, and verification evidence.
2. `quality_controller` invokes `reviewer`.
3. If accepted, `quality_controller` invokes `design_reviewer`.
4. If approved, `quality_controller` verifies task-specific test, build, documentation, and scope evidence.
5. It returns an exact `Verdict: APPROVED` or `Verdict: REJECTED` footer.
6. The orchestrator fixes rejected issues and invokes it again.
7. After 3 rejections, the orchestrator asks the user.

Documentation-only, comment-only, formatting-only, discussion, and read-only
work skip the gate unless the user requests a review. Nothing in the extension
replaces the controller's judgment.

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
the delegated reviewer response is returned. Because delegation tools follow
Pi's active selection, a direct `task` call can also run outside `/orchestrator`.
