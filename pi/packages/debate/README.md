# @ergon.studio/pi-debate

Standalone Pi extension that registers a `debate` tool for intentional
two-participant coding review and convergence.

## What It Provides

- Two independent temporary Pi sessions, one per participant
- Alternating first-pass and review turns in the same working directory
- The parent session's current model, thinking level, and active tools for both participants
- Exact terminal verdicts: `AGREE`, `CONTINUE`, or `BLOCKED`
- A bounded `max_turns` of 2-12 turns, defaulting to 6
- Abort propagation and cleanup of temporary session files
- A complete transcript returned to the parent

The participant roles are labels such as `coder` and `reviewer`, not configured
Pi agents. The package does not depend on orchestrator mode or its bundled
specialist agents. Each participant retains its own conversation history while
receiving the other participant's latest response.

## Tool

```json
{
  "role_a": "coder",
  "role_b": "reviewer",
  "task": "Review and improve the parser implementation",
  "max_turns": 6
}
```

Role A takes the first turn. After that the participants alternate until a
turn after the first ends with exactly `Verdict: AGREE` or
`Verdict: BLOCKED`, or the turn limit is reached. Missing or malformed verdicts
mean `CONTINUE`.

Only the final non-empty response line is parsed as the verdict. Quoted verdicts
inside the response cannot terminate the debate.

The package registers the tool globally and does not change Pi's active tool
selection. Each temporary session inherits that selection, except that the
`debate` tool itself is excluded to prevent recursive debates. Both sessions
work in the parent's current directory, so code and file changes are immediately
visible to the other participant and the parent.

## Install

Local development install:

```bash
cd /path/to/ergon.studio
pi install ./pi/packages/debate
```

After publishing:

```bash
pi install npm:@ergon.studio/pi-debate
```

## Development

```bash
cd pi/packages/debate
npm test
npm run build
npm pack --dry-run
```

Optional loader smoke test:

```bash
pi --no-extensions -e ./pi/packages/debate --list-models zzzz_no_model_match
```
