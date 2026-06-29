# @ergon.studio/pi-plan

Self-contained Pi extension that adds a Scout-inspired `/plan` mode.

The package is modeled after `pi-brainstorm`: it registers a slash command,
stores mode state in the session, adjusts active tools while the mode is
running, injects mode-specific instructions into the system prompt, blocks
implementation tools, and restores the previous tool set when the mode ends.

## What It Does

`/plan` starts a planning session with read-only project investigation plus the
legacy Scout scratchpad exception. The agent can inspect the project and discuss
architecture, but it cannot edit implementation files or run shell commands.

The legacy Ergon Scout prompt is stored verbatim in `prompts/scout.md`. The
extension strips only the old OpenCode frontmatter before injecting it into Pi,
then appends the minimal Pi-specific boundary required for `/plan` mode and
tool restrictions.

The workflow remains the original Scout process:

1. Optimal Solution
2. Strip It Down
3. Compare to Current
4. High-Level Plan
5. Iterative Zoom-In
6. Friction Points
7. Plan
8. Assume You're Wrong

The final artifact is `.ergon.studio/HANDOFF.md`, reviewed in the editor before
it is written.

## Commands

| Command         | Behavior                                                                       |
| --------------- | ------------------------------------------------------------------------------ |
| `/plan <topic>` | Start planning mode with an optional topic                                     |
| `/plan`         | While inactive, start planning mode; while active, open the finish/cancel menu |
| `/plan finish`  | Review and save `.ergon.studio/HANDOFF.md`                                     |
| `/plan cancel`  | Exit planning mode without writing a handoff                                   |

## Tool Policy

While active, `/plan` allows investigation tools that are available in the
current Pi environment:

- `read`
- `find`
- `grep`
- `ls`
- `ask_user_question`
- `subagent` with `subagent_type: "Explore"` only
- `get_subagent_result`
- `write` and `edit`, only for `.ergon.studio/scratchpad.md`

Everything else is blocked until planning mode ends. This keeps the mode honest:
it can produce an implementation-ready plan and preserve Scout's project notes,
but it cannot quietly become the implementation.

## Handoff

`/plan finish` opens a handoff draft in the editor, then writes it to:

```text
.ergon.studio/HANDOFF.md
```

The draft is seeded from the latest assistant planning response after `/plan`
started. If no assistant plan exists yet, it falls back to a structured
checklist. If `.ergon.studio/HANDOFF.md` already exists, it is included in plan
context and `/plan finish` asks before replacing it.

## Development

```bash
cd pi
npm test
npm run build

cd packages/plan-mode
npm pack --dry-run
```
