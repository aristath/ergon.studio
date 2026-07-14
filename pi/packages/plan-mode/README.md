# @ergon.studio/pi-plan

Self-contained Pi extension that adds a Scout-inspired `/plan` mode.

The package is modeled after `pi-brainstorm`: it registers a slash command,
stores mode state in the session, injects mode-specific instructions into the
system prompt, and produces a reviewed handoff without changing Pi's active
tools.

## What It Does

`/plan` starts a planning session with read-only project investigation plus the
legacy Scout scratchpad exception. The prompt directs the agent to inspect the
project and discuss architecture without editing implementation files or running
implementation commands.

The legacy Ergon Scout prompt is stored verbatim in `prompts/scout.md`. The
extension strips only the old OpenCode frontmatter before injecting it into Pi,
then appends the minimal Pi-specific behavioral boundary required for `/plan`
mode.

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

## Tool Behavior

`/plan` does not inspect, replace, restore, or block Pi tools. Entering and
leaving the mode preserves the active tool selection established by Pi, the user,
and other extensions.

The injected prompt tells the model to use available tools for read-only
investigation and to limit direct writes to `.ergon.studio/scratchpad.md`. The
extension itself writes `.ergon.studio/HANDOFF.md` only after the user reviews the
draft. These are workflow instructions rather than an additional permission
layer.

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
