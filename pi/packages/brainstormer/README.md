# @ergon.studio/pi-brainstormer

Self-contained Pi extension that adds a freeform `/brainstorm` mode.

`/brainstorm` is deliberately a single command. If no brainstorm session is
active, it starts one. If a brainstorm session is already active, it opens a menu
with the available choices. In a non-UI Pi session, running `/brainstorm` again
ends the brainstorm as done, because there is no menu surface for choosing an
option.

## What It Does

Brainstorm mode is for exploration before planning. The agent becomes a thinking
partner: imaginative, curious, direct about weak assumptions, and patient enough
to let an idea breathe before turning it into a plan.

It does not write implementation code and does not write artifacts. When the
conversation converges on something concrete enough to plan, the agent suggests
switching to `/plan`. The user chooses when to run `/plan`.

If the user starts `/plan` while brainstorm mode is active, brainstorm mode gets
out of the way automatically. Plan mode owns the next prompt and workflow.
If plan mode is already active, `/brainstorm` refuses to start until `/plan` is
finished or cancelled.

The brainstorm prompt is stored in `prompts/brainstorm.md`. The extension injects
that prompt only while brainstorm mode is active, then appends a small Pi-specific
behavioral boundary.

## Command

| Command       | Behavior                                                                 |
| ------------- | ------------------------------------------------------------------------ |
| `/brainstorm` | Starts brainstorm mode, or opens the active brainstorm menu if already on |

Active menu options:

- Continue brainstorming
- Done brainstorming
- Cancel brainstorming

`Done brainstorming` exits the mode and reminds the user that `/plan` is the
natural next step when they want an implementation plan. `Cancel brainstorming`
exits without that nudge.

Non-UI fallback:

- A second `/brainstorm` exits as `Done brainstorming`.
- No extra command or topic argument is required.

## Tool Behavior

`/brainstorm` does not inspect, replace, restore, or block Pi tools. Entering and
leaving the mode preserves the active tool selection established by Pi, the user,
and other extensions.

The injected prompt tells the model to use available tools only for read-only
exploration, avoid implementation work and artifacts, and avoid delegating
implementation work. This is a behavioral instruction rather than an additional
permission layer.

## Development

```bash
cd pi
npm test
npm run build

cd packages/brainstormer
npm pack --dry-run
```
