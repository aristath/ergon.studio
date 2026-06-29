# @ergon.studio/pi-scratchpad

Pi package that injects project-local scratchpad notes into the system prompt
and gives Pi a matching `scratchpad` skill for maintaining those notes.

It is intentionally simple: no service, no database, no network, no setup
daemon. It reads one file from the current project and appends it to the prompt
before the agent starts. The bundled skill explains when and how to write useful
project notes.

## What It Does

This package contains two Pi resources:

- `extensions/`: reads `.ergon.studio/scratchpad.md` and injects it into the
  system prompt
- `skills/`: teaches the agent when to create or update the scratchpad file

On every `before_agent_start` event, the extension:

1. Reads the current working directory from Pi.
2. Looks for `.ergon.studio/scratchpad.md` inside that project.
3. If the file exists, appends it as `## Project Scratchpad`.
4. If the file does not exist, returns nothing and stays silent.

It does not create files automatically. That makes it safe to install globally:
projects without a scratchpad do not pay prompt-token cost and do not receive
onboarding text every turn.

## Install

From this package directory:

```bash
npm install
pi install .
```

From the monorepo root:

```bash
pi install ./pi/packages/scratchpad
```

From npm after publication:

```bash
pi install npm:@ergon.studio/pi-scratchpad
```

## Project Opt-In

Create this file in a project:

```text
.ergon.studio/scratchpad.md
```

Recommended structure:

```markdown
## Conventions

Things the user stated or agreed on during work.

## Notes

Discovered constraints, non-obvious facts, gotchas.

## Decisions

Choices made and reasoning: what was chosen, what was ruled out, and why.
```

The headings are conventions, not parser requirements. The extension injects the
whole file as Markdown.

The bundled skill tells the agent to create this file only when there is
something durable to remember. It should not create empty scratchpads, todo
lists, implementation plans, or summaries of recent work.

## Runtime Behavior

Input:

```markdown
## Conventions

- Use tabs in package-local TypeScript.

## Decisions

- Keep Pi packages under `pi/packages`.
```

Injected prompt block:

```markdown
## Project Scratchpad

## Conventions

- Use tabs in package-local TypeScript.

## Decisions

- Keep Pi packages under `pi/packages`.
```

If the file is missing, the hook returns `undefined`; Pi keeps the original
system prompt unchanged.

## Compaction Behavior

The scratchpad is not written into Pi's compaction summaries. Pi's extension
docs describe `session_before_compact` as the hook for canceling compaction or
providing a custom summary, while `before_agent_start` is the hook for modifying
the system prompt before each agent turn. This package uses
`before_agent_start`, reads the file fresh each turn, and leaves compaction
summaries focused on conversation history.

That means:

- edits to `.ergon.studio/scratchpad.md` take effect on the next agent turn
- compaction does not lose scratchpad context because the file remains on disk
- scratchpad content is not duplicated into summary text

## Why This Is Separate From Memory Steward

Scratchpad and memory steward solve different problems.

Scratchpad:

- project-local
- explicit user/project-authored file
- deterministic
- no services
- no semantic search

Memory steward:

- cross-session
- model-mediated save decisions
- SQLite + vector search
- background services
- semantic recall

Install both when you want both deterministic project context and cross-session
preference memory.

## Files In This Package

```text
extensions/index.ts
  Pi lifecycle hook and scratchpad file reader.

skills/scratchpad/SKILL.md
  Pi skill that tells the agent how to maintain the scratchpad file.

package.json
  Pi package metadata for both extension and skill resources.

tsconfig.json
  TypeScript project config for validation/builds.
```

## Development

Build all Pi packages from `pi/`:

```bash
npm run build
```

Run this package's tests:

```bash
npm test
```

Package dry-run:

```bash
npm pack --dry-run
```

Install locally:

```bash
pi install ./pi/packages/scratchpad
```

## Troubleshooting

Scratchpad not injected:

- confirm the file is exactly `.ergon.studio/scratchpad.md`
- confirm Pi is running with the expected project working directory
- confirm the package is installed with `pi list`
- confirm the file is readable by the current user

Unexpected stale content:

- the file is read fresh on each `before_agent_start`
- edit `.ergon.studio/scratchpad.md` and start a new agent turn

Too much prompt context:

- shorten the scratchpad file
- split long historical notes into a separate document
- keep only conventions, gotchas, and active architectural decisions
