---
name: scratchpad
description: Read and write .ergon.studio/scratchpad.md when durable project-local conventions, notes, or decisions should survive context compaction and future sessions
---

# Scratchpad

Your private project notes. One file: `.ergon.studio/scratchpad.md`

When this package is installed, the extension injects that file into the system
prompt automatically when it exists. You do not need to read it manually unless
you are about to write to it.

Three sections:

```markdown
## Conventions

Fix ESLint issues, never suppress them
PRs should be small and focused

## Notes

Auth middleware reads JWT from Authorization header only, not cookies
Can't use fs.watch on NFS mounts -- use polling

## Decisions

Chose uuid v4 over nanoid because nanoid causes ESM/CJS issues in this build setup
```

## Conventions Section

Things the user told you, corrected you on, or agreed on together:

- Working methods, such as "PRs should be small and focused"
- Project-wide rules, such as "always test the unhappy path"
- Preferences stated during the session

Write here immediately when the user states a preference or corrects your
approach. Do not accumulate notes to write later.

## Notes Section

Things you discovered while working:

- A constraint you hit
- A non-obvious fact about the codebase
- A gotcha that would be rediscovered after context resets

If you had to look for it, write it down. If discovering it required reading
code, running a command, or tracing a call path, the next session would have to
do the same work again.

If a note becomes wrong or outdated, update or delete it. Stale notes are worse
than no notes.

## Decisions Section

Choices you made and why:

- Chose X over Y because Z
- Ruled out approach X because Y
- Picked this library, pattern, or structure for this reason

Write here when you commit to an approach after considering alternatives. The
reasoning matters, not just what you chose.

## What Never Goes Here

- Status reports
- Todo lists or implementation plans
- Summaries of what you just did
- Anything that reads like documentation

## First Time

Create the file only when you have something worth writing. Do not create it
empty.
