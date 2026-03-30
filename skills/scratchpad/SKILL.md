---
name: scratchpad
description: Read and write .ergon.studio/scratchpad.md — your private notes that survive context compaction and session boundaries
---

# Scratchpad

Your private notes. One file: `.ergon.studio/scratchpad.md`

Two sections:

```markdown
## Conventions

Fix ESLint issues, never suppress them
PRs should be small and focused

## Notes

Auth middleware reads JWT from Authorization header only, not cookies
Can't use fs.watch on NFS mounts — use polling
Chose uuid v4 over nanoid — nanoid causes ESM/CJS issues in this build setup
```

Already injected into your context automatically — you don't need to read it manually unless you're about to write to it.

---

## Conventions section

Things the user told you, corrected you on, or agreed on together:
- Working methods ("PRs should be small and focused")
- Project-wide rules ("always test the unhappy path")
- Preferences stated during the session

Write here immediately when the user states a preference or corrects your approach. Don't accumulate — write it now.

---

## Notes section

Things you discovered while working:
- A constraint you hit (can't do X because Y)
- A non-obvious fact about the codebase (how something actually works)
- A decision you made and why (chose X over Y because Z)
- A gotcha that would bite you again after context resets

Write here the moment you discover it — not at the end of the task.

If a note becomes wrong or outdated, update or delete it. Stale notes are worse than no notes.

---

## What never goes here

- Status reports ("11 of 12 tasks complete")
- Todo lists or implementation plans
- Summaries of what you just did
- Anything that reads like documentation

---

## First time

Create the file only when you have something worth writing. Don't create it empty.
