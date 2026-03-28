---
name: handoff
description: Write a structured handoff note at .ergon.studio/HANDOFF.md when pausing or completing work, so the next session picks up exactly where this one left off
---

# Ergon Handoff

A handoff note lives at `.ergon.studio/HANDOFF.md`. It's the first thing the next session should read — before the scratchpad index, before anything else.

Write one when:
- You're done with a task and follow-up work is expected
- You're stopping mid-task (blocked, paused, or the session is ending)
- You've made decisions the next agent needs to know before touching anything

---

## Reading a handoff

At the start of any session, check if `.ergon.studio/HANDOFF.md` exists:

```bash
# check with read tool: .ergon.studio/HANDOFF.md
```

If it exists, read it before reading the scratchpad index. It tells you the immediate context — what was just done, what to do next, what to avoid. After you've oriented yourself and started work, you can archive or clear it.

---

## Writing a handoff

Use the `write` tool to create or overwrite `.ergon.studio/HANDOFF.md`. Keep it short — this is a note to the next agent, not a status report to management.

```markdown
# Handoff

## Completed this session
- [what was actually finished and working]

## In progress
- [what was started but not done, and where it stands]

## Decisions pending
- [anything that needs a call before work can continue]

## Start here next session
[One clear, concrete first action — not a list. What should the next agent do first?]

## Watch out for
[Anything that would bite the next agent if they didn't know — gotchas, broken state, assumptions]
```

---

## When to clear it

After the next session has read the handoff and is underway, overwrite `HANDOFF.md` with a new one reflecting current state — or delete it if the work is genuinely done and there's nothing to carry forward.

Don't let stale handoffs accumulate. One handoff, always current.

---

## What doesn't belong here

- Decisions and reasoning → those go in the scratchpads (`index.md` + topic files)
- Project conventions → those go in `conventions.md`
- Handoff is only about immediate continuity: what happened, what's next, what to watch out for
