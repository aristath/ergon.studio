---
name: scratchpad
description: Project knowledge base at .ergon.studio/scratchpads/ — read for prior context before starting work, write to preserve decisions and findings for future agents
---

# Ergon Scratchpad System

The project's persistent knowledge base. Committed to git. Grows over the life of the project and across sessions.

**Location**: `.ergon.studio/scratchpads/` relative to the working directory.

---

## When to use

**Before starting any multi-step task**: Read `index.md` to see what knowledge exists. Pull relevant scratchpads. This is how you find prior decisions, architectural choices, ongoing work, and gotchas — without re-reading the entire codebase.

**Before replying after completing work**: Write to the scratchpad if any of these happened:
- A decision was made → write what was decided *and why*, what alternatives were considered
- Something was implemented → write what changed and any non-obvious aspects
- A problem was encountered and solved → write the problem, root cause, and fix
- A new pattern or constraint exists in the codebase → write it so the next agent doesn't have to rediscover it
- Work is in progress and you're stopping → write current state, what's done, what's next, what's blocked

Write **before** replying to the user — once you've replied and moved on, the context is gone.

Then update `index.md` if you created or renamed a file.

If no scratchpad exists for the topic, create one. If one already exists, extend it.

---

## Index format

`index.md` is the entry point. Keep it tight — one line per scratchpad, description focused on what's *inside*, not just the topic name:

```markdown
# Scratchpads

- [auth](auth.md) — JWT approach, why we chose stateless sessions, Redis cache decision
- [api](api.md) — REST conventions, versioning strategy, rate limiting design
- [testing](testing.md) — test infrastructure setup, known coverage gaps, flaky test list
- [payments](payments.md) — Stripe integration, webhook idempotency, refund edge cases
```

---

## What to write

**Good content** (write this):
- Decisions and the reasoning behind them — "we chose X because Y; Z was ruled out because..."
- Non-obvious gotchas — "the webhook handler needs idempotency keys because Stripe retries..."
- Current state of ongoing work — "auth is done; authorization is blocked on schema decision in [api.md]"
- Constraints and tradeoffs — things that look wrong but are intentional

**Bad content** (don't write this):
- Output that belongs in your response to the user
- Summaries of code that `grep`/`read` can retrieve on demand
- Status updates without substance ("worked on X today")
- Things that belong in code comments or commit messages

---

## Scratchpad structure

No enforced format. Use headings that fit the content. When starting a new scratchpad, this is a reasonable skeleton:

```markdown
# Topic Name

## Current state
What's the situation right now?

## Key decisions
What was decided here and why? What alternatives were rejected?

## Gotchas
What would trip up someone new to this area?

## Open questions / next steps
What's unresolved? What needs to happen next?
```

Keep files focused. If a scratchpad grows unwieldy, split it by sub-topic and update `index.md`.

---

## First-time setup

If `.ergon.studio/scratchpads/` doesn't exist yet:

```bash
mkdir -p .ergon.studio/scratchpads
printf '# Scratchpads\n\n' > .ergon.studio/scratchpads/index.md
```

---

## Practical tips

- Read the index first, not individual files. The index tells you what to read.
- Cross-reference between scratchpads when relevant: "see [auth.md] for session design".
- When a decision is reversed, update the scratchpad — don't leave stale context that will mislead future agents.
- Short and accurate beats long and comprehensive. Future agents skim.
