# Ergon → Pi Migration Analysis

> Historical analysis. This document records the original migration thinking from
> the OpenCode plugin to standalone Pi packages. It is not the current operational
> source of truth.
>
> Current package behavior, install steps, boot behavior, ports, generated files,
> and troubleshooting live in `README.md` and the package-local READMEs under
> `packages/*/README.md`.

Comprehensive breakdown of the ergon.studio OpenCode plugin and how each piece maps to Pi primitives.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agents](#agents)
- [Plugin Hooks & Custom Tools](#plugin-hooks--custom-tools)
- [Memory Steward](#memory-steward)
- [Skills & Artifacts](#skills--artifacts)
- [Pi Mapping](#pi-mapping)
- [Migration Plan](#migration-plan)

---

## Architecture Overview

Ergon is a multi-agent orchestration plugin for OpenCode. The user talks to an **orchestrator** (lead dev persona) who coordinates specialists as needed. The orchestrator never outsources judgment — it makes the calls. Quality gates are mandatory before any task is declared complete.

A **memory steward** — a small LLM running in its own `llama-server` — watches conversations, writes durable facts to persistent memory on its own judgment, and injects relevant prior notes back into future turns.

```
┌─────────────────────────────────────────────────────┐
│                     User                            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│                 Orchestrator                         │
│  (lead dev — talks to user, coordinates team)        │
│                                                      │
│  Delegates to:                                       │
│   ├── architect    (plans)                           │
│   ├── coder        (implements)                      │
│   ├── reviewer     (quality gate — bugs)             │
│   ├── design_reviewer (quality gate — optimality)    │
│   ├── critic       (challenges assumptions)          │
│   ├── researcher   (codebase investigation)          │
│   ├── tester       (runs tests, produces evidence)   │
│   └── quality_controller (orchestrates review loop)  │
│                                                      │
│  Tools:                                              │
│   ├── task (delegate to specialist)                  │
│   ├── run_parallel (concurrent specialists)          │
│   └── debate (two-agent alternating debate)          │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────┴───────────────────────────────┐
│              Memory Steward                          │
│                                                      │
│  Recall  (before each turn):                         │
│    user message → steward rewrite → openmemory query │
│    → inject into system prompt                       │
│                                                      │
│  Save  (after each turn, async):                     │
│    last exchange → steward judge → openmemory store  │
│                                                      │
│  Backed by:                                          │
│   ├── llama-server (Qwen 3.5 4B, dedicated GPU)     │
│   └── openmemory (SQLite semantic memory)            │
└──────────────────────────────────────────────────────┘
```

---

## Agents

### Primary Agents (talk to user directly)

#### `orchestrator`

**Role:** Lead developer. Coordinates the team, makes calls, enforces quality gates.

**Personality:** Direct, opinionated, brief. Never a sycophant. Calls out bad ideas. Swearing is fine when it lands.

**Key behaviors:**

- Does trivial work itself — doesn't spin up a team to change a string
- Briefs specialists clearly with full context
- Synthesizes specialist output — doesn't forward raw
- Sends specialists back if they deliver garbage
- Asks user only when it genuinely matters
- **Mandatory:** Invokes `quality_controller` after every code task before declaring completion
- Writes `.ergon.studio/HANDOFF.md` before every final reply

**Temperature:** 0.7

---

#### `scout`

**Role:** Thinking partner + strategic planner. Explores ideas freely, then builds a rigorous top-down plan.

**Two modes:**

1. **Freeform mode** (default): Imaginative, curious, challenges assumptions. Watches for convergence, then proposes shifting to planning.

2. **Planning mode** (8-phase process, no skipping):
   1. **Optimal Solution** — Imagine best possible solution, unconstrained, from first principles
   2. **Strip It Down** — Cut over-engineering, look holistically
   3. **Compare to Current** — Read existing implementation, identify the delta
   4. **High-Level Plan** — Top-down shape of the solution
   5. **Iterative Zoom-In** — 2-3 passes of increasing specificity
   6. **Friction Points** — What's hard, risky, uncertain
   7. **Plan** — Concrete enough for orchestrator/coder to act on
   8. **Assume You're Wrong** — Attack the plan, find what doesn't fit

**Does NOT:** Write implementation code, make final decisions without user input on genuine uncertainties.

**Temperature:** 0.6

---

### Subagents (delegated work)

#### `architect`

**Role:** Plans technical approaches; thinks ten steps ahead.

**Focus:** What does this decision make easy/hard? What does it close off? Where should the design leave seams?

**Output:** Concrete plan — specific files, specific changes, specific approach. A coder should be able to start immediately.

**Does NOT:** Write code, hand-wave ("figure out details later"), over-build.

**Permissions:** No edit, no bash.

**Temperature:** 0.5

---

#### `coder`

**Role:** Takes a plan and produces working code. Not commentary. Not pseudocode.

**The One Rule:** Read before you write. Always. Every time. No exceptions.

**Focus:** Execute the brief faithfully. Stay in scope. Don't refactor unasked. Don't add features.

**When the plan is wrong:** Stop. Flag it. Let the lead dev decide.

**Temperature:** 0.2

---

#### `reviewer`

**Role:** Quality gate. Checks whether implementation matches the brief and is bug-free.

**Focus:** Did the coder do what was asked? Real bugs (logic errors, off-by-one, null handling, race conditions). Readability for future maintainers.

**Verdict:** Accept / Revise / Rethink. No hedging.

**Does NOT:** Challenge the design (that's the critic's job), rewrite code, produce vague praise.

**Permissions:** No edit.

**Temperature:** 0.2

---

#### `design_reviewer`

**Role:** Reviews code for optimality, design quality, and architectural soundness.

**Focus:**

- Optimality — is this the best approach?
- Design quality — well-structured, maintainable, extensible?
- Architectural soundness — fits project patterns?
- Performance — obvious inefficiencies?
- Trade-offs — were the right ones made?

**Verdict:** "APPROVED" or "Needs Improvement" with specific issues.

**Does NOT:** Check for bugs (reviewer's job), rewrite code, nitpick style.

**Permissions:** No edit, no bash.

**Temperature:** 0.1

---

#### `critic`

**Role:** Challenges plans and assumptions before they break in production.

**Focus:** Untested assumptions, edge cases at scale, misuse scenarios, long-term maintainability, what this makes hard to change later.

**Approach:** Rank findings by impact. Lead with what will actually kill them. Suggest alternatives when current idea is weak.

**Does NOT:** Nitpick, manufacture objections, review code for bugs (reviewer's job).

**Permissions:** No edit.

**Temperature:** 0.6

---

#### `researcher`

**Role:** Digs into the codebase to understand how things actually work.

**Focus:** Trace call paths, check git history, find tests, follow dependencies. Go looking for things nobody thought to look at.

**Output structure:**

- **Facts** — verified in code/tests/history
- **Inferences** — likely true but not fully confirmed
- **Open questions** — couldn't determine, lead dev should know

**Does NOT:** Make recommendations (architect's job), guess, dump everything found.

**Permissions:** No edit.

**Temperature:** 0.3

---

#### `tester`

**Role:** Runs tests and produces evidence. Proof, not opinions.

**Focus:** Test what's most likely to break. Test unhappy paths. Empty input, missing fields, unexpected types.

**Output:** Structured, scannable. What / How / Result / Detail per test.

**Does NOT:** Write test plans, review code quality, speculate, pad output.

**Temperature:** 0.1

---

#### `quality_controller`

**Role:** Runs the full quality loop. Returns APPROVED or REJECTED.

**The Quality Loop:**

1. **Phase 1: Reviewer Pass** — Check for bugs. If issues → REJECTED
2. **Phase 2: Design Reviewer Pass** — Check for optimality. If improvements → REJECTED
3. **Phase 3: Verification Evidence** — Verify task-specific test, build, documentation, and scope evidence. Missing blocking evidence → REJECTED

**Iteration limit:** The parent orchestrator tracks rejections and asks the user for direction after 3.

**Permissions:** No edit, no bash.

**Temperature:** 0.1

---

## Plugin Hooks & Custom Tools

### Hook: `config`

**Purpose:** Validates agent model references against actual provider endpoints.

**How it works:**

- Scans `agent.*.model` and `mode.*.model` in opencode config
- For each `provider/model` reference, hits the provider's `/models` endpoint
- Removes model config for agents whose provider doesn't serve that model
- Unreachable providers are left untouched (don't guess)
- Timeout-protected (default 3s) so dead providers don't stall launch

### Hook: `event(session.created)`

**Purpose:** Logs "Ergon session started" to opencode logs.

### Hook: `event(session.idle)`

**Purpose:** Fires the memory steward's **save path** (fire-and-forget).

**How it works:**

- Fetches last session messages via `client.session.messages`
- Walks backward to find most recent user + assistant pair
- Deduplicates by assistant message ID (prevents re-judging same exchange)
- Sends to steward's `judgeSave` → if save-worthy, writes to openmemory
- All failures silently swallowed

### Hook: `event(session.deleted)`

**Purpose:** Cleans up `pendingRecall` and `lastAttemptedAssistantId` maps.

### Hook: `chat.message` (Recall Path — Half 1)

**Purpose:** Before the main model sees the user's message, recall relevant memories.

**How it works:**

- Extracts user message text
- Sends to steward's `rewriteQuery` (strips filler → tight search query)
- Queries openmemory with the rewritten query
- Renders recall block and stashes in `pendingRecall` map (keyed by sessionID)
- **Timeout-protected** (default 5s) — opencode awaits this hook, so a hung steward would stall every turn

**Why stash instead of inject directly:** Injecting as an extra TextPart in the user message causes Qwen 3.5's Jinja template to reject it ("System message must be at the beginning"). The actual injection happens in `experimental.chat.system.transform`.

### Hook: `experimental.chat.system.transform` (Recall Path — Half 2)

**Purpose:** Injects scratchpad + recalled memories into the system prompt.

**How it works:**

- Reads `.ergon.studio/scratchpad.md` (or generates placeholder)
- Picks up recall block from `pendingRecall` map
- Appends to the **last existing system entry** (never pushes new entries — Qwen 3.5 rejects multiple system messages)
- Logs diagnostic info

### Hook: `experimental.session.compacting`

**Purpose:** Re-injects scratchpad when context is compacted so it survives long sessions.

### Custom Tool: `debate`

**Purpose:** Run two agents in an alternating coding debate.

**How it works:**

- Creates two child sessions (one per agent)
- Agent A does first pass, Agent B reviews/improves, then alternate
- Each turn ends with `Verdict: AGREE | CONTINUE | BLOCKED`
- Stops on AGREE, BLOCKED, or `max_turns` (default 6)
- Returns rendered transcript
- Cleans up child sessions in `finally`

### Custom Tool: `run_parallel`

**Purpose:** Run multiple agents concurrently, return combined output.

**How it works:**

- Creates child sessions for each task
- Runs all via `Promise.all`
- Returns combined output with agent headers
- Cleans up child sessions
- **Warning:** Avoid write-capable agents in parallel (file conflicts)

---

## Memory Steward

### Architecture

A small LLM (~4B parameters) running in its own `llama-server` process on a dedicated GPU, permanently resident so it's never evicted by main-model swaps.

```
User message
    │
    ▼
chat.message hook
    │
    ├──► steward.rewriteQuery() ──► "test implementation" (or null/NONE)
    │                                   │
    │                                   ▼
    │                          openmemory.query() ──► [memories]
    │                                   │
    │                                   ▼
    │                          Inject into system prompt
    │
Main model responds
    │
    ▼
session.idle event
    │
    ├──► steward.judgeSave() ──► { save: { content: "..." } } (or null)
    │                                │
    │                                ▼
    │                       openmemory.store()
```

### Two Jobs

| Job | When | Prompt | Output |
|-----|------|--------|--------|
| **Rewrite** | Before each turn (sync) | `## rewrite` in `prompts/steward.md` | 3-8 word search query, or `NONE` |
| **Judge** | After each turn (async) | `## judge` in `prompts/steward.md` | JSON: `{ "save": null }` or `{ "save": { "content": "..." } }` |

### Single Source of Truth

`prompts/steward.md` contains:

- YAML frontmatter: client config (URL, model, temperature) + service runtime config (binary path, model path, GPU device, inference flags)
- Body: `## rewrite` and `## judge` prompt sections

Both `src/steward.ts` (client) and `scripts/run-steward.sh` (service launcher) parse the same file.

### Graceful Degradation

Every external dependency has a silent fallback:

- openmemory-js not installed → recall and save no-op
- Steward not running → connection refused → no-op
- Malformed JSON → parsed as null → no-op
- Steward returns NONE → recall skipped
- Query returns empty → no injection
- Store throws → swallowed

**The steward never blocks a turn, breaks a plugin load, or crashes a session.**

### Dependencies

- **llama-server** (Vulkan build) — serves the steward model
- **openmemory-js** — SQLite-backed semantic memory with synthetic embeddings
- **systemd user service** (`llama-steward.service`) — keeps steward permanently resident

---

## Skills & Artifacts

### `.ergon.studio/scratchpad.md`

**Purpose:** Persistent project notes that survive context compaction and session boundaries.

**Three sections:**

- `## Conventions` — User-stated principles and working methods
- `## Notes` — Discovered constraints, non-obvious facts, gotchas
- `## Decisions` — Choices made and reasoning (what was chosen, what was ruled out, why)

**Lifecycle:**

- Created on-demand when there's something worth writing
- Injected into system prompt automatically via `experimental.chat.system.transform`
- Re-injected after compaction via `experimental.session.compacting`
- Agents write to it immediately when they discover something or the user states a preference

### `.ergon.studio/HANDOFF.md`

**Purpose:** Session-to-session continuity. Written at end of session, read at start of next.

**Structure:**

- Completed this session
- In progress
- Decisions pending
- Start here next session
- Watch out for

### Quality evidence

Quality status is per task rather than mutable project state. The
`quality_controller` evaluates the original request, reviewer and design-reviewer
verdicts, relevant verification commands, documentation when applicable, and
scope integrity. It returns an exact APPROVED or REJECTED footer.

### `skills/handoff/SKILL.md`

Teaches agents how to read/write handoff notes. When to write, when to clear, what belongs/doesn't belong.

### `skills/scratchpad/SKILL.md`

Teaches agents how to use the scratchpad. What goes in each section, what never goes here, when to create the file.

---

## Pi Mapping

### Primitive Comparison

| Ergon (OpenCode) | Pi Equivalent | Notes |
|-----------------|---------------|-------|
| Agent definitions (`.md` with frontmatter) | **Custom Pi agents** (`.pi/agents/*.md`) | Nearly identical format — Pi uses the same markdown + frontmatter pattern |
| `task` tool (delegate to agent) | **`subagent({ agent, task })`** | Pi's subagent system is the native equivalent |
| `run_parallel` tool | **`subagent({ tasks: [...] })`** | Pi has native parallel subagent execution |
| `debate` tool | **Custom Pi extension tool** or **chain** | Needs custom implementation — Pi doesn't have a built-in debate primitive |
| Permission restrictions (`edit: deny`, `bash: deny`) | **Agent `tools` field** | Pi agents declare their allowed tools explicitly — omit write tools for read-only agents |
| `chat.message` hook (recall) | **`before_agent_start` event** | Pi's extension event fires before the agent processes the prompt — inject recall as a message |
| `experimental.chat.system.transform` | **`before_agent_start` event** | Pi provides `systemPromptOptions` and `systemPrompt` modification in this event |
| `experimental.session.compacting` | **`session_before_compact` event** | Pi has native compaction events |
| `event(session.idle)` (save) | **`turn_end` event** | Pi fires after each turn completes |
| `event(session.created)` | **`session_start` event** | Direct equivalent |
| `config` hook (model validation) | **Not needed** | Pi handles model resolution natively |
| Scratchpad injection | **`before_agent_start` + `session_before_compact`** | Combine into one extension |
| Handoff skill | **Pi skill** | Nearly drop-in compatible — just adapt paths |
| Memory steward (llama-server) | **External service (unchanged)** | The steward architecture stays the same — only the plugin hooks change |
| openmemory | **External service (unchanged)** | Same HTTP API, same SQLite backend |

### Component-by-Component Migration Plan

#### 1. Memory Steward Extension ⭐ Complex

**What:** Pi extension that hooks into the conversation lifecycle for recall and save.

**Replaces:** `chat.message` hook, `experimental.chat.system.transform`, `experimental.session.compacting`, `event(session.idle)`

**Pi events used:**

- `before_agent_start` — recall path (rewrite query → search openmemory → inject into system prompt)
- `turn_end` — save path (judge exchange → save to openmemory)
- `session_before_compact` — scratchpad re-injection
- `session_start` — notify session started

**Keeps unchanged:**

- `src/steward.ts` — steward HTTP client (reuse as-is)
- `src/memory.ts` — openmemory HTTP client (reuse as-is)
- `prompts/steward.md` — single source of truth for steward config + prompts
- `scripts/run-steward.sh` — steward launcher
- llama-server systemd service

**Key differences from OpenCode:**

- Pi's `before_agent_start` can inject a message AND modify the system prompt — no need for the `pendingRecall` map workaround
- Pi's `turn_end` gives us the completed turn data directly
- Pi's `session_before_compact` lets us customize what survives compaction

---

#### 2. Custom Agents

**What:** Pi agent definitions in `.pi/agents/` (project) or `~/.pi/agent/agents/` (global).

**Direct mapping (similar roles, adapt system prompts):**

| Ergon Agent | Pi Agent | Overlap with Pi builtins? |
|------------|----------|---------------------------|
| `orchestrator` | Custom: `orchestrator` | No — unique personality and orchestration rules |
| `scout` | Custom: `scout` | ⚠️ Pi has builtin `scout` — ours is different (thinking partner + 8-phase planner) |
| `architect` | Custom: `architect` | Partial overlap with Pi's `planner` — ours has specific "think 10 steps ahead" persona |
| `coder` | Custom: `coder` | Overlap with Pi's `worker` — ours has "read before you write" rule and strict scoping |
| `reviewer` | Custom: `reviewer` | Overlap with Pi's `reviewer` — ours has Accept/Revise/Rethink verdict system |
| `design_reviewer` | Custom: `design_reviewer` | No Pi equivalent — unique role |
| `critic` | Custom: `critic` | No Pi equivalent — unique role |
| `researcher` | Custom: `researcher` | Overlap with Pi's `researcher` — ours has Facts/Inferences/Open Questions output |
| `tester` | Custom: `tester` | No Pi equivalent — unique role |
| `quality_controller` | Custom: `quality_controller` | No Pi equivalent — but could become a **chain** instead |

**Permission model:** Instead of `permission: { edit: deny }`, Pi agents declare `tools: read, bash` (only list allowed tools). Read-only agents get `tools: read` (or omit entirely for tool-less agents).

---

#### 3. Quality Controller → Agent-Owned Sequence

The quality controller owns a sequential workflow:

1. Run reviewer
2. If accepted, run design_reviewer
3. If approved, verify task-specific test, build, documentation, and scope evidence
4. If evidence is missing, invoke tester once
5. Return an exact APPROVED or REJECTED footer

The parent orchestrator supplies the original request, changes, and verification
already performed. It also owns retry counting across fresh quality-controller
invocations. The extension exposes the delegation tools but does not replace the
quality controller's judgment with deterministic chain state.

---

#### 4. Debate Tool → Standalone Pi Package

Implemented as `@ergon.studio/pi-debate` under `pi/packages/debate`.

- Registers a global `debate` tool without changing Pi's active tool selection
- Creates two independent role-based temporary Pi sessions and alternates turns
- Uses the parent session's current model, thinking level, active tools, and working directory
- Stops on an exact terminal `AGREE` or `BLOCKED` verdict, or `max_turns`
- Cleans up both temporary sessions after success, failure, or abort

---

#### 5. Skills & Artifacts

| Artifact | Migration |
|----------|-----------|
| `skills/scratchpad/SKILL.md` | **Direct port** to `~/.pi/agent/skills/scratchpad/` — minimal changes needed |
| `skills/handoff/SKILL.md` | **Direct port** to `~/.pi/agent/skills/handoff/` — minimal changes needed |
| `.ergon.studio/scratchpad.md` | Keep as-is — path stays the same |
| `.ergon.studio/HANDOFF.md` | Keep as-is — path stays the same |
| Quality evidence | Keep per invocation; do not create a project-level completion file |

---

#### 6. What Goes Away

| Ergon Feature | Why It's Not Needed in Pi |
|--------------|--------------------------|
| `config` hook (model validation) | Pi resolves models natively |
| `pendingRecall` map workaround | Pi's `before_agent_start` can modify system prompt directly |
| `tool.run_parallel` custom tool | Pi has native `subagent({ tasks: [...] })` |
| `task` tool | Pi has native `subagent({ agent, task })` |
| `ergon init` CLI | Pi discovers agents/skills from standard locations automatically |
| Permission restrictions in config | Pi uses positive tool lists per agent |
| System prompt system-message workaround | Pi handles system prompt composition properly |

---

## Migration Plan

### Phase 1: Foundation (Skills + Artifacts)

- [ ] Port `scratchpad` skill to Pi format
- [ ] Port `handoff` skill to Pi format
- [ ] Verify `.ergon.studio/` artifact paths work with Pi

### Phase 2: Agents

- [ ] Port `orchestrator` agent to Pi format
- [ ] Port `scout` agent (handle name collision with Pi builtin)
- [ ] Port `architect` agent
- [ ] Port `coder` agent
- [ ] Port `reviewer` agent
- [ ] Port `design_reviewer` agent
- [ ] Port `critic` agent
- [ ] Port `researcher` agent
- [ ] Port `tester` agent
- [ ] Port `quality_controller` agent (or convert to chain)

### Phase 3: Memory Steward Extension

- [ ] Create Pi extension skeleton
- [ ] Port recall path (`before_agent_start` event)
- [ ] Port save path (`turn_end` event)
- [ ] Port scratchpad injection (`before_agent_start`)
- [ ] Port compaction survival (`session_before_compact`)
- [ ] Reuse existing `steward.ts` and `memory.ts` modules

### Phase 4: Debate Tool

- [ ] Decide approach (extension tool vs. prompt template vs. chain)
- [ ] Implement

### Phase 5: Quality Workflow

- [ ] Create quality-check chain
- [ ] Integrate with orchestrator agent prompt

### Phase 6: Testing & Iteration

- [ ] Test orchestrator + subagent delegation
- [ ] Test memory steward recall/save
- [ ] Test quality loop
- [x] Test debate tool
- [ ] Iterate on agent prompts based on real usage
