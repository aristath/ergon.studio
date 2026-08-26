# Complete review — `@ergon.studio/dsh` (the DSH implementation)

Date: 2026-08-26 · Scope: `dsh/` in this repo (commit `63f3bf3`, working tree clean)
Reviewed against: installed harness `@deepseek-ai/dsh` **0.1.1-rc.2** (the version pinned in
`contracts.md`), by reading the installed `lib/` sources — not just the READMEs.

> Note: the npm harness package is **stale** — upstream plugin-side fixes are committed and
> pushed but not re-published. This review therefore treats 0.1.1-rc.2 as the contract
> baseline (matching `contracts.md`). Re-verify on the next harness install.

## Verdict

Strong, shippable implementation. Clean architecture, honest contracts, and a genuinely good
test suite (89/89 pass, strict `tsc` clean, generated preset byte-in-sync with the shipped
`standard` preset apart from intended deltas, checked-in tgz in sync with source).

Two findings are significant enough to fix before relying on the memory system or using
`debate`/`run_parallel` in production sessions:

1. **Subagent turns trigger the memory recall *and* save hooks** → 4B cost amplification and
   (worse) long-term memory corpus pollution from debate/parallel noise.
2. **`debate`/`run_parallel` are not in the deny lists, and the plugin's spawn calls pass no
   `maxDepth`** → unbounded agent nesting is possible (the harness depth cap only guards the
   generic `subagent` tool, which defaults to `maxDepth: 3`).

Everything else is polish: stale doc comments, one stale tool reference in the live
orchestrator persona, doc/implementation drift in `contracts.md`, and a few robustness nits.

---

## What was verified (evidence)

| Check | Result |
| --- | --- |
| `npm test` (build + `node --test`) | **89/89 pass** |
| `tsc --noEmit` (strict) | clean |
| `presets/ergon` after build | no drift (generated == checked-in) |
| `ergon.studio-dsh-0.1.1.tgz` vs source | dist/agents/presets identical; package `files` correct |
| ergon preset vs shipped `standard` preset (row-level diff) | only intended deltas: +9 specialist rows, persona text, omitted `tool-pwsh` (Windows-only) and two `disabled: true` external rows |
| plan-mode policy text | byte-identical to shipped `standard` |
| Plugin config with **no** row config (the actual bundle row) | validates fine — `schemastery` treats bare `z.string()` as *optional*, so the README's "row config is optional" precedence (row → env → steward.md → defaults) holds end-to-end |
| `session/event` on resume | constructor seeds (replay/resume/fork) **do not** re-emit → no re-judging of past turns |
| `agent/inbox/inserted` for subagent first messages | **yes** — the in-process spawn driver delivers the child prompt via `child.followup(createUserMessage({…, source: {kind: "user"}}))`, which goes through the inbox and fires the event |
| Subagent sessions in `~/.dsh/sessions` | none yet (no delegation has been used in the field) → finding 1 is latent, not observed |
| Memory store (`openmemory` on :8082) | up, populated, no debate noise yet |

---

## Findings

### HIGH

#### H1 — Subagent turns fire the recall and save hooks (cost + memory pollution)

`src/plugin.ts` subscribes at **profile scope**:

- `agent/inbox/inserted` → steward rewrite + openmemory search (recall)
- `session/event` on `turn/end` reason `completed` → steward judge + possible openmemory save

Both hooks only check that `ctx.agents.get(session.id)` exists — they do **not** check
delegation depth. The harness delivers every spawned child's first message through the inbox
as a `user`-kind message, and `session/event` is a firehose for **every** session append in
the process (dsh-session `append()` → `invokeContainedSessionObservers(…, "session/event", …)`;
verified in `dsh-session/lib/index.js`). Consequences per debate turn / parallel task / any
`subagent` tool use:

- one `rewriteQuery` call to the 4B + one `/memory/query` (a 6-turn debate = 12 calls);
- one `judgeSave` call per child turn end, and the judge is asked to save "durable" content —
  debate transcripts and parallel-task output are exactly the technical material a 4B will
  happily persist. **This pollutes the long-term corpus that every future session recalls.**
- recall results are additionally injected into the child's own context (per-agent cache),
  from a search over the child's own brief — noisy at best.

Evidence the gate is available: child session headers carry `delegationDepth`
(`dsh-subagent` `childSessionMeta` → `delegationDepth: childDepth`; top-level sessions are 0,
verified in stored session headers).

**Fix (one line per handler):** in both handlers, skip when
`payload.agent.session?.header?.delegationDepth > 0` (recall: `agent.session.header.delegationDepth > 0`).
Subagents are one-shot workers; the top-level session is the only one that should talk to the
steward. Add tests: "child agent (depth 1) message does not trigger rewrite/query" and
"child turn/end does not trigger judge".

#### H2 — Unbounded nesting via `debate`/`run_parallel`

- `roster.ts` `BASE_DENY` denies `subagent`, `subagent_fork`, `send_message`, `list_agents`,
  `interrupt_agent`, `workflow`, `ralph`, goals, `ask_user_question` — but **not** `debate` or
  `run_parallel`.
- The harness's depth budget (`resolveChildDepth`) only applies when the caller passes
  `maxDepth`. The generic `dsh-tool-subagent` tool does (`maxDepth: … .default(3)`), but
  `src/debate.ts` calls `ctx.subagents.start("spawn", {…})` **without `maxDepth`** →
  `resolveChildDepth(parent, undefined)` → **no cap**.
- The plugin's tools are registered at profile scope (root realm), so they are visible to
  every agent in the process, including spawned children.

So: orchestrator → `debate(coder, reviewer)` → the depth-1 `coder` child can itself call
`debate`/`run_parallel` → depth 2 → … with no service-level bound. Each level multiplies LLM
work; a confused or over-eager child (the debate prompt even says "make the changes you
believe are right") is a realistic trigger. The built-in delegation tools are correctly
no-nested; the plugin's own tools are the backdoor.

**Fix (belt and suspenders):**
1. Add `debate` and `run_parallel` to `BASE_DENY` in `roster.ts` (spawns stay leaves; the
   top-level orchestrator keeps both, since the persona row carries no toolFilter). Update
   `roster.test.mjs` expectations.
2. Defense in depth: pass an explicit absolute cap in the `start()` requests in `debate.ts`
   (e.g. `maxDepth: 4`, or derive from the parent's header `delegationDepth`).

### MEDIUM

#### M1 — Profile-lifetime `Map`s keyed by per-session objects (slow leak)

`src/plugin.ts` keeps `recallCache: Map<object,string>`, `lastRecalledText: Map<object,string>`,
`savedTurns: Map<object,number>`. The plugin is mounted **once per profile process** (profile
bundle row) — not per session. The header comment in `plugin.ts` ("Mounted once per session by
the ergon agent preset") is stale — that was the OpenCode/port-era design; `preset-gen.ts`
explicitly documents that the plugin is a *profile-level* bundle mount (a preset row would
double-mount it).

Consequence: every finished session's agent/session objects (and, through `savedTurns` keys,
their full event logs) stay strongly referenced for the life of the web process. A GUI profile
running for days accumulates them all. `contracts.md` §4 even documents the *intended* design
as `WeakMap<Agent, string>` — the implementation deviated.

**Fix:** all three maps use object keys only → convert to `WeakMap` (get/set/delete are all
supported; the stale-guard and dedup logic is unaffected). Fix the header comment. (A
`session/end`-driven `delete` would also work, but WeakMap is simpler and matches the doc.)

#### M2 — Live orchestrator persona references a nonexistent `task` tool

`agents/orchestrator.md:76`: "Use the `task` tool to delegate to specialists." There is no
`task` tool in DSH — delegation is the per-specialist tools (`scout`, `architect`, `coder`, …)
plus `run_parallel`/`debate`. This is an OpenCode-port artifact in the persona the primary
session actually runs on; a model reading it will try to call `task` and fail. The rest of the
Orchestration section (lines 79–98) already describes the DSH reality correctly.

**Fix:** reword line 76 to "Use the specialist tools (`scout`, `architect`, `coder`, …) to
delegate." Rebuild + repack (preset embeds the persona).

#### M3 — `contracts.md` §4 documents behavior the code doesn't have

- §4 lists the recall cache as `WeakMap<Agent, string>` — implementation is `Map` (see M1).
- §4 point 3 promises an `agent/pre-step` fallback that re-triggers recall if a user message
  was claimed without a recall. `plugin.ts` has **no** `agent/pre-step` listener. Empirically
  recall fires for first messages today (inbox insertion happens before listener registration
  is an issue, and this session's recall worked), so it may be redundant — but the contract doc
  should match the code.

**Fix:** either implement the pre-step fallback (it's cheap insurance) or amend §4 to the
actual single-trigger design. Keep `contracts.md` honest — it's the pin for the next harness
upgrade.

#### M4 — `run_parallel` has no task-count cap

`args.tasks` is only validated as "array of {agent,brief}". N tasks → N concurrent in-process
child agents; the subagent service has no concurrency cap (checked `dsh-subagent` — the only
caps are the per-call `maxDepth` and persistence-inspection internals). The tool description
warns about write conflicts but not scale.

**Fix:** cap tasks (e.g. 8–10, consistent with the workflow tool's fan-out spirit) and mention
the cap in the tool description.

### LOW

#### L1 — `prompts/steward.md` ships machine-specific legacy config

The frontmatter still carries the legacy standalone-runner block: `llama_server_bin:
/home/aristath/llama.cpp/...`, `model_path: /home/aristath/models/...`, `device: Vulkan1`, etc.
`parseStewardMd` only reads `url`/`model`/`temperature`, so it's functionally inert — but this
file is published to npm (`files` includes `prompts/`) and leaks the local filesystem layout
into a public tarball. **Fix:** drop the legacy block (keep `scripts/run-steward.sh`'s own docs)
or move the legacy config out of the package.

#### L2 — Any `agents/*.md` becomes a roster entry

`roster.ts` `loadRoster()` treats every `*.md` in `agents/` as an agent (id = filename). A
future `README.md` or `NOTES.md` there would become a spawnable "agent" with an empty persona
and only `BASE_DENY`. **Fix:** validate the id set (or at least log/warn on unexpected ids);
`validatePresetFile`-style check in the build.

#### L3 — `memory_search` limit coercion

`Math.min(Math.max(args.limit ?? config.recallLimit, 1), 20)` — a non-numeric `limit` becomes
`NaN` and flows into the request body (`k: NaN` → JSON `null`). **Fix:** `Number.isFinite`
guard with fallback to `config.recallLimit`.

#### L4 — Generated preset drops `tool-pwsh`

`standard` carries `tool-pwsh` with `disabled: !!js process.platform !== 'win32'` (active on
Windows); the ergon preset omits the row entirely, so Windows users of the ergon preset lose
PowerShell. Also omitted: two `disabled: true` external subagent rows (no behavior delta).
**Fix (if Windows matters):** carry the row with the same platform gate. Otherwise document
the omission in `contracts.md`.

#### L5 — CLI `init` hard-fails on drifted skills; self-install silently skips

`src/cli.ts` `ensureSkills()` **throws** when an installed skill differs from the package
copy (without `--force`), aborting init mid-run — while `src/install.ts` (the self-install
path) just warns and keeps the user copy. Same situation, two behaviors. A user who hand-edits
a skill can no longer run plain `init`. **Fix:** align CLI to "report + skip" and let `--force`
be the only overwrite path (matching the README's "never overwrites a copy you have edited —
refresh one with `init --force`").

#### L6 — Preset self-install existence check is partial

`install.ts` checks only `agent.cordis.yml` for "already installed"; a half-written install
(dir exists, `agent.cordis.yml` present, `preset.yml` missing) is never repaired. Minor edge;
cheap to also check `preset.yml` and `cpSync` individually.

#### L7 — A failed recall is never retried for identical text

The dedup sets `lastRecalledText` *before* the async recall; if the steward times out, the
same message re-queued later is treated as "already recalled" and skipped. Per-agent lifetime,
so blast radius is one stuck message per session. Optional: only mark "seen" after a settled
attempt (success or explicit NONE), or clear the marker on timeout.

#### L8 — `contracts.md` §1 says "zod schema"

It's `@deepseek-ai/schemastery` (zod-compatible but different semantics: bare `z.string()` is
*optional*, which is precisely why the config-less bundle row validates). One-line note so a
future maintainer doesn't "normalize" the schema to plain zod — that would make the no-config
row a `ValidationError` and break the install.

---

## Strengths (worth keeping visible in the repo)

- **Honest contract pinning.** Every claim in `contracts.md` that I spot-checked against the
  installed 0.1.1-rc.2 sources held: module shape, `resolveConfig` Standard-Schema validation,
  event names/payloads, the subagent seam (`start`/`settleRun`, descriptor, persona +
  toolFilter), bundle-patch application order, pnpm forwarder behavior.
- **Correct config semantics.** The row → env → steward.md → defaults precedence is
  genuinely optional-config, verified end-to-end (empty row config validates; env falls back
  inside the clients).
- **Fail-open discipline.** Steward down → recall/save off, warn once; memory down → `[]`;
  incomplete package → warn, no crash; dead agents ignored. Matches the stated design, and
  each path has a test.
- **KV-cache-aware context injection.** Append-only runtime-context snapshots for memory and
  scratchpad (re-read per assembly, snapshot diff) — a genuinely good fit for the harness's
  projection model, and it's what makes scratchpad survive compaction without a hook.
- **Recall correctness details.** Staleness guard (newer message supersedes slow older),
  per-agent dedup, `source.kind` filtering with the "absent ⇒ user" default, per-stage
  timeouts. All tested, including the subtle ones.
- **Save correctness.** Completed-only filter (parity with the Pi-side `3bef697` change),
  per-session turn dedup, reverse-walk turn text collection, resume-safety (seeded events
  don't re-emit).
- **Preset parity discipline.** Generated from source at build time, validated structurally,
  `validatePresetFile` actively rejects the stale `ergon-plugin` row, and the diff against
  shipped `standard` is exactly the intended delta — including byte-identical plan-mode text.
- **Test suite.** 89 tests, real HTTP stubs, env isolation (`DSH_HOME`), definition-cache
  reset hooks, scripted subagent seams, and edge cases that matter (stale recall, duplicate
  turn/end, first-turn verdict no-stop, fail-open per task in `run_parallel`).

## Suggested order of work

1. H1 (depth gate on both memory hooks) + its two tests — protects the memory corpus.
2. H2 (deny `debate`/`run_parallel` in `BASE_DENY` + explicit `maxDepth` on spawns).
3. M1 (`WeakMap` ×3 + header comment) — five-minute change, same commit as H1 is natural.
4. M2 (orchestrator `task` line) — one line, but needs rebuild + `npm pack` + profile reinstall
   to take effect.
5. M3 (align `contracts.md` §4) together with 1–2.
6. M4 + L1–L8 in a polish pass.

After any source change: `npm run build && npm test`, `npm pack`, reinstall into the profile
(`dsh plugin --profile web add ./ergon.studio-dsh-0.1.1.tgz`), bump the tgz in git, and — per
the standing note — re-verify contracts once the re-published harness lands.
