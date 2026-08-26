# Handoff

## Completed this session

Implemented **all 14 findings** from `.ergon.studio/REVIEW-dsh-2026-08-26.md` (2 HIGH, 4 MEDIUM, 8 LOW) in `dsh/` (@ergon.studio/dsh, the DSH cordis plugin), plus one drive-by fix. Quality controller **APPROVED** (first pass, no rejections); the three actionable non-blocking nits it raised were closed in the same session.

Findings → implementation (all in `dsh/`):
- **H1** subagent turns triggered recall+save → `delegationDepth(session)` gate in both handlers (`src/plugin.ts`).
- **H2** unbounded nesting via `debate`/`run_parallel` → both added to `BASE_DENY` (`src/roster.ts`) + explicit `maxDepth: SPAWN_MAX_DEPTH` (=3) on both spawn sites (`src/debate.ts`).
- **M1** three profile-lifetime `Map`s → `WeakMap` + header comment corrected.
- **M2** `agents/orchestrator.md` no longer references the nonexistent `task` tool.
- **M3** contracts §4-documented pre-step recall fallback implemented (waterfall, user-kind only, depth-0 only, deduped) + tradeoff documented.
- **M4** `run_parallel` hard cap of 10 tasks (`MAX_PARALLEL_TASKS`), documented in tool + param descriptions.
- **L1** `prompts/steward.md` legacy llama.cpp block deleted.
- **L2** `EXPECTED_ROSTER_IDS` (10) — `loadRoster` warns on extras, `generatePreset` throws (build fails).
- **L3** exported pure `clampRecallLimit(requested, fallback)` (trunc, clamp [1,20]) used by the `memory_search` tool.
- **L4** `preset-gen.ts` now emits the harness's `!!js` platform gates: tool-bash disabled on win32 + new tool-pwsh row (preset has 16 rows).
- **L5** CLI `init` reports + keeps drifted skills (`kept` bucket) instead of throwing.
- **L6** `install.ts` self-install repairs a missing `preset.yml` (new test added post-approval).
- **L7** failed/timed-out recall clears its `lastRecalledText` marker (`forgetIfOwned`) so an identical re-queued message is retried; settled NONE/empty keeps the marker.
- **L8** contracts §1: schemastery ≠ zod (bare `z.string()` optional).
- **Drive-by**: `status()` no longer reports the pnpm `profiles/node_modules` artifact as a missing-plugin profile.

## Verification state

- `npm test` in `dsh/`: **102/102** (was 89; +13 new).
- `npm run build` (tsc + preset regen) clean; `presets/ergon/agent.cordis.yml` regenerated with 16 rows, `!!js` gates at the tool-bash/tool-pwsh rows.
- Repacked `ergon.studio-dsh-0.1.1.tgz` (gitignored — do NOT commit).
- Deployed: web profile `node_modules` updated (remove+add — see Notes for why plain add fails); `node dist/cli.js init --force --profile web` refreshed the live preset; `diff -r` confirms `~/.dsh/.agent-presets/ergon` == generated; `status` clean (2 profiles, both installed); deployed `dist/` byte-identical to current build.

## Start here next session

1. **Restart the dsh `web` profile process** — the plugin code (H1/M1/M3/M4/L3/L7 behavior) only takes effect on process restart. Preset-level changes (M2 persona, L4 platform rows, H2 specialist deny lists) already apply to new sessions.
2. **Commit the changeset** (16 modified files under `dsh/`, all tracked; working tree also has the untracked review file `.ergon.studio/REVIEW-dsh-2026-08-26.md` — decide whether to commit it too; tgz is gitignored). Suggested scope: one commit for the 14 findings, or one per letter group if preferred.
3. Optional polish: none outstanding. QC nit #4 (keep `presetSchema`/`jsExpr` exports) was consciously accepted — see scratchpad Decisions.

## Watch out for

- **pnpm dedupes same-version `file:` installs by path**: re-`add`ing a repacked 0.1.1 tgz says "Already up to date" and does nothing. Deploy = remove then add.
- The web profile is where the plugin lives; `headless` also has it installed (older dist until its next deploy — it is not the active testing target).
- `!!js` preset lines are validated against the harness's own tag definition and its dsh-base patch, but not integration-tested against a live DSH boot — if the profile fails to start after restart, suspect preset YAML first.
- L7 retry does NOT cover an HTTP-failed rewrite (steward `rewriteQuery` returns null for both NONE and failure; QC judged the current fail-open behavior correct).
- The stale OpenCode-era HANDOFF content this replaced described a quality-loop system in `agents/`+`opencode.json` — that work predates this session and is unrelated to `dsh/`.

## Key files

- Review: `.ergon.studio/REVIEW-dsh-2026-08-26.md`
- Plugin: `dsh/src/plugin.ts` (recall/save/pre-step, tools, WeakMaps, depth gate, caps)
- `dsh/src/roster.ts` (BASE_DENY, EXPECTED_ROSTER_IDS) · `dsh/src/debate.ts` (SPAWN_MAX_DEPTH) · `dsh/src/preset-gen.ts` (!!js, tool-pwsh) · `dsh/src/install.ts` (preset.yml repair) · `dsh/src/cli.ts` (skill drift, status skip) · `dsh/src/index.ts` (re-exports)
- Tests: `dsh/tests/{plugin,roster,debate,preset}.test.mjs`
- Docs: `dsh/README.md`, `dsh/contracts.md` (pinned harness contracts — re-verify on harness upgrade)
- Context: `.ergon.studio/scratchpad.md` (conventions, notes, decisions from this session)
