# Ergon Studio Scratchpad

## Conventions
- Behavior-changing work in `dsh/` gets `quality_controller` approval before completion is declared.
- `.ergon.studio/HANDOFF.md` is rewritten before the final reply of every session.
- `dsh/ergon.studio-dsh-0.1.1.tgz` is gitignored build output — never commit it.

## Notes
- **pnpm dedupes same-version `file:` installs by path**: re-`add`ing a repacked tgz with an unchanged version says "Already up to date" and does NOT update the profile. Deploy = `dsh plugin --profile web remove @ergon.studio/dsh` then `add ./<tgz>`.
- `~/.dsh/profiles/node_modules` is a pnpm virtual-store byproduct of `dsh plugin add`, not a profile. The CLI `status` now skips it.
- Live deployment has two halves: tgz add updates profile `node_modules` (needs a **web profile process restart** for plugin code), while `node dist/cli.js init --force --profile web` refreshes `~/.dsh/.agent-presets/ergon/` (preset-level changes like persona text/deny lists apply to **new sessions** without restart).
- The harness stamps `session.header.delegationDepth` on subagent sessions; the plugin reads it for the H1 depth gate (absent header → 0 = top-level).
- DSH plugin `Config` schemas use `@deepseek-ai/schemastery` (zod-compatible but bare `z.string()` is optional, not required).
- Harness presets use `!!js` YAML tags for platform gates; js-yaml needs a custom `Type("tag:yaml.org,2002:js")` (constructs `{__jsExpr}`) both to dump and to load. Our `loadPresetYaml` mirrors the harness's `cordis-plugin-include` definition exactly.
- `dsh-tools` `defineTool` validates args against JSON schema: non-integer `limit` values (including NaN/null) are rejected at the tool layer before `execute` — so `clampRecallLimit` is defense-in-depth, only unit-testable directly.
- `npm test` in dsh/ always rebuilds (build + `node --test tests/*.test.mjs`); tests import from `dist/`, never `src/`.
- Debate/parallel subagents are leaves (all delegation denied in BASE_DENY), so `maxDepth: 3` is a pure hard bound that never engages legitimately.

## Decisions
- **L7**: `steward.rewriteQuery` returns null for both "steward said NONE" and HTTP failure (indistinguishable). Retry-on-failure covers timeout/throw paths only; an HTTP-failed rewrite stays "settled". QC judged this the correct fail-open degradation — changing the steward contract would ripple into the judge path for no gain.
- **M3**: the pre-step fallback is deliberately stricter than the inbox trigger — only explicit `source.kind === "user"` (inbox also accepts absent source for spawn-driver robustness). Synthetic step messages can't pollute recall queries; tradeoff documented in contracts.md §4.
- **H2**: `SPAWN_MAX_DEPTH = 3` mirrors the harness generic-subagent default; debate/parallel spawns carry it explicitly because `ctx.subagents.start` bypasses the harness's generic maxDepth=3 enforcement for tool-registered subagents.
- **L3**: guard extracted to an exported pure `clampRecallLimit` (trunc + clamp [1,20]) rather than inlining, because the tool layer already rejects bad limits and the guard must be testable.
- **QC non-blocking nit accepted, not done**: `presetSchema`/`jsExpr` exports in preset-gen.ts are module-internal (used by generator + `loadPresetYaml`); kept exported with JSDoc for future preset consumers — cosmetic, no change.
- **Drive-by**: `status()` skips `profiles/node_modules` (was reported as a bogus "plugin missing" profile).
