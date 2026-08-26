# DSH contract pin — @ergon.studio/dsh

Verified against `@deepseek-ai/dsh` **v0.1.1-rc.2** at
`/home/aristath/.nvm/versions/node/v26.7.0/lib/node_modules/@deepseek-ai/dsh/`
(2026-07-19). Every claim below was read from the installed `lib/index.js`
sources, not the READMEs. Re-verify on harness upgrade; pre-1.0 APIs can move.

## 1. Static package plugin module shape

Plain ESM, **named exports** (reference: `dsh-persona/lib/index.js`):

```js
import z from "@deepseek-ai/schemastery";
const name = "ergon";                    // cordis plugin name (string)
const inject = ["systemPrompt"];         // hard dependencies (string[]), optional
const Config = z.object({ ... });        // zod schema, optional; validated + defaulted
function apply(ctx, config) { ... }      // ctx = plugin scope context
export { Config, apply, inject, name };
```

- `config` arrives only if `Config` is exported; zod defaults are applied.
- `inject`: declare **only** hard dependencies. Accessing `ctx.<svc>` without
  declaring it is rejected by the Guard. Optional deps: `ctx.get("svc")` +
  absence check.
- Service rows mounted in a preset live in an **entry-local realm**
  (`isolate: true` group), so this plugin's registrations are scoped to the
  mounting preset — never the host composition.

## 2. Plugin context API (from shipped plugins)

| API | Use | Evidence |
| --- | --- | --- |
| `ctx.on(event, listener)` | event subscription; disposer bound to plugin fiber | dsh-plan-mode, dsh-goal-round-driver |
| `ctx.effect(() => reg, label)` | own a contribution that returns a disposer | dsh-persona: `ctx.effect(() => ctx.systemPrompt.section({...}), "persona.section()")` |
| `ctx.get(name)` | optional service, no inject needed | cordis-plugin-development SKILL |
| `ctx.logger.warn(...)` | diagnostics | dsh-plan-mode |
| `ctx.tools.register(defineTool({...}))` | model tool | dsh-tool-subagent, dsh-tool-web |
| `ctx.systemPrompt.section({name, order, text, complete?})` | static system-prompt prefix section | dsh-persona, dsh-plan-mode |
| `ctx.systemPrompt.context({name, order, text})` | **dynamic context** (snapshot, see §4) | dsh-system-prompt lib |
| `ctx.subagents.registerProvider(p)` / `.start(name, req)` | subagent seam | dsh-subagent-spawn-in-process, dsh-tool-subagent |
| `ctx.agents.get(sessionId)` | live agent lookup from session | dsh-goal-round-driver |
| `ctx.inject([...], (c) => ...)` | scoped service callback | dsh-goal, dsh-plan-mode |

## 3. Event contracts (listener signatures, verified in code)

Global-scope subscription; agent-scoped events carry the agent injected by the
fused dispatcher. Waterfall listeners receive `(payload, next)` and **must**
call/return `next()`.

### `agent/pre-step` (waterfall) — the step gate

Dispatched by `AgentLoop.preStep()` **after** assembly, before messages enter
the step.

```js
// payload
{ messages: UserMessage[], turn: number, step: number, signal: AbortSignal, agent }
// default waterfall result
{ kind: "enter", messages: claimedOrClaimedPlusContext }
// listener can return
{ kind: "reject" }                      // closes the turn (reason: "blocked")
{ ...decision, messages: [...] }        // append/prepend entering messages
```

Reference (dsh-plan-mode):

```js
ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
  const decision = await next();
  if (decision.kind === "reject" || signal.aborted) return decision;
  return { ...decision, messages: [...decision.messages, narration] };
});
```

**Timing fact:** `systemPrompt.assemble()` runs *before* this waterfall inside
`preStep()`. A pre-step listener can therefore never inject into the *current*
step's assembly — the earliest a cache update becomes visible is the **next**
assembly (next step of the same turn).

### `agent/inbox/inserted` (notification) — message queued

Emitted by the AgentLoop inbox when any message is inserted
(followup/steer/inject), **before** the driver wakes. Payload: `{ message }`
(+ injected `agent`). This is the earliest observable moment a user message
exists — the recall trigger point.

`agent/inbox/claimed { message, turn }` — at claim time (step 1 for queued
followups).

### `session/event` (notification) — durable session log

```js
ctx.on("session/event", (session, event) => {
  const agent = ctx.agents.get(session.id);
  if (agent === void 0 || agent.session !== session) return;
  // event: { type, data, seq, source? }
});
```

Durable types used here (all `session.append`-ed by the loop, so they exist on
the log and replay): `turn/start {turn}`, `turn/end {turn, reason}`,
`step/start {turn, step}`, `step/end {turn, step, ...}`, `user/message`,
`assistant/message`, `tool/call`, `tool/result`, plus `compaction/*`.

`turn/end.reason` kinds: `completed`, `blocked`, `max-tokens`, `aborted`,
`disposed` (durable `aborted` for user/parent cancels). **Save hook** =
`turn/end` with reason `completed` (parity with OpenCode idle-save), run
fire-and-forget, off the critical path.

### Others

- `agent/session-start { agent }` (notification; dsh-goal)
- `agent/created` / `agent/disposed` (registry lifecycle)
- `agent/status { status }`
- `tools/post-execute (exec, result, next)` (waterfall) — result can carry
  `additionalContexts: UserMessage[]` prepended to the next step (dsh-repeat-tool-reminder)
- `subagent/start` / `subagent/end` — service-scoped to the delegating parent;
  share a `runId`; `subagent/end` carries `lastAssistantMessage`

## 4. System prompt: sections vs contexts

`assemble(context = {})` with `context = { agent, scope: agent, signal? }`
(`assembleContextFor`, dsh-agent). Assembly = merge global+scope layers (scope
shadows), evaluate:

```js
text: typeof section.text === "function" ? section.text(context) : section.text
```

Then the `system-prompt/assemble` waterfall (final value authoritative, except
a `complete` section is re-imposed alone).

- **`section({name, order, text, complete?})`** — part of the *system prompt
  prefix*. Stable text = KV-cache friendly. Text function may return `""`.
- **`context({name, order, text})`** — dynamic **runtime context**. All context
  entries are interpolated (`{{var}}`), empty ones filtered, joined, and passed
  to `RuntimeContextProjection.project()`, which appends **one** user-role
  snapshot message to the session **only when the rendered text differs** from
  the retained snapshot (`source: {kind:'plugin', plugin, form:'snapshot',
  sections}`); the retained reference is cleared when a replacement surface
  event (compaction/prune) supersedes it. Consequences:
  - history stays append-only; prefix never busts the KV cache;
  - a changing context costs one snapshot message *per change*, not per step;
  - text functions are called **synchronously** — no `await` inside them.
  Async work must happen in an event listener that fills a cache the text
  function reads.

**Ergon recall design (consequence of the above):**

1. `agent/inbox/inserted` listener: on a user-authored message, kick off
   fire-and-forget steward recall (query rewrite → embed → openmemory search),
   fail-open, bounded timeout; result stored in a `WeakMap<Agent, string>`.
2. `systemPrompt.context({ name: "ergon:memory", order: 95,
   text: (c) => c.agent ? recall.get(c.agent) ?? "" : "" })`.
3. `agent/pre-step` listener (fallback + freshness): re-trigger recall when the
   claimed batch contains a user message not yet recalled.

Latency budget: recall visible to the model at step 1 if the 4B rewrite+search
beats the first assembly; otherwise step 2 of the same turn. Acceptable;
scratchpad (synchronous file read) is visible at step 1.

**Save design:** `session/event` → `turn/end` (reason `completed`) → gather the
turn's user+assistant text by folding `session.events` since that turn's
`turn/start` → fire-and-forget steward `judge` → conditional `store`
(openmemory, key = session id). Errors logged, never propagated.

## 5. Subagent seam (debate + specialists)

Providers registered by `dsh-subagent-spawn-in-process` (name **`spawn`**,
`providerName` default) and fork (name **`fork`**). Spawn capabilities:
`{outputSchema: true, depthLimit: true, toolFilter: true, persona: true}`,
`inheritsParentContext: false`.

```js
const run = await ctx.subagents.start("spawn", {
  label,                        // display label (durable descriptor)
  prompt: [{ type: "text", text }],  // UserMessage content blocks
  parent,                       // calling agent (exec.agent)
  signal,                       // exec.signal
  persona,                      // optional string → scoped shadowing persona section
  toolFilter: { deny: [...] },  // or { allow: [...] }; exactly one style
  outputSchema,                 // optional JSON object schema → structured result
  maxDepth,                     // optional number
  agentOptions                  // optional { provider?, model?, maxTokens? }
});
// run.result → Promise<SubagentResult>; run.dispose() idempotent
const { output, structured, diagnostic, stopReason } = await settleRun(run);
// settleRun (dsh-subagent) NEVER rejects: stopReason ∈ completed|aborted|error
```

`assertSubagentMaxDepth` guards depth; `assertObjectJsonSchema` validates
outputSchema (object-rooted). The in-process driver builds the child with
`applyChildComposition(childCtx, parent, { persona, toolFilter })` in the
child's creation window — child composition = parent preset rows + persona +
toolFilter (this is what makes each preset specialist row self-contained).

## 6. Tools

```js
import { defineTool } from "@deepseek-ai/dsh-tools";
const dispose = ctx.tools.register(defineTool({
  name: "debate",
  description: "...",
  parameters: { arg: { type: "string", required: true, description: "..." } },
  output: { schema: { /* object schema the body must return */ } },
  // optional: render(exec, result) → result view; finalizeContent(exec, result);
  //           timeoutMs; isConcurrencySafe(args)
  async execute(args, exec) {
    // exec: { callId, name, arguments, signal /* readonly, required */, agent?, parent? }
    const parent = exec.agent; // the calling agent
    return { /* matches output.schema */ };
  },
}));
```

Result views: `{card:'generic'|'terminal'|'diff'|'search'|'read'|'web', ...}`
(render callback). Bodies must cooperate with `exec.signal` and return exactly
the canonical JSON value declared by `output.schema`.

## 7. Agent preset mechanics

- Location: `~/.dsh/.agent-presets/<id>/{agent.cordis.yml,preset.yml}`.
  `preset.yml`: `{ name, description, order? }`. User-owned; never edit shipped
  presets in the harness install.
- Mounted **once per process** under a standing scope when a session selects
  the preset (`agent-preset/selected`); `standingKeyFor(id)` validates a mount;
  new sessions pick up edits via mtime+size generation stamping (no restart).
- **Row package resolution** (`dsh-agent-presets` PresetTree.import):
  - absolute filesystem path → `file://` URL, always resolves;
  - `.`/`./` relative → against the preset directory (files travel with preset);
  - **bare specifier → resolved against the host composition base** = the
    profile directory (Loader `baseUrl`), i.e. Node resolution from
    `~/.dsh/profiles/<profile>/` → the profile's `node_modules`. So
    `@deepseek-ai/dsh-*` and anything installed via
    `dsh plugin --profile <p> add <spec>` resolves by **npm name**;
    `cordis:` builtins pass through.
- Service rows require an `isolate: true` group (entry-local realm).
- Presets must not own registries/agent-loop/persistence/sandbox rows — the
  host composition owns those.
- The `agent-presets` **service row** is inserted only by `dsh-web-app`
  (`config: {default: standard}`). Headless profiles need a `--patch` overlay
  inserting the row for mount testing.

## 7a. Profile bundles & the `dsh plugin` forwarder (verified 2026-07-25)

`dsh plugin --profile <name> <args...>` (lib/plugin-9h8shc4d.js) is a **thin
pnpm forwarder**: initializes the profile on first use, runs `pnpm <args...>`
with cwd = the profile dir (relative `./…` specs re-anchored to the *invoking*
cwd), then **reconciles** `dsh.profile.bundles` in the profile `package.json`
against the installed state:

- A dependency whose installed `package.json` declares
  `"dsh": { "bundle": { "patch": "./<file>.yml" } }` is **appended** to
  `dsh.profile.bundles` (bundle-less deps get a one-time stderr warning and
  stay plain dependencies).
- Removing the dependency removes it from the list. Bundles are applied in
  list order **after** the template bundles (`web`: base+web-app; `headless`:
  base+headless) and **before** the profile's own `cordis.patch.yml`
  (the user layer — it wins per row, last write wins).

Bundle patch format (`loadProfile` + `applyEntryPatches`, dsh-app-boot):
a YAML list of patch entries —
- `- insert: [ {id, name, config?, ...} ]` appends rows to the profile root;
- `- id: <rowId>` + any keys overrides that row; a `config` override
  **replaces the whole config** (no merge);
- an id patch targeting a row that does not exist **warns and skips**
  (`patch: entry "X" not found`) — so one bundle patch is portable across
  profiles that lack the row (ergon's `agent-presets` default override in
  headless).

Consequences for the ergon architecture:
- The plugin mounts at **profile level** for every session (bundle row
  `ergon-plugin`), not inside the preset — a preset row would double-mount it
  in profiles that also carry the bundle (duplicate tools + duplicate event
  listeners). The generated preset therefore contains no plugin row, and
  `validatePresetFile` rejects stale presets that still carry one.
- `agent-presets.config.default` is the row that picks the session default
  preset; the user layer / settings document can override it later.
- `dsh --profile <p> --dump-config` composes the tree **without booting** —
  the pre-restart verification path (also `--dump-default-config`, which
  skips the user layer).

## 8. `dsh-tool-subagent` row config (specialist tools)

```yaml
name: "@deepseek-ai/dsh-tool-subagent"
config:
  provider: spawn            # required
  toolName: coder            # the tool the model sees
  enableRunInBackground: true   # default
  backgroundMode: one-shot  # one-shot | continuable (default one-shot)
  agentOptions:             # optional { provider?, model?, maxTokens? }
  persona: |                # optional persona template ({{model}}, {{cwd}} vars)
  toolFilter:               # { allow? , deny? } — one style
    deny: [edit, write, bash]
  maxDepth: 3               # number | "provider-managed" (default 3)
```

inject: `["tools","subagents","systemPrompt"]`; prompt order 116.5.

## 9. Verification commands

- `dsh --profile headless --patch <overlay.yml>` — separate process; overlay
  inserts the `agent-presets` row (`config: {default: ergon}`) without touching
  the live web profile.
- `--dump-config` / `--dump-default-config` — composed tree without boot.
- Live: preset edits apply to **new** sessions in the running web process.

## 10. OpenCode → DSH mapping (build decisions)

| OpenCode | DSH |
| --- | --- |
| 10 agent files (frontmatter) | preset: 1 persona row (orchestrator) + 9 `dsh-tool-subagent` rows |
| `debate` tool | `debate` tool via `ctx.tools.register` + `ctx.subagents.start("spawn", ...)` × N + `settleRun` |
| `run_parallel` tool | same seam, N parallel one-shots |
| memory steward recall (chat.message → system.transform) | `agent/inbox/inserted` → cache → `systemPrompt.context()` snapshot |
| memory steward save (session.idle) | `session/event` `turn/end` completed → fire-and-forget |
| scratchpad re-inject at compaction | `systemPrompt.context()` reading the file each assembly (snapshot diff) — no compaction hook needed |
| openmemory MCP for main model | `memory_search` tool on the ergon plugin (profile-level via the bundle) |
| skills (scratchpad, handoff) | SKILL.md dirs → `~/.dsh/skills/` via installer |
| per-agent `opencode.json` permission denies | `toolFilter.deny` per specialist row |
| `ergon init` CLI | `npx @ergon.studio/dsh init` (writes preset, skills, validates) |
