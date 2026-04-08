# ergon.studio

A multi-agent orchestration plugin for [OpenCode](https://opencode.ai).

Ergon brings a team of AI agents to your OpenCode workflow. You talk to the **orchestrator** — a lead developer who understands your goal, breaks it down, and coordinates specialists (architect, coder, reviewer, critic, researcher, tester) as needed.

It also ships with a **memory steward** — a small parallel LLM that watches your conversations, writes durable facts to persistent memory on its own judgment, and injects relevant prior notes back into future turns before the main model sees them. See [Memory Steward](#memory-steward) below.

## Why

Local models produce mediocre output on one pass. But if you make the same model plan before it codes, review after it codes, and iterate on feedback — the results get dramatically better.

Ergon adds the behavior a good lead developer adds: break problems down, bring in the right people, inspect results critically, iterate on weak spots, and decide when work is ready to ship.

## Quick Start

### Requirements

- Node.js 16.7+
- OpenCode installed

### Install

```bash
# Add to your opencode.json config
npm install ergon-studio
```

Then add to your `~/.config/opencode/opencode.json`:

```json
{
  "plugin": ["ergon-studio"],
  "default_agent": "orchestrator"
}
```

### Initialize

Run the init command to set up ergon agents:

```bash
npx ergon init
```

This will:
1. Install ergon agents to `~/.config/opencode/agents/`
2. Update your global `opencode.json` with the plugin and default agent
3. Configure permission restrictions for read-only agents

### Usage

Start a session with OpenCode and refer to the orchestrator:

```
@orchestrator Build a REST API endpoint for user authentication
```

The orchestrator will coordinate the team automatically.

## Agents

Ergon ships with seven agents:

| Agent | Role | Temperature |
|-------|------|-------------|
| `orchestrator` | Lead developer — talks to the user, coordinates the team | 0.7 |
| `architect` | Plans and designs before anyone builds | 0.5 |
| `coder` | Takes a brief and produces working code | 0.2 |
| `reviewer` | Quality gate — checks correctness and adherence to the brief | 0.2 |
| `tester` | Produces evidence — runs tests and reports findings | 0.1 |
| `critic` | Challenges assumptions and finds what a friendly team would miss | 0.6 |
| `researcher` | Digs into the codebase and gathers context before decisions | 0.3 |

### Agent Modes

- **Primary agents** (`orchestrator`): Can communicate directly with the user
- **Subagents** (all others): Run under orchestration, focused on specific tasks

## Memory Steward

The memory steward is a small (2B) LLM running alongside your main coding model in a **separate** `llama-server` process. It gives OpenCode durable, cross-session memory without relying on the main model to decide when to consult or update that memory.

### What it does

The steward has exactly two jobs, each fired at a different point in the conversation lifecycle:

| Phase | When | Job | Critical path? |
|-------|------|-----|----------------|
| **Recall** | Before the main model sees a new user message (`chat.message` hook) | Rewrite the user's message into a short search query, pull relevant memories from openmemory, and inject them as a `synthetic: true` text part so the main model sees them in the same turn | Yes — synchronous |
| **Save** | After the assistant finishes responding (`event` hook on `session.idle`) | Look at the completed exchange (last user message + last assistant response) and decide whether anything durable is worth remembering. If yes, write to openmemory. | No — fire-and-forget |

The two jobs never get conflated into one LLM call. Each prompt is laser-focused, which matters a lot for small-model reliability.

### Why this shape

- **The main LLM still has direct MCP access to openmemory** — the steward is additive, not a replacement. It's the "annoying little brother taking notes" that reminds the main model of things it would otherwise forget.
- **The recall path is the only latency tax**, and it's minimal: a single small-model call (query rewrite, ~3–8 word output) plus one local SQLite query. Save happens off the critical path.
- **Query rewriting matters for synthetic-embedding search.** "Can you please do me a favor my sweet friend and test the implementation" and "run tests" embed very differently in `OM_EMBED_KIND=synthetic` space. The steward collapses filler so the search actually hits.
- **Openmemory handles sector classification, decay, and simhash dedup itself.** The steward just says *what* to save; openmemory decides *where*, *how long to keep it*, and *whether it's a duplicate*.

### Architecture

```
user message
     │
     ▼
┌─────────────────────────────┐
│ chat.message hook (ergon)   │
│                             │
│  ┌────────────┐             │       ┌──────────────┐
│  │  steward   │──query────▶│───────▶│  openmemory  │
│  │  (port     │             │       │  (SQLite)    │
│  │   8081)    │◀──results──────────│              │
│  └────────────┘             │       └──────────────┘
│        │                    │
│        ▼                    │
│  inject synthetic text      │
│  part with recalled notes   │
└──────────┬──────────────────┘
           │
           ▼
   main coding model
   (via llama-router, e.g. Qwen3-Coder-Next 80B)
           │
           ▼
    assistant response
           │
           ▼
┌─────────────────────────────┐
│ event(session.idle) (ergon) │
│                             │
│  fetch last exchange via    │
│  client.session.messages    │
│        │                    │
│        ▼                    │
│  ┌────────────┐             │
│  │  steward   │─judge──────┐│
│  │  (port     │            ││
│  │   8081)    │            ▼│
│  └────────────┘    ┌──────────────┐
│                    │  openmemory  │
│                    │  (SQLite)    │
│                    └──────────────┘
└─────────────────────────────┘
      (fire-and-forget)
```

Two `llama-server` processes cohabit your GPUs: the large main model (likely row-split across all available cards via `llama-router`) and the tiny steward pinned to a single Vulkan device. The steward stays permanently resident so model-swap in the router doesn't evict it.

### Prerequisites

1. **openmemory MCP server configured** in `~/.config/opencode/opencode.json`:

   ```json
   "mcp": {
     "openmemory": {
       "type": "local",
       "command": ["npx", "-y", "openmemory-js", "mcp"],
       "environment": {
         "OM_EMBED_KIND": "synthetic",
         "OM_TIER": "hybrid",
         "OM_DB_PATH": "/home/you/.local/share/openmemory/openmemory.sqlite"
       }
     }
   }
   ```

   Ergon reads this block automatically via `client.config.get()` and applies the env vars to its own process before loading `openmemory-js`, so both the main model's MCP and the steward path write to the same SQLite file.

2. **`openmemory-js` installed** as an optional ergon dependency (this already happens automatically when you `npm install ergon-studio`). If the install fails — e.g. the `sqlite3` native build can't compile on your system — ergon still works, but the memory paths silently no-op.

3. **A steward `llama-server` process** running on `127.0.0.1:8081` serving a small instruct-tuned model. Qwen 3.5 2B is the recommended default — same prompt style as typical main coders, reliable JSON emission, fits in <3 GB VRAM at Q8.

### Single source of truth: `prompts/steward.md`

Everything about the steward — endpoint, model, temperature, GPU device, model file path, and both prompts — lives in **one file**: `prompts/steward.md`. That file is read by two different things that must stay in lockstep:

1. **`src/steward.ts`** — the ergon plugin's steward HTTP client. Reads the YAML frontmatter for URL/model/temperature defaults, and reads the `## rewrite` and `## judge` body sections for the two prompts.
2. **`scripts/run-steward.sh`** — the bash wrapper that actually launches `llama-server`. Parses the same YAML frontmatter to extract `port`, `model_path`, `device`, `llama_server_bin`, etc., then execs `llama-server` with the right args.

Both parse the same file, so editing `prompts/steward.md` changes the steward's service runtime *and* ergon's client-side config at the same time. No config duplication, nowhere to forget to update.

### Setting up the steward llama-server

1. **Edit `prompts/steward.md`** to match your paths. At minimum, update the `llama_server_bin`, `model_path`, and `device` fields in the frontmatter to point to your actual llama.cpp build, your downloaded Qwen 3.5 2B GGUF, and the GPU you want to dedicate to the steward.

   ```yaml
   ---
   port: 8081
   llama_server_bin: /home/aristath/llama.cpp/build-vulkan/bin/llama-server
   model_path: /home/aristath/models/qwen35-2b/qwen35-2b-instruct-Q8_0.gguf
   device: Vulkan1
   temperature: 0.3
   # ... rest of config + the prompts ...
   ---
   ```

2. **Install the systemd unit.** A template lives at `prompts/steward.service`. It assumes your ergon.studio checkout is at `/home/aristath/ergon.studio/` — if it's elsewhere, edit the `ExecStart` path before copying:

   ```bash
   cp prompts/steward.service ~/.config/systemd/user/llama-steward.service
   systemctl --user daemon-reload
   systemctl --user enable --now llama-steward.service
   systemctl --user status llama-steward.service
   ```

3. **Verify it's responding:**

   ```bash
   curl -s http://127.0.0.1:8081/v1/models
   ```

The unit does nothing other than run `scripts/run-steward.sh`, which is the file that actually parses your frontmatter and invokes `llama-server`. That means you can tune the model, device, or any inference flag by editing `prompts/steward.md` and running `systemctl --user restart llama-steward.service` — no `systemctl daemon-reload` needed, no touching the unit file.

Key choices baked into the launch command (all configurable via `steward.md`):

- **`temperature: 0.3`**, not zero. Judgment tasks need a little sampling slack; greedy decoding makes small models brittle on edge cases.
- **`device: Vulkan1`** pins the steward to one physical card so it doesn't fight the main model's tensor-parallel split across other cards.
- **`ctx_size: 16384`** — the steward never needs more. It only ever sees one exchange or one rewrite prompt at a time.
- **No `--models-preset`** — this instance has one job, permanently resident. It must **not** share `llama-router` with the main coder, or it would get evicted on every main-model swap.

### Configuration

All steward config lives in the frontmatter of `prompts/steward.md`:

```yaml
# Client config (read by src/steward.ts)
url: http://127.0.0.1:8081
model: ergon-studio-memory-steward
temperature: 0.3

# Service runtime config (read by scripts/run-steward.sh)
port: 8081
llama_server_bin: /home/aristath/llama.cpp/build-vulkan/bin/llama-server
model_path: /home/aristath/models/qwen35-2b/qwen35-2b-instruct-Q8_0.gguf
device: Vulkan1
n_gpu_layers: 99
ctx_size: 16384
top_k: 40
top_p: 0.95
```

These values become the defaults exported from `src/steward.ts` as `DEFAULT_STEWARD_URL`, `DEFAULT_STEWARD_MODEL`, and `DEFAULT_TEMPERATURE`. If you want to override them programmatically (e.g. for testing, or to run against a different steward instance), pass them via `createErgonPlugin`:

```typescript
import { createErgonPlugin, createStewardClient } from "ergon-studio"

export const ErgonPlugin = createErgonPlugin({
  steward: createStewardClient({
    baseURL: "http://127.0.0.1:9000",
    model: "my-custom-steward",
    temperature: 0.2,
  }),
})
```

The openmemory database path, embedding kind, and tier are picked up automatically from `opencode.json`'s `mcp.openmemory.environment` block via `client.config.get()`. You don't duplicate that config anywhere either.

### Prompts

The two prompts live as `## rewrite` and `## judge` sections in the body of `prompts/steward.md`. At module load, `src/steward.ts` reads the file, parses the frontmatter for config, splits the body on `## ` headings, and exposes each section as an exported constant:

- `REWRITE_PROMPT` — used by `rewriteQuery`. Tells the steward to strip politeness and filler, keep specifics, and emit `NONE` for greetings/acknowledgments.
- `JUDGE_PROMPT` — used by `judgeSave`. Tells the steward what counts as a durable fact worth saving.

**To tune the steward's behavior, edit `prompts/steward.md` directly.** No TypeScript changes required. Because the prompts and config are loaded from disk at plugin load time (not baked into the compiled JS), you don't even need to rebuild — just restart opencode (or the session) and the new prompt takes effect. For service-runtime changes (model path, device, etc.), `systemctl --user restart llama-steward.service` is enough.

- If recalls are missing specifics or over-stripping, tune the `## rewrite` section.
- If the steward is saving too aggressively (noise) or too conservatively (missing real preferences), tune the `## judge` section.

You can also pass fully custom prompts programmatically via `createStewardClient` options:

```typescript
createStewardClient({ rewritePrompt: "...", judgePrompt: "..." })
```

### Graceful degradation

Every external dependency in the steward pipeline has a silent fallback:

| Failure | What ergon does |
|---------|-----------------|
| `openmemory-js` not installed or fails to build | `getMemory()` returns `null`; recall and save no-op; other ergon features unaffected |
| Steward `llama-server` not running (connection refused) | `rewriteQuery` / `judgeSave` return `null`; turn proceeds without memory injection |
| Steward returns malformed JSON (save judgment) | Parsed as `null`, no save attempted |
| Steward returns `NONE` (rewrite) | Treated as "no searchable intent," recall skipped |
| `openmemory_query` returns `[]` or throws | No injection, turn proceeds normally |
| `openmemory_store` throws | Swallowed; save errors never surface to the user |
| `client.config.get()` fails | Fall back to whatever env vars are already set (or openmemory-js's built-in defaults) |

The steward should never block a turn, break a plugin load, or crash a session. If something's wrong, the worst case is "memory doesn't help this turn."

### Inspecting what the steward is doing

To see the steward actually firing, tail `llama-steward.service`:

```bash
journalctl --user -u llama-steward -f
```

Each request logs the prompt and generated text. You can verify both the rewrite output and the save judgments by eye as your conversation progresses.

To inspect what's actually in openmemory:

```bash
sqlite3 ~/.local/share/openmemory/openmemory.sqlite \
  "SELECT content, primary_sector, tags, datetime(created_at/1000, 'unixepoch') FROM memories ORDER BY created_at DESC LIMIT 20;"
```

### Testing

```bash
npm test
```

Runs three test suites:

- **`tests/steward.test.mjs`** — unit tests for the HTTP client (mocked fetch)
- **`tests/memory.test.mjs`** — unit tests for the openmemory wrapper (mocked `Memory` class)
- **`tests/plugin.test.mjs`** — integration tests for the `chat.message` and `session.idle` hooks (mocked steward + memory deps injected via `createErgonPlugin`)

None of the tests need `openmemory-js` installed or a running steward server — all external dependencies are mocked via dependency injection through the `createErgonPlugin({ steward, memory })` factory.

## Development

### Setup

```bash
npm install
```

### Build

```bash
npm run build
```

### Test

```bash
npm test
```

### Local Development

For local plugin development, add this to your `~/.config/opencode/opencode.json`:

```json
{
  "plugin": ["/path/to/ergon.studio"]
}
```

OpenCode will load plugins from local paths automatically.

## Plugin API

Ergon is an OpenCode plugin with the following surface:

| Export | Purpose |
|--------|---------|
| `ErgonPlugin` | Default `Plugin` export — use this in your `opencode.json`'s `plugin` array for out-of-the-box behavior |
| `createErgonPlugin(deps?)` | Factory that returns a `Plugin`. Lets callers inject custom `steward` or `memory` clients for testing or customization |
| `createStewardClient(opts?)` | Constructs a standalone steward HTTP client — useful if you want to call the steward from your own code or override prompts/URL/model |
| `createMemoryClient(opts)` | Constructs a standalone openmemory wrapper — takes an injected `Memory` class |
| `REWRITE_PROMPT`, `JUDGE_PROMPT` | The two steward prompts as exported constants |
| `DEFAULT_STEWARD_URL`, `DEFAULT_STEWARD_MODEL`, `DEFAULT_TEMPERATURE`, `DEFAULT_RECALL_LIMIT` | Default configuration values |

### Hooks implemented

| Hook | Purpose |
|------|---------|
| `event` (on `session.created`) | Logs session start |
| `event` (on `session.idle`) | Fires the memory steward's save path (fire-and-forget) |
| `chat.message` | Fires the memory steward's recall path before the main model sees the turn |
| `experimental.chat.system.transform` | Injects the project scratchpad into the system prompt |
| `experimental.session.compacting` | Re-injects the scratchpad when context is compacted so it survives long sessions |
| `tool.run_parallel` | Runs multiple specialist agents concurrently in child sessions |

### Minimal plugin skeleton

```typescript
import type { Plugin } from "@opencode-ai/plugin"
import { createErgonPlugin } from "ergon-studio"

// The default ErgonPlugin export is equivalent to this:
export const MyPlugin: Plugin = createErgonPlugin()
```

### Customizing the memory steward

```typescript
import { createErgonPlugin, createStewardClient, createMemoryClient } from "ergon-studio"

export const ErgonPlugin = createErgonPlugin({
  // Custom steward config (e.g. different port, model, prompts)
  steward: createStewardClient({
    baseURL: "http://127.0.0.1:9000",
    model: "my-custom-steward",
    temperature: 0.2,
    rewritePrompt: "...",
    judgePrompt: "...",
  }),
  // Custom memory backend — inject any class that implements
  // { add(content, meta?), search(query, opts?) }
  // memory: createMemoryClient({ Memory: MyCustomMemoryClass }),
})
```

### Under the hood: what `ErgonPlugin` does on session start

```typescript
event: async ({ event }) => {
  if (event.type === "session.created") {
    await client.app.log({
      body: {
        service: "ergon-plugin",
        level: "info",
        message: "Ergon session started",
      },
    })
  }
  if (event.type === "session.idle") {
    // Memory steward save path — async, off critical path
    // ...
  }
}
```

## Configuration

Agents are defined as Markdown files with YAML frontmatter in `agents/`:

```markdown
---
description: Lead developer who orchestrates the team for complex tasks
temperature: 0.7
mode: primary
---

## Identity
You are the lead dev. The user is your product manager. You two build things together.
...
```

## License

MIT
