# @ergon.studio/dsh

Ergon multi-agent orchestration for [DeepSeek Harness](https://www.npmjs.com/package/@deepseek-ai/dsh) (DSH).

One command turns a DSH profile into the full Ergon experience: a 10-agent
team preset, automatic memory recall and saving through a local 4B memory
steward, a persistent scratchpad, and two multi-agent tools (`debate`,
`run_parallel`).

This is a port of the [ergon.studio](https://ergon.studio) OpenCode plugin to
the DSH plugin system. Design notes and the pinned harness contracts live in
[contracts.md](./contracts.md).

## What you get

| Piece | How it works in DSH |
| --- | --- |
| **10-agent team** | The `ergon` agent preset: orchestrator as the session persona + 9 specialist tools (`scout`, `architect`, `coder`, `researcher`, `reviewer`, `design_reviewer`, `critic`, `tester`, `quality_controller`) via `dsh-tool-subagent` rows, each with its persona and deny-based tool filter. |
| **`debate` tool** | Two agents alternate on a task until one agrees, blocks, or max turns. |
| **`run_parallel` tool** | Up to 10 specialists run concurrently (hard cap per call); results are combined per agent (fail-open per task). |
| **Memory recall** | On each user message in a top-level session: local 4B steward rewrites the query → openmemory search → results surfaced as a dynamic prompt context (append-only snapshot, KV-cache friendly). A pre-step fallback re-triggers recall if a user message reaches the step unrecalled; subagent turns never recall. |
| **Memory save** | On completed top-level turns: the steward judges the exchange and stores distilled notes in openmemory. Subagent turns never save (no cost amplification, no corpus noise). |
| **Scratchpad / handoff** | `.ergon.studio/scratchpad.md` and `HANDOFF.md` re-read on every prompt assembly (survives compaction) + the matching skills. |
| **`memory_search` tool** | Explicit semantic search over the memory corpus. |

The plugin mounts at **profile level** (a DSH *bundle*: the package declares
`dsh.bundle`, and `dsh plugin add` registers it as a profile layer), so every
session in that profile gets the tools and the memory/scratchpad contexts.
The preset itself only carries agent configuration.

All external dependencies are **fail-open**: a dead steward or memory service
degrades to "no memory" — never to a broken session.

## Requirements

- DSH ≥ 0.1.1 (pre-1.0; contracts pinned in [contracts.md](./contracts.md))
- Node ≥ 20, `pnpm` on PATH (used by `dsh plugin`)
- For the memory features (optional — everything else works without):
  - `ergon-steward.service` — a local vLLM instance serving a ~4B
    "memory steward" model (default `http://127.0.0.1:18091`)
  - an [openmemory](https://github.com/alexandros-lab/openmemory) store
    (default `http://127.0.0.1:8082`)

## Install

```sh
# from the npm registry (published):
dsh plugin --profile web add @ergon.studio/dsh
```

That's the whole install. On the next profile boot:

1. the bundle layer mounts the `ergon` plugin and sets the session **default
   agent preset to `ergon`** (override per session via the preset picker, or
   globally in the profile's own `cordis.patch.yml` — the user layer wins);
2. the plugin **self-installs** its agent preset to
   `~/.dsh/.agent-presets/ergon/` and its skills to `~/.dsh/skills/`
   (install-only-when-missing; never overwrites a copy you have edited —
   refresh one with `init --force`);
3. new sessions start in the Ergon team.

Existing user presets are left alone; an older preset version still carrying
the now-forbidden `ergon-plugin` row is detected by the CLI validator
(`init`/`status`) with instructions to regenerate.

To install from a local checkout before publishing:

```sh
cd ergon.studio/dsh && npm run build && npm pack
dsh plugin --profile web add ./ergon.studio-dsh-0.1.2.tgz
```

## Configuration

Row config (profile `cordis.patch.yml`, applied after the bundle layer):

```yaml
- id: ergon-plugin
  config:
    stewardUrl: http://127.0.0.1:18091   # optional
    stewardModel: ergon-studio-memory-steward  # optional
    memoryUrl: http://127.0.0.1:8082     # optional
    recallTimeoutMs: 5000                # default
    recallLimit: 5                       # default
```

Precedence: row config → environment → `prompts/steward.md` (shipped
definition) → built-in defaults:

| Setting | Env var | Default |
| --- | --- | --- |
| `stewardUrl` | `ERGON_STEWARD_URL` | `http://127.0.0.1:18091` |
| `stewardModel` | `ERGON_STEWARD_MODEL` | `ergon-studio-memory-steward` |
| `memoryUrl` | `ERGON_MEMORY_URL` | `http://127.0.0.1:8082` |

## CLI

```sh
npx @ergon.studio/dsh init [--profile web] [--force]  # (re)install preset + skills, validate, check profile
npx @ergon.studio/dsh status                           # show install state
```

## Uninstall

```sh
dsh plugin --profile web remove @ergon.studio/dsh
rm -rf ~/.dsh/.agent-presets/ergon ~/.dsh/skills/handoff ~/.dsh/skills/scratchpad
```

Restart the profile (removing the bundle restores the `standard` default
preset).

## Development

```sh
npm run build   # tsc + regenerate presets/ergon from agents/*.md
npm test        # build + node --test (102 tests against dist/)
```

- `agents/*.md` — the 10 agent definitions (roster source of truth); any
  extra file fails the preset build (it would silently become a spawnable agent)
- `src/plugin.ts` — the cordis plugin (tools, recall, save, scratchpad)
- `src/preset-gen.ts` — preset generator (validated against the shipped
  `standard` preset)
- `presets/ergon/` — generated, checked in
- `cordis.patch.yml` — the bundle layer (plugin row + default preset)
- `contracts.md` — pinned DSH contracts; **re-verify on harness upgrade**

## License

MIT
