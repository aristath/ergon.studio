# @ergon.studio/pi-memory-steward

Self-contained Pi extension for cross-session memory.

It adds a bounded memory recall/save loop to Pi using:

- a small steward LLM on `127.0.0.1:18091`
- a dedicated embedding server on `127.0.0.1:18092`
- a local SQLite database with `sqlite-vec`
- Pi lifecycle hooks for recall before the agent starts and save after turns end

The package is designed to be installable as a standalone Pi module. It does not
depend on the root `ergon.studio` OpenCode plugin or root repo service scripts.

## Quick Start

From this package directory:

```bash
npm install
./scripts/ergon-memory-steward setup --start --enable-linger
./scripts/ergon-memory-steward doctor
```

Then install the Pi package:

```bash
pi install .
```

Expected doctor result:

```text
ok: env file exists
ok: llama-server executable
ok: steward model exists
ok: embedder model exists
ok: memory DB directory exists
ok: steward service active
ok: embedder service active
ok: steward models endpoint
ok: embedder models endpoint
ok: embedding dimensions
memory steward doctor passed
```

In Pi, the status should become:

```text
✓ memory active
```

## What It Does

The extension has two paths.

Recall path:

1. Pi emits `before_agent_start`.
2. The steward rewrites the user prompt into a short search query.
3. The embedder converts that query into a vector.
4. SQLite + `sqlite-vec` finds similar saved memories.
5. Relevant memories are appended to the system prompt.

Save path:

1. Pi emits `turn_end` after each model response.
2. The extension ignores tool-use and incomplete responses.
3. The steward judges whether each completed user/assistant exchange is durable.
4. If worth saving, the embedder creates a vector for the memory.
5. The memory is stored in SQLite with content-hash deduplication.

The recall path is synchronous and bounded to 5 seconds per external stage. The
steward rewrite and embedding request each receive that full budget, so the
complete recall path can take about 10 seconds in the worst case. The save path
is async and best-effort, with shutdown draining so final-turn saves are not
casually dropped.

The two llama.cpp services are supervised by package-local wrapper scripts. The
wrappers start `llama-server`, probe the actual HTTP contract, and exit if the
process is alive but unhealthy. Systemd then restarts the service. This protects
against GPU/backend failures where the process remains active but embeddings or
model APIs return errors.

## Runtime Pieces

| Piece           | Port/file                                                             | Purpose                         |
| --------------- | --------------------------------------------------------------------- | ------------------------------- |
| Pi extension    | `extensions/index.ts`                                                 | Hooks into Pi session lifecycle |
| Steward server  | `127.0.0.1:18091`                                                     | Query rewrite and save judgment |
| Embedder server | `127.0.0.1:18092`                                                     | 768-dimensional embeddings      |
| Memory DB       | `${XDG_DATA_HOME:-~/.local/share}/ergon-memory-steward/memory.sqlite` | SQLite memory store             |
| Env file        | `${XDG_CONFIG_HOME:-~/.config}/ergon-memory-steward.env`              | Local machine config            |
| Service units   | `${XDG_CONFIG_HOME:-~/.config}/systemd/user/ergon-*.service`          | Boot/runtime supervision        |

Reserved ports are intentionally high and package-specific. Do not repoint this
module at common ports such as `8081` or `8082`; those may be used by older or
unrelated services.

## Self-Contained Layout

After `setup`, runtime state lives outside the package install directory:

```text
${XDG_DATA_HOME:-~/.local/share}/ergon-memory-steward/
├── memory.sqlite
└── models/
    ├── Qwen3.5-4B-UD-Q8_K_XL.gguf
    └── granite-embedding-311m-multilingual-r2-Q4_K_M.gguf
```

Generated local config lives here:

```text
${XDG_CONFIG_HOME:-~/.config}/ergon-memory-steward.env
```

Installed user services live here:

```text
${XDG_CONFIG_HOME:-~/.config}/systemd/user/ergon-steward.service
${XDG_CONFIG_HOME:-~/.config}/systemd/user/ergon-embedder.service
```

This split matters:

- package code can be upgraded or reinstalled
- models and memory DB persist across upgrades
- systemd has stable absolute paths
- machine-specific paths stay out of package source
- non-default XDG config/data directories are honored consistently

## Commands

The package command is:

```bash
ergon-memory-steward <command>
```

When developing from the repo, use:

```bash
./scripts/ergon-memory-steward <command>
```

Commands:

| Command                 | What it does                                               |
| ----------------------- | ---------------------------------------------------------- |
| `setup`                 | Generates env config, provisions models, installs services |
| `setup --start`         | Runs setup, enables services, restarts services            |
| `setup --enable-linger` | Enables user services to start at boot before login        |
| `doctor`                | Verifies files, services, ports, endpoints, embeddings     |
| `start`                 | Starts both user services                                  |
| `stop`                  | Stops both user services                                   |
| `restart`               | Restarts both user services                                |
| `status`                | Shows both user service statuses                           |
| `logs`                  | Follows both service logs                                  |
| `env`                   | Prints generated env config                                |
| `install-services`      | Reinstalls service units using existing env config         |

Setup options:

```bash
./scripts/ergon-memory-steward setup \
  --llama-server /path/to/llama-server \
  --steward-model /path/to/steward.gguf \
  --embedder-model /path/to/embedder.gguf \
  --data-dir ~/.local/share/ergon-memory-steward \
  --db-path ~/.local/share/ergon-memory-steward/memory.sqlite \
  --start \
  --enable-linger
```

Model provisioning hardlinks model files into the module data directory when the
filesystem supports it. If hardlinking is not possible, setup copies them.

## Boot Behavior

The services are normal systemd user services:

```bash
systemctl --user enable --now ergon-steward.service ergon-embedder.service
```

To launch them at boot before an interactive login, enable lingering:

```bash
loginctl enable-linger "$USER"
```

`setup --start --enable-linger` does both.

Check boot readiness:

```bash
systemctl --user is-enabled ergon-steward.service ergon-embedder.service
loginctl show-user "$USER" -p Linger
```

Expected:

```text
enabled
enabled
Linger=yes
```

## Configuration

Runtime config is read in this order:

1. Environment variables from `${XDG_CONFIG_HOME:-~/.config}/ergon-memory-steward.env`
2. Frontmatter defaults in `prompts/steward.md`
3. hardcoded safe defaults

Important env values:

| Variable                        | Purpose                                                  |
| ------------------------------- | -------------------------------------------------------- |
| `ERGON_MEMORY_STEWARD_DIR`      | Package directory used by service units                  |
| `ERGON_LLAMA_SERVER_BIN`        | `llama-server` executable                                |
| `ERGON_STEWARD_URL`             | Steward endpoint, usually `http://127.0.0.1:18091`       |
| `ERGON_STEWARD_PORT`            | Steward port, usually `18091`                            |
| `ERGON_STEWARD_MODEL_PATH`      | Steward GGUF path                                        |
| `ERGON_STEWARD_MODEL`           | OpenAI-compatible model alias                            |
| `ERGON_STEWARD_DEVICE`          | llama.cpp device selector                                |
| `ERGON_STEWARD_N_GPU_LAYERS`    | Steward GPU layer count                                  |
| `ERGON_STEWARD_CTX_SIZE`        | Steward context size                                     |
| `ERGON_STEWARD_TEMPERATURE`     | Steward generation temperature                           |
| `ERGON_STEWARD_TOP_K`           | Steward top-k                                            |
| `ERGON_STEWARD_TOP_P`           | Steward top-p                                            |
| `ERGON_STEWARD_ENABLE_THINKING` | `on`, `off`, or `auto`                                   |
| `ERGON_EMBEDDER_URL`            | Embedder endpoint, usually `http://127.0.0.1:18092`      |
| `ERGON_EMBEDDER_PORT`           | Embedder port, usually `18092`                           |
| `ERGON_EMBEDDER_MODEL_PATH`     | Embedding GGUF path                                      |
| `ERGON_EMBEDDER_MODEL`          | Embedding model alias                                    |
| `ERGON_EMBEDDER_DIMENSIONS`     | Vector dimension, usually `768`                          |
| `ERGON_EMBEDDER_CTX_SIZE`       | Embedder context size                                    |
| `ERGON_EMBEDDER_N_GPU_LAYERS`   | Embedder GPU layer count                                 |
| `ERGON_MEMORY_DB_PATH`          | SQLite DB path                                           |
| `ERGON_RECALL_LIMIT`            | Number of memories injected per recall                   |
| `ERGON_LLAMA_HEALTH_INTERVAL`   | Seconds between steady-state probes, default `30`        |
| `ERGON_LLAMA_START_PERIOD`      | Seconds allowed for initial healthy probe, default `120` |
| `ERGON_LLAMA_HEALTH_FAILURES`   | Consecutive failed probes before restart, default `2`    |

Do not edit service unit files for normal tuning. Edit or regenerate the env
file, then restart:

```bash
./scripts/ergon-memory-steward restart
```

`setup` is intentionally idempotent. On rerun it first reads the generated env
file, reuses provisioned model paths, and rewrites the user service units with
the currently resolved env-file path. Passing `--data-dir` intentionally moves
managed model and DB targets to the new data directory unless explicit
`--steward-model`, `--embedder-model`, or `--db-path` values are supplied.

## Development Checks

From this package directory:

```bash
npm test
npm pack --dry-run
```

From the Pi workspace root:

```bash
cd pi
npm test
npm run build
```

`npm test` builds the package and runs runtime-focused tests for:

- SQLite memory store persistence, deduplication, recall, delete, and dimension mismatch behavior
- steward prompt parsing, query rewriting, save judgment parsing, and thinking-tag stripping
- embedder vector truncation, padding, and batch behavior
- setup idempotency and generated systemd unit paths
- service supervision for healthy and failed startup probes

## Package Defaults

`prompts/steward.md` contains portable defaults:

```yaml
url: http://127.0.0.1:18091
port: 18091
llama_server_bin: llama-server
model_path: ~/.local/share/ergon-memory-steward/models/Qwen3.5-4B-UD-Q8_K_XL.gguf
embedder_url: http://127.0.0.1:18092
embedder_model: granite-embedding-311m
embedder_model_path: ~/.local/share/ergon-memory-steward/models/granite-embedding-311m-multilingual-r2-Q4_K_M.gguf
memory_db_path: ~/.local/share/ergon-memory-steward/memory.sqlite
recall_limit: 5
```

The generated env file usually overrides the model paths with the exact files
that setup provisioned.

## Services

`ergon-steward.service`:

- reads `~/.config/ergon-memory-steward.env`
- runs `scripts/run-steward.sh`
- starts `llama-server` on `18091`
- serves `/v1/models` and `/v1/chat/completions`

`ergon-embedder.service`:

- reads `~/.config/ergon-memory-steward.env`
- runs `scripts/run-embedder.sh`
- starts `llama-server` on `18092`
- serves `/v1/models` and `/v1/embeddings`

Both services are restarted on failure.

The wrapper supervision matters because some backend failures do not terminate
`llama-server`; for example, a GPU device loss can leave the process running
while `/v1/embeddings` returns HTTP 500. The embedder wrapper probes
`/v1/embeddings` and verifies the configured vector dimension. The steward
wrapper probes `/v1/models`. If either check fails repeatedly, the wrapper exits
non-zero and systemd restarts the service.

## Pi Extension Lifecycle

The extension registers handlers at package load time but does not open the DB
or probe network services from the factory. Runtime initialization happens on
`session_start` and first use.

Hooks:

| Hook                 | Behavior                                             |
| -------------------- | ---------------------------------------------------- |
| `session_start`      | initialize clients/store, check health, set status   |
| `before_agent_start` | recall memories and inject a system prompt block     |
| `turn_end`           | judge/save each completed response in the background |
| `session_shutdown`   | drain pending saves, close DB, clear status          |

Status values:

| Status                             | Meaning                              |
| ---------------------------------- | ------------------------------------ |
| `✓ memory active`                  | Steward, embedder, and DB are usable |
| `⚠ memory: steward unreachable`    | `18091` is not answering             |
| `⚠ memory: embedder unreachable`   | `18092` is not answering             |
| `⚠ memory: embedder model missing` | configured embedder GGUF is absent   |
| `⚠ memory: memory db unavailable`  | SQLite store could not open          |

## SQLite Store

The DB stores memory content, metadata, FTS rows, and vectors.

Public store contract:

```typescript
interface MemoryStore {
	recall(vector: number[], k?: number, queryText?: string): MemoryItem[];
	save(content: string, vector: number[]): void;
	delete(id: string): void;
	list(limit?: number): MemoryItem[];
	close(): void;
	isAvailable(): boolean;
}
```

Vector dimension is configured by `ERGON_EMBEDDER_DIMENSIONS` or
`embedder_dimensions`. Existing DBs keep their original dimension; if the config
changes from `768` to something else, create a new DB or re-embed the memories.

## Verification

Check services:

```bash
./scripts/ergon-memory-steward status
```

Check ports:

```bash
ss -ltnp | rg ':(18091|18092)\b'
```

Check steward:

```bash
curl -sS http://127.0.0.1:18091/v1/models
```

Check embedder dimensions:

```bash
curl -sS http://127.0.0.1:18092/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"granite-embedding-311m","input":"doctor"}'
```

Or run the full check:

```bash
./scripts/ergon-memory-steward doctor
```

## Troubleshooting

Steward unreachable:

```bash
./scripts/ergon-memory-steward status
journalctl --user -u ergon-steward.service -n 80 --no-pager
```

Embedder unreachable:

```bash
./scripts/ergon-memory-steward status
journalctl --user -u ergon-embedder.service -n 80 --no-pager
```

Model missing:

```bash
./scripts/ergon-memory-steward setup --start
```

Port already in use:

```bash
ss -ltnp | rg ':(18091|18092)\b'
```

If another process uses either reserved port, stop that process or change both
the env URL and port values together.

Services do not start after reboot:

```bash
systemctl --user is-enabled ergon-steward.service ergon-embedder.service
loginctl show-user "$USER" -p Linger
```

If linger is not enabled:

```bash
loginctl enable-linger "$USER"
```

DB dimension mismatch:

- keep `ERGON_EMBEDDER_DIMENSIONS=768`, or
- move the old DB aside and let the module create a fresh DB, or
- re-embed existing memories into a new DB

## Development

Build all Pi packages from `pi/`:

```bash
npm run build
```

Package dry-run:

```bash
npm pack --dry-run
```

Install locally into Pi:

```bash
pi install ./pi/packages/memory-steward
```

The package intentionally ships TypeScript extension sources. Pi loads
extensions through `jiti`, so the package does not need to publish compiled
`dist/` output.

## Files In This Package

```text
extensions/index.ts
  Pi lifecycle integration.

src/steward.ts
  Steward prompt/config parser and HTTP client.

src/embedder.ts
  Embedding HTTP client and dimension handling.

src/memory-store.ts
  SQLite + sqlite-vec store.

prompts/steward.md
  Prompt text plus portable defaults.

scripts/ergon-memory-steward
  Setup, doctor, service operations.

scripts/run-steward.sh
  Starts the dedicated steward llama-server.

scripts/run-embedder.sh
  Starts the dedicated embedding llama-server.

scripts/ergon-steward.service
scripts/ergon-embedder.service
  User systemd units installed by setup.
```
