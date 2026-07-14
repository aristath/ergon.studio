# Pi Packages for ergon.studio

Standalone Pi packages live under `pi/packages/`.

Each package has its own `package.json`, README, extension entrypoints,
dependencies, and release surface. The root `ergon.studio` package remains the
OpenCode plugin; Pi work stays separate here.

## Packages

| Package                                                       | Description                                                                                             | Runtime needs                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [`@ergon.studio/pi-brainstormer`](packages/brainstormer/)     | Freeform `/brainstorm` mode for exploratory thinking before planning                                    | none                                                          |
| [`@ergon.studio/pi-memory-steward`](packages/memory-steward/) | Cross-session memory with SQLite vector search, Granite embeddings, and a small steward LLM             | two local user services on reserved ports `18091` and `18092` |
| [`@ergon.studio/pi-orchestrator`](packages/orchestrator-mode/) | Legacy Ergon `/orchestrator` mode with bundled specialist agents and agent-owned quality gate           | none                                                          |
| [`@ergon.studio/pi-plan`](packages/plan-mode/)                | Scout-inspired `/plan` workflow for architecture planning and `.ergon.studio/HANDOFF.md` creation       | none                                                          |
| [`@ergon.studio/pi-scratchpad`](packages/scratchpad/)         | Injects and maintains `.ergon.studio/scratchpad.md` when a project opts in                              | none                                                          |

## Install Workspace Dependencies

```bash
cd pi
npm install
```

## Build

```bash
cd pi
npm run build
```

## Install Packages Into Pi

Local development install:

```bash
cd /path/to/ergon.studio
pi install ./pi/packages/brainstormer
pi install ./pi/packages/memory-steward
pi install ./pi/packages/orchestrator-mode
pi install ./pi/packages/plan-mode
pi install ./pi/packages/scratchpad
```

After publishing:

```bash
pi install npm:@ergon.studio/pi-brainstormer
pi install npm:@ergon.studio/pi-memory-steward
pi install npm:@ergon.studio/pi-orchestrator
pi install npm:@ergon.studio/pi-plan
pi install npm:@ergon.studio/pi-scratchpad
```

## Memory Steward Runtime Setup

Installing the Pi package registers the extension with Pi, but the memory
steward also needs its local runtime services.

From `pi/packages/memory-steward`:

```bash
./scripts/ergon-memory-steward setup --start --enable-linger
./scripts/ergon-memory-steward doctor
```

This provisions:

- `${XDG_DATA_HOME:-~/.local/share}/ergon-memory-steward/models/`
- `${XDG_DATA_HOME:-~/.local/share}/ergon-memory-steward/memory.sqlite`
- `${XDG_CONFIG_HOME:-~/.config}/ergon-memory-steward.env`
- `${XDG_CONFIG_HOME:-~/.config}/systemd/user/ergon-steward.service`
- `${XDG_CONFIG_HOME:-~/.config}/systemd/user/ergon-embedder.service`

The services are dedicated to this package:

- steward LLM: `127.0.0.1:18091`
- embedding server: `127.0.0.1:18092`

Do not reuse common service ports such as `8081` or `8082` for this package.
Those ports may belong to older or unrelated memory services.

## Boot Behavior

The memory-steward setup command can enable boot launch:

```bash
./scripts/ergon-memory-steward setup --start --enable-linger
```

Equivalent manual commands:

```bash
systemctl --user enable --now ergon-steward.service ergon-embedder.service
loginctl enable-linger "$USER"
```

Check:

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

## Package Development Rules

- Keep every Pi package self-contained under `pi/packages/<name>/`.
- Put extension entrypoints in `extensions/`.
- Put package internals in `src/`.
- Put user-facing runtime scripts in `scripts/`.
- Put prompts/config defaults in `prompts/`.
- Use package-local README files as the source of truth for setup and operation.
- Keep generated runtime state outside the package install directory.
- List Pi core imports in `peerDependencies` with `"*"`.
- Put third-party runtime dependencies in `dependencies`.
- Make extensions degrade gracefully; a package must not crash the Pi session if
  an optional runtime service is down.

## Release Checks

Before publishing a package:

```bash
cd pi
npm test
npm run build

cd packages/brainstormer
npm pack --dry-run

cd packages/memory-steward
npm pack --dry-run

cd ../orchestrator-mode
npm pack --dry-run

cd ../plan-mode
npm pack --dry-run

cd ../scratchpad
npm pack --dry-run
```

For memory-steward, also run:

```bash
cd pi/packages/memory-steward
./scripts/ergon-memory-steward doctor
```

## Adding A New Package

Create:

```text
pi/packages/<name>/
├── package.json
├── README.md
├── tsconfig.json
├── extensions/
└── src/
```

Add package metadata:

```json
{
  "name": "@ergon.studio/pi-<name>",
  "type": "module",
  "main": "extensions/index.ts",
  "pi": {
    "extensions": ["./extensions"]
  },
  "peerDependencies": {
    "@earendil-works/pi-coding-agent": "*"
  }
}
```

If the package owns services or persistent runtime state, also add:

```text
scripts/
prompts/
```

and document:

- install
- setup
- boot behavior
- generated files
- config variables
- verification commands
- troubleshooting
