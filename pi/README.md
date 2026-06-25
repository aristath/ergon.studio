# Pi Extensions for ergon.studio

This directory can host multiple standalone Pi extensions.
Each extension should live in its own folder with its own `package.json`.

Available extensions:
- `agent-switch` — adds `/agent` command, interactive agent list, and active-agent UI state

Naming convention for future releases: use `@ergon.studio/*` package names.

Example install (local):

```bash
pi install ./pi/agent-switch
```

More extensions can be added under `pi/` as independent packages.
