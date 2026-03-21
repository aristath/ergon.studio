# ergon.studio

`ergon.studio` is an OpenAI-compatible orchestration proxy for local coding models.

It sits in front of an existing host client and turns one model request into a coordinated multi-agent turn:

- the orchestrator acts as the lead developer and coordinates the team
- specialists handle focused sub-work
- host-provided tools are passed through unchanged
- the proxy streams orchestration worklog plus final assistant output

The host keeps the UI, sessions, tools, MCPs, and approvals.

`ergon.studio` keeps the orchestration brain.

The proxy uses:
- one upstream OpenAI-compatible endpoint for all internal orchestration turns
- markdown-defined agents
- markdown-defined workrooms

## Running ergon

Install the proxy in the mode you want:

```bash
pip install .
```

For the configuration TUI, install the `tui` extra:

```bash
pip install '.[tui]'
```

Default mode launches the configuration TUI and the local proxy server together:

```bash
ergon
```

Headless mode runs just the server:

```bash
ergon --serve
```

If `textual` is not installed, `ergon` will tell you that the configuration TUI
requires the `tui` extra. `ergon --serve` does not need it.

The configuration TUI has separate tabs for:
- upstream endpoint settings
- agent definitions
- workroom definitions

The UI uses standard navigation:
- `Tab` / `Shift+Tab` to move focus
- arrow keys inside lists, tabs, and editors

## Local workspace

The first TUI launch creates a local workspace under:

```text
~/.config/ergon/
```

That workspace contains:
- `config.json`
- `definitions/agents/*.md`
- `definitions/workrooms/*.md`

The upstream API key may be left blank. In that case, ergon uses `not-needed` for the upstream client.

You can override the defaults if needed:
- `--app-dir /path/to/workspace`
- `--definitions-dir /path/to/definitions`
- `--upstream-base-url http://localhost:8080/v1`
- `--upstream-api-key your-key`
- `--host 127.0.0.1`
- `--port 4000`

## Development checks

Install the dev dependencies in the project virtualenv:

```bash
.venv/bin/pip install -e '.[dev]'
```

Run the quality checks:

```bash
./scripts/check
```

Available commands:
- `./scripts/format`
- `.venv/bin/ruff check .`
- `.venv/bin/mypy`
- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- `./scripts/check-real-e2e`

## Real model E2E

Real upstream smoke tests live in `tests/real_proxy_e2e.py`.

They read their target from `.env.e2e-tests`:

```bash
UPSTREAM_BASE_URL=http://localhost:8080/v1
MODEL=qwen3-coder-next-q6k
```

Run them with:

```bash
./scripts/check-real-e2e
```

If the local upstream is unavailable, the real E2E module skips cleanly.
