# ergon.studio

`ergon.studio` is an OpenAI-compatible orchestration proxy for local coding models.

It sits in front of an existing host client and turns one model request into an orchestrated multi-agent turn:

- the orchestrator plans the turn
- specialists handle focused sub-work
- host-provided tools are passed through unchanged
- the proxy streams orchestration worklog plus final assistant output

The host keeps the UI, sessions, tools, MCPs, and approvals.

`ergon.studio` keeps the orchestration brain.

The proxy uses:
- one upstream OpenAI-compatible endpoint for all internal orchestration turns
- markdown-defined agents
- markdown-defined workflows

It does not create local project state on startup.
Point it at a definitions directory containing:
- `agents/orchestrator.md`
- `workflows/*.md`

Required startup inputs:
- `--definitions-dir` or `ERGON_DEFINITIONS_DIR`
- `--upstream-base-url` or `ERGON_UPSTREAM_BASE_URL`

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
