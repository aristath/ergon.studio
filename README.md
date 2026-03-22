# ergon.studio

`ergon` is an orchestration proxy for local LLMs.

It sits between your coding client (IDE, chat UI, terminal tool) and a local
model endpoint. The client talks to ergon like it would talk to any
OpenAI-compatible model. Behind the scenes, ergon coordinates a team of AI
agents to produce better results than a single model pass.

## Why

Local models produce mediocre output on one pass. But if you make the same
model plan before it codes, review after it codes, and iterate on the
feedback — the results get dramatically better.

That's what ergon does. It adds the behavior a good lead developer adds:
break the problem down, bring in the right people, inspect results critically,
iterate on weak spots, and decide when the work is ready to ship.

## How It Works

You talk to the orchestrator. The orchestrator is the lead dev — it
understands your goal, decides what kind of help is needed, and coordinates
the team.

- For simple tasks, the orchestrator handles them directly.
- For bigger work, it opens channels — one-on-one calls or conference calls
  where specialists (architect, coder, reviewer, tester, critic, researcher)
  collaborate in natural language.
- After each conversation, the orchestrator reads the results and decides
  what happens next: iterate, change approach, bring in someone else, or
  deliver.

The orchestrator stays in control throughout. There's no rigid pipeline — just
judgment, delegation, and iteration.

Your client keeps everything it already owns: the UI, sessions, tool
execution, MCP integrations, approvals, and diffs. Ergon just makes the model
smarter.

## Quick Start

### Requirements

- Python 3.12+
- A local OpenAI-compatible model endpoint (e.g., llama.cpp, vLLM, Ollama)

### Install

```bash
pip install .
```

For the configuration TUI:

```bash
pip install '.[tui]'
```

### Run

Default mode launches the configuration TUI and the proxy server together:

```bash
ergon
```

Headless mode runs just the server:

```bash
ergon --serve
```

### Connect Your Client

Point your coding client at the proxy endpoint (default: `http://127.0.0.1:4000/v1`).
Ergon exposes a standard `/v1/chat/completions` endpoint — any OpenAI-compatible
client will work.

## Configuration

### Workspace

The first launch creates a workspace at `~/.config/ergon/` containing:

- `config.json` — upstream endpoint, proxy host/port
- `definitions/agents/*.md` — agent role definitions
- `definitions/channels/*.md` — channel presets

### CLI Options

```
--serve                     Run headless (no TUI)
--app-dir PATH              Custom workspace location
--definitions-dir PATH      Custom definitions location
--upstream-base-url URL     LLM endpoint (e.g., http://localhost:8080/v1)
--upstream-api-key KEY      API key (can be left blank for local models)
--host HOST                 Proxy bind address (default: 127.0.0.1)
--port PORT                 Proxy bind port (default: 4000)
```

### TUI

The configuration TUI has tabs for:

- Upstream endpoint settings
- Agent definitions
- Channel presets

Navigation: `Tab`/`Shift+Tab` to move focus, arrow keys inside lists and
editors.

## Agents

Ergon ships with seven default agents:

| Agent | Role |
|-------|------|
| `orchestrator` | Lead developer — talks to the user, coordinates the team |
| `architect` | Plans before anyone builds, thinks ten steps ahead |
| `coder` | Takes a brief and produces working code |
| `reviewer` | Quality gate — checks correctness and adherence to the brief |
| `tester` | Produces evidence by actually running things |
| `critic` | Challenges assumptions and finds what a friendly team would miss |
| `researcher` | Digs into the codebase and gathers context before decisions |

Agents are defined as markdown files with YAML frontmatter. You can edit the
defaults or add your own — a designer, security auditor, documentation writer,
or anything else that fits your workflow.

## Channels

Channels are collaborative conversations where agents work together. The
orchestrator opens them as needed.

Ergon ships with two presets:

- **best-of-n** — Three coders tackle the same problem independently. The
  orchestrator compares and picks the best approach.
- **debate** — Architect, coder, critic, and reviewer discuss a problem from
  different angles before committing to a plan.

The orchestrator can also open ad-hoc channels with any combination of
agents. Presets are shortcuts, not constraints.

## Development

Install dev dependencies:

```bash
pip install -e '.[dev]'
```

Run checks:

```bash
./scripts/check
```

Individual commands:

- `./scripts/format` — auto-format
- `ruff check .` — lint
- `mypy` — type check
- `python -m pytest tests/` — unit tests
- `./scripts/check-real-e2e` — real model smoke tests

### Real Model E2E

Smoke tests against a real upstream live in `tests/real_proxy_e2e.py`. They
read from `.env.e2e-tests`:

```bash
UPSTREAM_BASE_URL=http://localhost:8080/v1
MODEL=qwen3-coder-next-q6k
```

If the upstream is unavailable, these tests skip cleanly.

## License

See [LICENSE](LICENSE).
