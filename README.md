# ergon.studio

`ergon` is an OpenAI-compatible orchestration proxy for local LLMs.

It sits between your coding client (IDE, chat UI, terminal tool) and a local
model endpoint. The client talks to ergon like it would talk to any
OpenAI-compatible model. Behind the scenes, ergon coordinates a team of AI
agents — a lead developer who talks to you, and specialists it brings in as
needed.

## Why

Local models produce mediocre output on one pass. But if you make the same
model plan before it codes, review after it codes, and iterate on the
feedback — the results get dramatically better.

That's what ergon does. It adds the behavior a good lead developer adds:
break the problem down, bring in the right people, inspect results critically,
iterate on weak spots, and decide when the work is ready to ship.

## How It Works

You talk to the **orchestrator** — the lead dev. It understands your goal,
decides what kind of help is needed, and coordinates a team of specialists.

For simple tasks the orchestrator answers directly. For bigger work, it has
three tools:

- **`open_channel`** — start a conversation with one or more specialists
  (architect, coder, reviewer, tester, critic, researcher). The conversation
  flows in natural language; the orchestrator can follow up and redirect.
- **`run_parallel`** — run N independent code-generation sessions in parallel
  and get all results back at once. Good for best-of-N implementations.
- **`message_channel` / `close_channel`** — continue or end an open channel.

The orchestrator stays in control throughout. There's no rigid pipeline — just
judgment, delegation, and iteration. After each specialist delivers, the
orchestrator reads the result and decides what comes next.

Your client keeps everything it already owns: the UI, sessions, tool
execution, MCP integrations, approvals, and diffs. Ergon makes the model
smarter without touching any of that.

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

Point your coding client at the proxy endpoint (default:
`http://127.0.0.1:4000/v1`). Ergon exposes standard
`/v1/chat/completions` and `/v1/responses` endpoints — any
OpenAI-compatible client works.

## Configuration

### CLI Options

```
--serve                     Run headless (no TUI)
--app-dir PATH              Workspace location (default: ~/.config/ergon)
--definitions-dir PATH      Custom definitions location
--upstream-base-url URL     LLM endpoint (e.g., http://localhost:8080/v1)
--upstream-api-key KEY      API key (leave blank for local models)
--host HOST                 Proxy bind address (default: 127.0.0.1)
--port PORT                 Proxy bind port (default: 4000)
--instruction-role ROLE     Role for system messages (default: system)
--disable-tool-calling      Disable tool calling in agent invocations
--log                       Write debug log to ~/.ergon.studio/debug.log
```

### Workspace

On first launch, ergon creates a workspace at `~/.config/ergon/` containing:

```
~/.config/ergon/
├── config.json                    # upstream endpoint, proxy host/port
└── definitions/
    ├── agents/
    │   ├── orchestrator.md        # required
    │   ├── coder.md
    │   └── ...
    └── channels/
        ├── best-of-n.md
        └── debate.md
```

`config.json` stores:

```json
{
  "upstream_base_url": "http://localhost:8080/v1",
  "upstream_api_key": "",
  "host": "127.0.0.1",
  "port": 4000,
  "instruction_role": "system",
  "disable_tool_calling": false
}
```

### TUI

The configuration TUI (requires `pip install '.[tui]'`) has tabs for:

- **Upstream** — endpoint URL, API key, instruction role, tool calling toggle
- **Agents** — view, create, edit, and delete agent definitions
- **Channels** — view, create, edit, and delete channel presets

Navigation: `Tab`/`Shift+Tab` to move focus, arrow keys inside lists and
editors.

## Agents

Ergon ships with seven default agents:

| Agent | Role | Temperature |
|-------|------|-------------|
| `orchestrator` | Lead developer — talks to the user, coordinates the team | 0.7 |
| `architect` | Plans and designs before anyone builds | 0.5 |
| `coder` | Takes a brief and produces working code | 0.2 |
| `reviewer` | Quality gate — checks correctness and adherence to the brief | 0.2 |
| `tester` | Produces evidence — actually runs things and reports what it finds | 0.1 |
| `critic` | Challenges assumptions and finds what a friendly team would miss | 0.6 |
| `researcher` | Digs into the codebase and gathers context before decisions | 0.3 |

### Definition Format

Agents are defined as Markdown files with YAML frontmatter:

```markdown
---
id: coder
role: coder
temperature: 0.2
max_tokens: 8192
---

## Identity
You are the coder. ...

## How You Work
...

## Subsession
Your workspace for this task is {workspace}.
Use read_file, list_files, and write_file with absolute paths.
When you're done, reply with your findings or a summary — no tool calls in that final reply.
```

**Frontmatter fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Agent identifier (alphanumeric, hyphens, underscores) |
| `role` | No | Role label |
| `temperature` | No | Model temperature (0.0–1.0) |
| `max_tokens` | No | Maximum tokens for this agent's responses |

**Body sections** (`## SectionName` headers):

The body is plain Markdown. Sections are extracted by `## Header` and have
different semantics:

- **Static sections** (e.g., `## Identity`, `## How You Work`) — built into
  the agent's system message at invocation time. Write the agent's permanent
  character and operating principles here.

- **`## Orchestration`** — injected at runtime into the orchestrator's second
  system message with `{open_channels_section}` substituted. Only meaningful
  for the orchestrator.

- **`## Subsession`** — injected at runtime into sub-session invocations with
  `{workspace}` substituted. Present only on agents that support `run_parallel`
  (currently `coder` and `tester`). Tells the agent how to use its workspace.

### Adding Your Own Agents

Add a `.md` file to `~/.config/ergon/definitions/agents/`. The `orchestrator`
definition is required; all others are optional. Custom agents appear in
the orchestrator's tool schemas and can be referenced in channel presets.

Examples: designer, security auditor, documentation writer, database expert.

## Channels

Channels are collaborative conversations the orchestrator opens with one or
more specialists. They run in natural language — the orchestrator briefs the
channel, reads what comes back, and decides what to do next.

### Orchestrator Tools

**`open_channel`** — Start a new channel.

```
participants  list of agent IDs; repeat an ID for multiple instances
              (e.g., ["coder", "coder"] for two coders on the same call)
preset        name of a channel preset (mutually exclusive with participants)
message       the opening message from the orchestrator
recipients    which participants to address in this message
```

**`message_channel`** — Send another message into an open channel.

```
channel       channel ID (e.g., "channel-1")
message       the message content
recipients    which participants to address
```

**`close_channel`** — End a channel when the conversation is done.

```
channel       channel ID to close
```

**`run_parallel`** — Run N isolated sub-sessions of a code-generation agent
in parallel. All results come back at once in the next turn.

```
agent         agent ID (must have a ## Subsession section: coder or tester)
count         number of parallel sessions, 1–8 (default 1)
task          the task prompt given to every session
```

`run_parallel` is for **best-of-N code generation only** — when you want
multiple independent implementations of the same task and will pick or combine
the best result. It is a one-shot call with no feedback loop between sessions.
For review, analysis, research, or anything that needs back-and-forth, use
a channel.

### Channel Presets

Presets are named participant lists. The orchestrator can open a preset by name
instead of listing participants manually.

Ergon ships with two presets:

**`best-of-n`** — Three coders tackle the same problem independently. The
orchestrator compares the results and picks or combines the best approach.

**`debate`** — Architect, coder, critic, and reviewer on the same channel.
Good for stress-testing a plan from multiple angles before committing.

### Adding Your Own Presets

Add a `.md` file to `~/.config/ergon/definitions/channels/`:

```markdown
---
id: security-review
name: Security Review
participants:
  - researcher
  - reviewer
  - critic
---
```

All agent IDs must exist in your definitions.

## Workspaces

Agents running via `run_parallel` get an isolated **workspace overlay** — a
copy-on-write filesystem layer rooted at
`~/.ergon-workspace/{session_id}-parallel-{N}/`.

Inside the sub-session the agent's working directory is `/workspace/N/`. It can
use three tools:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read a UTF-8 file from the overlay (or real filesystem as fallback) |
| `write_file(path, content)` | Write a file into the overlay; never touches the real filesystem |
| `list_files(directory)` | List files in a directory (union of overlay and real filesystem) |

Each `run_parallel` batch receives the completed overlays from all prior
batches as read-only layers. This means a later agent can read what an earlier
one wrote, but cannot overwrite it.

After a `run_parallel` call completes, the orchestrator's loop history includes
a `Files written:` listing for each session, so it knows what was produced
without needing to re-read the files.

## Session Continuity

Ergon tracks your session with an `X-Ergon-Session` response header (also set
as an `ergon_session` cookie). Sending this header in subsequent requests
resumes the same orchestration context — open channels, in-progress tool
calls, and conversation history are all preserved.

When the orchestrator delegates tool execution to the client (e.g., running a
shell command or reading a file via the IDE's MCP tools), the tool call IDs are
encoded with a session token. The next request carries the tool results back,
and ergon reassembles the pending context automatically — no client changes
required.

Pending tool call records expire after 10 minutes of inactivity.

## API

Ergon exposes three endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion (streaming and non-streaming) |
| `POST` | `/v1/responses` | Response object API (streaming and non-streaming) |
| `GET` | `/v1/models` | List available upstream models (30-second cache) |

All request and response formats follow the OpenAI API specification. Clients
that work with OpenAI work with ergon without modification.

**Custom headers:**

| Header | Direction | Description |
|--------|-----------|-------------|
| `X-Ergon-Session` | Request & Response | Session ID for context continuity |

## Development

Install dev dependencies:

```bash
pip install -e '.[dev]'
```

Run all checks:

```bash
./scripts/check
```

Individual commands:

```bash
./scripts/format        # auto-format
ruff check .            # lint
mypy                    # type check
python -m pytest tests/ # unit tests
```

### Real Model E2E Tests

Smoke tests against a live upstream live in `tests/real_proxy_e2e.py`. Create
`.env.e2e-tests` in the repo root:

```
UPSTREAM_BASE_URL=http://localhost:8080/v1
MODEL=your-model-name
# UPSTREAM_API_KEY=optional
```

Run them:

```bash
./scripts/check-real-e2e
# or directly:
python -m pytest tests/real_proxy_e2e.py -v -s
```

Tests skip cleanly if the upstream is unavailable or the config file is
missing.

The orchestration smoke test (`test_orchestrator_builds_project_from_brief`)
sends a product brief to the full stack with no scripting — the orchestrator
decides which agents to use and what to build. After the run, inspect:

```
~/.ergon-workspace/{session_id}/run.log       # every orchestration event
~/.ergon-workspace/{session_id}-parallel-*/   # files the agents wrote
```

## License

See [LICENSE](LICENSE).
