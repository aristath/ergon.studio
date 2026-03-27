# ergon.studio

A multi-agent orchestration plugin for [OpenCode](https://opencode.ai).

Ergon brings a team of AI agents to your OpenCode workflow. You talk to the **orchestrator** — a lead developer who understands your goal, breaks it down, and coordinates specialists (architect, coder, reviewer, critic, researcher, tester) as needed.

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

Ergon is also an OpenCode plugin that logs session events:

```typescript
import type { Plugin } from "@opencode-ai/plugin"

export const ErgonPlugin: Plugin = async ({ client }) => {
  return {
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
    },
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
