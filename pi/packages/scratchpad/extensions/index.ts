// Ergon Scratchpad — Pi extension
//
// Reads .ergon.studio/scratchpad.md and injects it into the system prompt
// when present so the agent has access to project conventions, discovered
// notes, and architectural decisions without adding noise to every project.
//
// The scratchpad survives compaction because it is file-backed and re-read on
// every before_agent_start. Pi's compaction hook is for canceling/providing a
// summary, so this package does not force scratchpad contents into summaries.

import type {
  BeforeAgentStartEvent,
  ExtensionAPI,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

// ── Scratchpad ───────────────────────────────────────────────────────────────

function readScratchpad(cwd: string): string | null {
  const p = join(cwd, ".ergon.studio", "scratchpad.md");
  try {
    return existsSync(p) ? readFileSync(p, "utf8") : null;
  } catch {
    return null;
  }
}

function scratchpadBlock(cwd: string): string | null {
  const scratchpad = readScratchpad(cwd);
  if (scratchpad) {
    return `## Project Scratchpad\n\n${scratchpad}`;
  }
  return null;
}

// ── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI): void {
  pi.on(
    "before_agent_start",
    async (event: BeforeAgentStartEvent, ctx: ExtensionContext) => {
      const block = scratchpadBlock(event.systemPromptOptions?.cwd ?? ctx.cwd);
      if (!block) return;

      return {
        systemPrompt: (event.systemPrompt ?? "") + "\n\n" + block,
      };
    },
  );
}
