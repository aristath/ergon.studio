// Scratchpad + handoff file helpers.
//
// The scratchpad is re-read on every prompt assembly (through the plugin's
// dynamic context entry), so it re-enters the model's context after
// compaction without any compaction hook — the runtime-context projection
// appends a fresh snapshot whenever the file content changes.

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const SCRATCHPAD_REL = join(".ergon.studio", "scratchpad.md");
export const HANDOFF_REL = join(".ergon.studio", "HANDOFF.md");

export function scratchpadPath(cwd: string): string {
  return join(cwd, SCRATCHPAD_REL);
}
export function handoffPath(cwd: string): string {
  return join(cwd, HANDOFF_REL);
}

export function readScratchpad(cwd: string): string | null {
  const p = scratchpadPath(cwd);
  try {
    return existsSync(p) ? readFileSync(p, "utf8") : null;
  } catch {
    return null;
  }
}

export function readHandoff(cwd: string): string | null {
  const p = handoffPath(cwd);
  try {
    return existsSync(p) ? readFileSync(p, "utf8") : null;
  } catch {
    return null;
  }
}

/** Rendered when the project has no scratchpad yet (parity with OpenCode). */
export const NO_SCRATCHPAD =
  `No scratchpad yet for this project. ` +
  `When you discover something worth keeping (a constraint, a gotcha, a decision and why), ` +
  `create \`.ergon.studio/scratchpad.md\` with \`## Conventions\`, \`## Notes\`, and \`## Decisions\` sections.`;

/** Build the scratchpad context block (used by the plugin's context entry). */
export function scratchpadBlock(cwd: string): string {
  const pad = readScratchpad(cwd);
  if (pad && pad.trim().length > 0) {
    const handoff = readHandoff(cwd);
    const extra =
      handoff && handoff.trim().length > 0
        ? `\n\n## Handoff (read this first)\n\n${handoff.trim()}`
        : "";
    return `## Project Scratchpad\n\n${pad.trim()}${extra}`;
  }
  return `## Project Scratchpad\n\n${NO_SCRATCHPAD}`;
}
