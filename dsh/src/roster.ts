// Agent roster: loads the ergon team definitions (agents/*.md) that ship
// with this package, and computes each specialist's tool-filter deny list
// for the `dsh-tool-subagent` rows and the debate/run_parallel tools.
//
// The .md files are the same source of truth the OpenCode plugin used:
// YAML frontmatter (description, temperature, mode, optional permission
// denies) + a persona body. In DSH the persona is inlined into the preset
// row by the installer, and the tool-filter deny list is applied both in the
// preset (at mount) and at runtime by the debate/run_parallel tools.

import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/** Directory containing the 10 agent .md files (package root /agents). */
export function agentsDir(): string {
  // dist/roster.js → package root is "..", then agents/
  return process.env.ERGON_AGENTS_DIR ?? join(__dirname, "..", "agents");
}

export interface RosterEntry {
  id: string;
  description: string;
  /** Persona body (file content minus frontmatter), trimmed. */
  persona: string;
  /** Full tool-filter deny list for this agent. */
  deny: string[];
}

/** Tools denied to every specialist (delegation + goal + user-facing + peer specialists). */
const BASE_DENY = [
  // generic delegation
  "subagent",
  "subagent_fork",
  "send_message",
  "list_agents",
  "interrupt_agent",
  "workflow",
  "ralph",
  // goal machinery
  "create_goal",
  "get_goal",
  "update_goal",
  // user-facing
  "ask_user_question",
];

/**
 * Per-agent extra denies, mirroring the OpenCode plugin's opencode.json
 * permission map (the frontmatter `permission:` blocks repeat the same).
 * Kept here as the single source of truth so the preset build and the
 * runtime tools agree.
 */
const EXTRA_DENY: Record<string, string[]> = {
  scout: ["bash"],
  architect: ["edit", "bash"],
  reviewer: ["edit"],
  design_reviewer: ["edit", "bash"],
  quality_controller: ["edit", "bash"],
  critic: ["edit"],
  researcher: ["edit"],
  coder: [],
  tester: [],
  orchestrator: [],
};

/** The 9 specialist ids (everyone except the primary orchestrator). */
const SPECIALISTS = [
  "scout",
  "architect",
  "coder",
  "reviewer",
  "design_reviewer",
  "critic",
  "researcher",
  "tester",
  "quality_controller",
];

/**
 * quality_controller runs the quality loop and needs to invoke reviewer,
 * design_reviewer, and tester as tools (parity with the OpenCode version,
 * where the task tool was unrestricted and the QC persona directs it to call
 * all three); it keeps those peer tools.
 */
const QC_KEEP: ReadonlySet<string> = new Set(["reviewer", "design_reviewer", "tester"]);

/**
 * Compute the deny list for one roster entry.
 *
 * - every specialist denies BASE_DENY;
 * - every specialist denies the other specialist tool names (no nesting),
 *   except quality_controller keeps reviewer + design_reviewer + tester;
 * - per-agent extra denies (edit/bash/write as configured);
 * - write denies apply to every tool that performs that operation: `edit`
 *   denies both `edit` and `write` (the OpenCode `edit` permission covered
 *   both file-mutation tools).
 */
export function denyListFor(id: string): string[] {
  const set = new Set<string>(BASE_DENY);
  if (SPECIALISTS.includes(id)) {
    for (const other of SPECIALISTS) {
      if (other === id) continue;
      if (id === "quality_controller" && QC_KEEP.has(other)) continue;
      set.add(other);
    }
  }
  for (const extra of EXTRA_DENY[id] ?? []) {
    set.add(extra);
    if (extra === "edit") set.add("write");
  }
  return [...set].sort();
}

/** Minimal frontmatter reader: `key: value` lines + one nested `permission:` map. */
function parseFrontmatter(content: string): {
  meta: Record<string, string>;
  permissions: string[];
  body: string;
} {
  const match = content.match(/^---[ \t]*\n([\s\S]*?)\n---[ \t]*\n?([\s\S]*)$/);
  if (!match) return { meta: {}, permissions: [], body: content };
  const meta: Record<string, string> = {};
  const permissions: string[] = [];
  let inPermission = false;
  for (const rawLine of match[1].split("\n")) {
    if (!rawLine.trim()) continue;
    const indented = /^[ \t]+\S/.test(rawLine);
    const line = rawLine.trim();
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    const value = line.slice(colonIdx + 1).trim();
    if (inPermission) {
      if (indented && value === "deny") permissions.push(key);
      else if (!indented) inPermission = false;
    }
    if (!indented) {
      if (key === "permission" && value === "") inPermission = true;
      else if (value) meta[key] = value;
    }
  }
  return { meta, permissions, body: match[2].trim() };
}

let rosterCache: RosterEntry[] | null = null;

/**
 * Load the roster from agentsDir(). Cached for the process lifetime — the
 * agent files ship with the package and don't change at runtime.
 */
export function loadRoster(): RosterEntry[] {
  if (rosterCache) return rosterCache;
  const dir = agentsDir();
  const files = readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort();
  const entries: RosterEntry[] = [];
  for (const file of files) {
    const id = file.slice(0, -3);
    const content = readFileSync(join(dir, file), "utf8");
    const { meta, permissions, body } = parseFrontmatter(content);
    // Frontmatter permission denies win/merge with the curated map.
    const deny = new Set(denyListFor(id));
    for (const p of permissions) {
      deny.add(p);
      if (p === "edit") deny.add("write");
    }
    entries.push({
      id,
      description: meta.description ?? "",
      persona: body,
      deny: [...deny].sort(),
    });
  }
  rosterCache = entries;
  return rosterCache;
}

export function getRosterEntry(id: string): RosterEntry | undefined {
  return loadRoster().find((e) => e.id === id);
}

/** Test hook. */
export function _resetRosterCacheForTests(): void {
  rosterCache = null;
}
