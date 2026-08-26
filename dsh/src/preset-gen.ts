// Ergon agent preset generator.
//
// Emits `agent.cordis.yml` + `preset.yml` for the ergon preset from the
// roster (agents/*.md) — the same file the installer writes to
// `~/.dsh/.agent-presets/ergon/`. The generated preset is based on the
// shipped `standard` preset (identity/shell/fs/jobs/skills/goal/plan-mode/
// compaction/tool rows) with three changes:
//
//   1. the persona row carries the orchestrator body;
//   2. the delegation group gains one `dsh-tool-subagent` row per specialist
//      (toolName = agent id, persona = agent body, deny-based toolFilter).
//
// The ergon plugin itself (`@ergon.studio/dsh`) is NOT a preset row: the
// package's `dsh.bundle` patch (cordis.patch.yml) mounts it at profile level
// for every session. A preset row would double-mount the plugin in profiles
// that also carry the bundle (duplicate tools + duplicate event listeners).
//
// The plan-mode section and the compaction group are copied verbatim from
// the shipped standard preset (copy-and-edit is the supported preset model;
// never edit shipped presets in place).

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import yaml from "js-yaml";
import { EXPECTED_ROSTER_IDS, loadRoster, type RosterEntry } from "./roster.js";

// === loader YAML dialect (the !!js tag) ===
//
// The preset loader (dsh-agent-presets) parses compositions with a js-yaml
// schema extended by a `!!js` scalar tag: `disabled: !!js process.platform
// !== 'win32'` round-trips as an expression node the loader evaluates at
// entry activation — how the shipped standard/base presets gate the shell
// rows by platform. We mirror that dialect so the generated preset can carry
// the same platform gates, and so validatePresetFile can parse it.

const jsExprType = new yaml.Type("tag:yaml.org,2002:js", {
  kind: "scalar",
  resolve: (data: unknown) => typeof data === "string",
  construct: (data: string) => ({ __jsExpr: data }),
  predicate: (v: unknown) => v instanceof Object && "__jsExpr" in v,
  represent: (v: object) => (v as { __jsExpr: string }).__jsExpr,
});

/** The preset YAML dialect: plain YAML + the loader's `!!js` expression tag. */
export const presetSchema = yaml.JSON_SCHEMA.extend(jsExprType);

/** Marker for a `!!js` expression value in a generated row (platform gates). */
export function jsExpr(expression: string): { __jsExpr: string } {
  return { __jsExpr: expression };
}

/** Load preset YAML with the `!!js` dialect (plain yaml.load throws on the tag). */
export function loadPresetYaml(raw: string): unknown {
  return yaml.load(raw, { schema: presetSchema });
}

/** Plan-mode policy text (verbatim from the shipped standard/code preset). */
const PLAN_MODE_SECTION = `You are in plan mode. Stay in plan mode until exit_plan_mode succeeds or the user switches the session mode. Imperative language to implement changes means plan the implementation, not execute it. A user's conversational agreement — including an answer confirming something you asked — approves nothing and does not end plan mode; fold the confirmed decision into the plan and submit it through exit_plan_mode.

Explore first. Use non-mutating reads, searches, static analysis, and checks to ground the plan in the actual repository. Do not edit or write files, change configuration, run formatters or code generation that rewrites tracked files, commit, or otherwise carry out the plan. Prefer existing functions and patterns over new machinery.

The tool catalog stays the same across modes for request-cache stability. These plan-mode rules override any later tool description or guidance that suggests using mutation tools; those tools remain listed to keep the tool catalog unchanged. Do not use todo_write to track this planning phase: it tracks implementation after an approved plan, while the plan itself belongs in exit_plan_mode.

Resolve discoverable facts by inspection. Use ask_user_question only for user-owned choices or material ambiguity that inspection cannot answer. Do not ask the user where code lives or how current behavior works when you can find out.

Make the plan decision-complete: state the goal and success criteria; group implementation changes by subsystem; identify public API, schema, and data-flow changes; cover edge cases, failure modes, tests, acceptance criteria, and explicit assumptions. Keep it concise enough to review but detailed enough that another engineer can implement it without making design decisions.

When ready, call exit_plan_mode with the complete plan markdown, starting with a # title. Make exit_plan_mode the only and final tool call in that assistant response: it presents the plan for approval, and implementation begins only in a later step after approval. Do not paste the final plan as a plain reply or ask "should I proceed?" through prose or ask_user_question. If review rejects it, incorporate the feedback and present again. If the review channel is unavailable or aborted, stay in plan mode and ask the user to switch modes manually; do not proceed with implementation.`;

/** The 9 specialists (orchestrator is the primary persona, not a tool). */
const SPECIALIST_ORDER = [
  "scout",
  "architect",
  "coder",
  "researcher",
  "reviewer",
  "design_reviewer",
  "critic",
  "tester",
  "quality_controller",
] as const;

export interface GeneratedPreset {
  "agent.cordis.yml": string;
  "preset.yml": string;
}

export function generatePreset(roster: RosterEntry[] = loadRoster()): GeneratedPreset {
  const byId = new Map(roster.map((e) => [e.id, e]));
  const orchestrator = byId.get("orchestrator");
  if (!orchestrator) throw new Error("ergon preset: orchestrator.md missing from roster");
  for (const id of SPECIALIST_ORDER) {
    if (!byId.get(id)) throw new Error(`ergon preset: ${id}.md missing from roster`);
  }
  // Fail the build on any extra agents/*.md: every roster id becomes a
  // spawnable agent via debate/run_parallel, so a stray file is a real
  // surface change, not a comment.
  for (const entry of roster) {
    if (!EXPECTED_ROSTER_IDS.includes(entry.id)) {
      throw new Error(
        `ergon preset: unexpected agent "${entry.id}" in agents/ — remove the file or extend the curated roster`,
      );
    }
  }

  const rows: Array<Record<string, unknown>> = [];

  // ── identity ────────────────────────────────────────────────────────────
  rows.push({
    id: "persona",
    name: "@deepseek-ai/dsh-persona",
    config: { text: orchestrator.persona + "\n" },
  });
  rows.push({
    id: "agent-instructions",
    name: "@deepseek-ai/dsh-agent-instructions",
    config: { maxBytes: 65536 },
  });

  // ── shell / fs / jobs ───────────────────────────────────────────────────
  // Shell rows carry the standard preset's platform gates (evaluated by the
  // preset loader at entry activation via the !!js tag): exactly one shell
  // stack per host — bash off on Windows, pwsh off everywhere else.
  rows.push({
    id: "tool-bash",
    name: "@deepseek-ai/dsh-tool-bash",
    disabled: jsExpr("process.platform === 'win32'"),
  });
  rows.push({
    id: "tool-pwsh",
    name: "@deepseek-ai/dsh-tool-pwsh",
    disabled: jsExpr("process.platform !== 'win32'"),
  });
  rows.push({ id: "tool-fs", name: "@deepseek-ai/dsh-tool-fs" });
  rows.push({
    id: "tool-fs-search",
    name: "@deepseek-ai/dsh-tool-fs-search",
    config: { sampleOverCapGlobResults: false },
  });
  rows.push({ id: "tool-jobs", name: "@deepseek-ai/dsh-tool-jobs" });

  // ── skills ──────────────────────────────────────────────────────────────
  rows.push({ id: "skill-filesystem", name: "@deepseek-ai/dsh-skill-filesystem" });
  rows.push({ id: "tool-skill", name: "@deepseek-ai/dsh-tool-skill" });

  // ── goals ───────────────────────────────────────────────────────────────
  rows.push({ id: "tool-goal", name: "@deepseek-ai/dsh-tool-goal" });

  // ── plan mode (verbatim from the shipped standard preset) ───────────────
  rows.push({
    id: "planning",
    name: "cordis:group",
    group: true,
    isolate: { planMode: true },
    config: [
      {
        id: "plan-mode",
        name: "@deepseek-ai/dsh-plan-mode",
        config: { section: PLAN_MODE_SECTION + "\n" },
      },
    ],
  });

  // ── compaction (verbatim from the shipped standard preset) ──────────────
  rows.push({
    id: "compaction",
    name: "cordis:group",
    group: true,
    isolate: { compaction: true, toolResultPruner: true },
    config: [
      { id: "compaction-basic", name: "@deepseek-ai/dsh-compaction-basic" },
      { id: "command-compact", name: "@deepseek-ai/dsh-command-compact" },
      {
        id: "tool-result-pruner",
        name: "@deepseek-ai/dsh-compaction-tool-result-pruner",
        config: { thresholdChars: 8192, headChars: 4096, tailChars: 1024 },
      },
    ],
  });

  // ── delegation: generic subagents, workflow/ralph, and the ergon team ────
  // Mirrors the shipped standard preset's delegation group: the workflow
  // engine (worker-thread provider) and the workflow/ralph tools sit in an
  // entry-local realm because `dsh-tool-workflow` injects the `workflowEngine`
  // service the worker row provides. Specialists deny workflow/ralph via their
  // toolFilter, so only the orchestrator sees them.
  const delegation: Array<Record<string, unknown>> = [
    { id: "tool-subagent-control", name: "@deepseek-ai/dsh-tool-subagent-control" },
    { id: "tool-subagent-list-agents", name: "@deepseek-ai/dsh-tool-subagent-control/list-agents" },
    {
      id: "tool-subagent",
      name: "@deepseek-ai/dsh-tool-subagent",
      config: { provider: "spawn", toolName: "subagent", backgroundMode: "continuable" },
    },
    {
      id: "tool-subagent-fork",
      name: "@deepseek-ai/dsh-tool-subagent",
      config: { provider: "fork", toolName: "subagent_fork", backgroundMode: "continuable" },
    },
    {
      id: "workflow-worker-thread",
      name: "@deepseek-ai/dsh-workflow-worker-thread",
      config: { provider: "spawn" },
    },
    { id: "tool-workflow", name: "@deepseek-ai/dsh-tool-workflow" },
    {
      id: "tool-ralph",
      name: "@deepseek-ai/dsh-tool-ralph",
      config: { subagentProvider: "spawn", maxRounds: 64 },
    },
  ];
  for (const id of SPECIALIST_ORDER) {
    const entry = byId.get(id)!;
    delegation.push({
      id: `tool-specialist-${id}`,
      name: "@deepseek-ai/dsh-tool-subagent",
      config: {
        provider: "spawn",
        toolName: id,
        backgroundMode: "one-shot",
        persona: entry.persona + "\n",
        toolFilter: { deny: entry.deny },
      },
    });
  }
  rows.push({
    id: "delegation",
    name: "cordis:group",
    group: true,
    isolate: { workflowEngine: true },
    config: delegation,
  });

  // ── remaining model-facing rows ─────────────────────────────────────────
  rows.push({ id: "tool-ask-user", name: "@deepseek-ai/dsh-tool-ask-user" });
  rows.push({
    id: "tool-todo",
    name: "@deepseek-ai/dsh-tool-todo",
    config: { allowParallelInProgress: true },
  });
  rows.push({
    id: "tool-web",
    name: "@deepseek-ai/dsh-tool-web",
    config: { fetch: false, searchTimeoutMs: 60000 },
  });

  const agentCordis = yaml.dump(rows, {
    lineWidth: -1,
    noRefs: true,
    quotingType: "'",
    schema: presetSchema,
  });
  const preset = yaml.dump(
    {
      name: "Ergon",
      description:
        "Ergon multi-agent team: orchestrator lead dev, 9 specialist tools, debate, parallel fan-out, memory steward, scratchpad.",
      order: 3,
    },
    { lineWidth: -1, noRefs: true },
  );

  return { "agent.cordis.yml": agentCordis, "preset.yml": preset };
}

/** Write the generated preset files into `outDir`. */
export function writePreset(outDir: string, preset: GeneratedPreset = generatePreset()): void {
  mkdirSync(outDir, { recursive: true });
  for (const [file, content] of Object.entries(preset)) {
    writeFileSync(join(outDir, file), content, "utf8");
  }
}

/** Load + sanity-check a generated agent.cordis.yml (used by the CLI). */
export function validatePresetFile(path: string): string[] {
  const raw = readFileSync(path, "utf8");
  // The !!js dialect: the shell rows carry platform gates the plain YAML
  // loader rejects.
  const rows = loadPresetYaml(raw) as Array<Record<string, unknown>>;
  const problems: string[] = [];
  if (!Array.isArray(rows) || rows.length === 0) {
    problems.push("agent.cordis.yml is not a non-empty row list");
    return problems;
  }
  // Flattened row ids: specialist rows live inside the delegation group.
  const ids: string[] = [];
  const walk = (list: Array<Record<string, unknown>>) => {
    for (const r of list) {
      if (typeof r?.id === "string") ids.push(r.id);
      if (r?.group && Array.isArray(r.config)) walk(r.config);
    }
  };
  walk(rows);
  const required = ["persona", ...SPECIALIST_ORDER.map((s) => `tool-specialist-${s}`)];
  for (const id of required) {
    if (!ids.includes(id)) problems.push(`missing row id "${id}"`);
  }
  // The plugin is mounted at profile level by the package's dsh.bundle patch;
  // a stale preset still carrying the row would double-mount it.
  if (ids.includes("ergon-plugin")) {
    problems.push(
      'row id "ergon-plugin" is no longer part of the preset (the plugin mounts at profile level via the ergon bundle); remove the row — `npx @ergon.studio/dsh init --force` regenerates the preset',
    );
  }
  return problems;
}
