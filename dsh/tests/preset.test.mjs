// Generated preset structure + validation.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, readFileSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import yaml from "js-yaml";

import { generatePreset, writePreset, validatePresetFile } from "../dist/preset-gen.js";
import { loadRoster, denyListFor } from "../dist/roster.js";

const SPECIALISTS = [
  "scout", "architect", "coder", "researcher", "reviewer",
  "design_reviewer", "critic", "tester", "quality_controller",
];

function flat(rows, prefix = "") {
  const out = [];
  for (const r of rows) {
    if (r.group) out.push(...flat(r.config || [], prefix + r.id + "/"));
    else out.push({ id: prefix + r.id, row: r });
  }
  return out;
}

test("generatePreset: top-level row ids", () => {
  const p = generatePreset();
  assert.ok(p["agent.cordis.yml"]);
  assert.ok(p["preset.yml"]);
  const rows = yaml.load(p["agent.cordis.yml"]);
  const ids = rows.map((r) => r.id);
  for (const id of ["persona", "agent-instructions", "tool-bash", "tool-fs", "tool-fs-search",
    "tool-jobs", "skill-filesystem", "tool-skill", "tool-goal", "planning", "compaction",
    "delegation", "tool-ask-user", "tool-todo", "tool-web"]) {
    assert.ok(ids.includes(id), `missing top-level row ${id}`);
  }
  // The ergon plugin mounts at profile level via the package's dsh.bundle
  // patch — it must NOT be a preset row (double mount).
  assert.ok(!ids.includes("ergon-plugin"), "ergon-plugin must not be a preset row");
});

test("generatePreset: persona row carries the orchestrator body", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const persona = rows.find((r) => r.id === "persona");
  assert.equal(persona.name, "@deepseek-ai/dsh-persona");
  const orch = loadRoster().find((e) => e.id === "orchestrator");
  assert.ok(persona.config.text.includes(orch.persona.slice(0, 60)));
  assert.ok(persona.config.text.includes("You are the lead dev."));
});

test("generatePreset: every specialist row has persona + deny toolFilter + one-shot mode", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const delegation = rows.find((r) => r.id === "delegation");
  assert.ok(delegation.group);
  const byId = new Map(delegation.config.map((r) => [r.id, r]));
  for (const s of SPECIALISTS) {
    const row = byId.get(`tool-specialist-${s}`);
    assert.ok(row, `missing specialist row ${s}`);
    assert.equal(row.name, "@deepseek-ai/dsh-tool-subagent");
    assert.equal(row.config.provider, "spawn");
    assert.equal(row.config.toolName, s);
    assert.equal(row.config.backgroundMode, "one-shot");
    const entry = loadRoster().find((e) => e.id === s);
    assert.equal(row.config.persona.trim(), entry.persona.trim() + "");
    assert.deepEqual(row.config.toolFilter.deny, entry.deny);
    assert.deepEqual(row.config.toolFilter.deny, denyListFor(s));
  }
});

test("generatePreset: generic subagent + fork rows are continuable", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const delegation = rows.find((r) => r.id === "delegation");
  const byId = new Map(delegation.config.map((r) => [r.id, r]));
  assert.equal(byId.get("tool-subagent").config.toolName, "subagent");
  assert.equal(byId.get("tool-subagent").config.backgroundMode, "continuable");
  assert.equal(byId.get("tool-subagent-fork").config.toolName, "subagent_fork");
  assert.equal(byId.get("tool-subagent-fork").config.provider, "fork");
});

test("generatePreset: delegation group isolates workflowEngine and carries workflow/ralph rows", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const delegation = rows.find((r) => r.id === "delegation");
  assert.deepEqual(delegation.isolate, { workflowEngine: true });
  const byId = new Map(delegation.config.map((r) => [r.id, r]));
  assert.equal(byId.get("workflow-worker-thread").name, "@deepseek-ai/dsh-workflow-worker-thread");
  assert.equal(byId.get("workflow-worker-thread").config.provider, "spawn");
  assert.equal(byId.get("tool-workflow").name, "@deepseek-ai/dsh-tool-workflow");
  assert.equal(byId.get("tool-ralph").name, "@deepseek-ai/dsh-tool-ralph");
  assert.deepEqual(byId.get("tool-ralph").config, { subagentProvider: "spawn", maxRounds: 64 });
});

test("generatePreset: planning + compaction groups carry the shipped isolates", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const planning = rows.find((r) => r.id === "planning");
  assert.deepEqual(planning.isolate, { planMode: true });
  assert.ok(planning.config[0].config.section.includes("You are in plan mode."));
  const compaction = rows.find((r) => r.id === "compaction");
  assert.deepEqual(compaction.isolate, { compaction: true, toolResultPruner: true });
  const pruner = compaction.config.find((r) => r.id === "tool-result-pruner");
  assert.deepEqual(pruner.config, { thresholdChars: 8192, headChars: 4096, tailChars: 1024 });
});

test("generatePreset: no ergon-plugin row (the bundle owns the plugin mount)", () => {
  const p = generatePreset();
  const rows = yaml.load(p["agent.cordis.yml"]);
  const flat = rows.flatMap((r) => (r.group ? r.config : [r]));
  assert.equal(flat.find((r) => r.id === "ergon-plugin"), undefined);
});

test("generatePreset: preset.yml has name/description/order", () => {
  const p = generatePreset();
  const meta = yaml.load(p["preset.yml"]);
  assert.equal(meta.name, "Ergon");
  assert.ok(meta.description.length > 10);
  assert.equal(typeof meta.order, "number");
});

test("generatePreset: throws when orchestrator missing", () => {
  assert.throws(() => generatePreset([]), /orchestrator/);
});

test("writePreset + validatePresetFile: round-trip in a temp dir", () => {
  const dir = mkdtempSync(join(tmpdir(), "ergon-preset-"));
  try {
    writePreset(dir);
    assert.ok(existsSync(join(dir, "agent.cordis.yml")));
    assert.ok(existsSync(join(dir, "preset.yml")));
    const problems = validatePresetFile(join(dir, "agent.cordis.yml"));
    assert.deepEqual(problems, []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("validatePresetFile: reports missing rows", () => {
  const dir = mkdtempSync(join(tmpdir(), "ergon-preset-bad-"));
  try {
    const bad = join(dir, "agent.cordis.yml");
    // Valid YAML row list but missing required ids.
    writeFileSync(bad, "- id: persona\n  name: x\n");
    const problems = validatePresetFile(bad);
    assert.ok(problems.some((p) => p.includes("tool-specialist-scout")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("validatePresetFile: rejects a stale preset still carrying the ergon-plugin row", () => {
  const dir = mkdtempSync(join(tmpdir(), "ergon-preset-stale-"));
  try {
    const stale = join(dir, "agent.cordis.yml");
    // A current (valid) preset, plus the now-forbidden plugin row appended.
    writeFileSync(stale, generatePreset()["agent.cordis.yml"] + "\n- id: ergon-plugin\n  name: '@ergon.studio/dsh'\n");
    const problems = validatePresetFile(stale);
    assert.ok(problems.some((p) => p.includes("no longer part of the preset")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
