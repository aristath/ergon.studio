// Roster loading + deny-list computation.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, cpSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { loadRoster, getRosterEntry, denyListFor, _resetRosterCacheForTests, EXPECTED_ROSTER_IDS } from "../dist/roster.js";

const SPECIALISTS = [
  "scout", "architect", "coder", "researcher", "reviewer",
  "design_reviewer", "critic", "tester", "quality_controller",
];

test("loadRoster: 10 entries, orchestrator + 9 specialists", () => {
  const roster = loadRoster();
  assert.equal(roster.length, 10);
  const ids = roster.map((e) => e.id);
  assert.ok(ids.includes("orchestrator"));
  for (const s of SPECIALISTS) assert.ok(ids.includes(s), `missing ${s}`);
  for (const e of roster) {
    assert.ok(e.persona.length > 100, `${e.id} persona too short`);
    assert.ok(e.deny.every((d) => typeof d === "string"));
  }
});

test("loadRoster: frontmatter description parsed", () => {
  const scout = getRosterEntry("scout");
  assert.ok(scout);
  assert.ok(scout.description.length > 10, "scout description should come from frontmatter");
  // Persona body must not include the frontmatter.
  assert.ok(!scout.persona.startsWith("---"));
});

test("denyListFor: every specialist denies delegation + goal + user-facing tools", () => {
  for (const s of SPECIALISTS) {
    const deny = new Set(denyListFor(s));
    for (const t of ["subagent", "subagent_fork", "send_message", "list_agents", "interrupt_agent",
      "workflow", "ralph", "debate", "run_parallel",
      "create_goal", "get_goal", "update_goal", "ask_user_question"]) {
      assert.ok(deny.has(t), `${s} should deny ${t}`);
    }
  }
});

test("denyListFor: specialists deny other specialists; QC keeps reviewer+design_reviewer+tester", () => {
  const qc = new Set(denyListFor("quality_controller"));
  assert.ok(!qc.has("reviewer"), "QC must keep reviewer");
  assert.ok(!qc.has("design_reviewer"), "QC must keep design_reviewer");
  assert.ok(!qc.has("tester"), "QC must keep tester");
  assert.ok(qc.has("coder"), "QC denies coder");
  assert.ok(qc.has("scout"), "QC denies scout");

  const coder = new Set(denyListFor("coder"));
  for (const s of SPECIALISTS) if (s !== "coder") assert.ok(coder.has(s), `coder should deny ${s}`);
});

test("denyListFor: per-agent tool denies (opencode.json parity)", () => {
  assert.ok(new Set(denyListFor("scout")).has("bash"));
  const arch = new Set(denyListFor("architect"));
  assert.ok(arch.has("bash") && arch.has("edit") && arch.has("write"), "architect: edit deny adds write");
  assert.ok(new Set(denyListFor("reviewer")).has("edit"));
  assert.ok(new Set(denyListFor("reviewer")).has("write"));
  assert.ok(new Set(denyListFor("design_reviewer")).has("bash"));
  assert.ok(new Set(denyListFor("critic")).has("edit"));
  assert.ok(new Set(denyListFor("researcher")).has("edit"));
  // coder and tester keep everything except the base denies
  const coder = new Set(denyListFor("coder"));
  assert.ok(!coder.has("edit") && !coder.has("bash") && !coder.has("write"));
  const tester = new Set(denyListFor("tester"));
  assert.ok(!tester.has("edit") && !tester.has("bash") && !tester.has("write"));
});

test("denyListFor: unknown id still gets base denies (safe default)", () => {
  const deny = new Set(denyListFor("ghost"));
  assert.ok(deny.has("subagent"));
  assert.ok(deny.has("ask_user_question"));
});

test("denyListFor: sorted + no duplicates", () => {
  for (const s of ["orchestrator", ...SPECIALISTS]) {
    const d = denyListFor(s);
    assert.deepEqual(d, [...d].sort(), `${s} deny list not sorted`);
    assert.equal(new Set(d).size, d.length, `${s} deny list has duplicates`);
  }
});

test("roster cache reset hook works", () => {
  const a = loadRoster();
  _resetRosterCacheForTests();
  const b = loadRoster();
  assert.notEqual(a, b, "cache should have been cleared");
  assert.deepEqual(a.map((e) => e.id), b.map((e) => e.id));
});

test("EXPECTED_ROSTER_IDS: the 10 curated ids", () => {
  assert.equal(EXPECTED_ROSTER_IDS.length, 10);
  assert.ok(EXPECTED_ROSTER_IDS.includes("orchestrator"));
  for (const s of SPECIALISTS) assert.ok(EXPECTED_ROSTER_IDS.includes(s), `missing ${s}`);
});

test("loadRoster: an extra agents/*.md becomes an entry and warns", () => {
  const realAgents = join(dirname(fileURLToPath(import.meta.url)), "..", "agents");
  const tmp = mkdtempSync(join(tmpdir(), "ergon-agents-"));
  const savedDir = process.env.ERGON_AGENTS_DIR;
  let warned = "";
  const realWarn = console.warn;
  try {
    cpSync(realAgents, tmp, { recursive: true });
    writeFileSync(
      join(tmp, "notes.md"),
      "---\ndescription: a stray agent\n---\n\n# Notes\nBody of the stray agent, long enough to pass the persona length check used by other tests.\n",
    );
    process.env.ERGON_AGENTS_DIR = tmp;
    console.warn = (m) => { warned += String(m); };
    _resetRosterCacheForTests();
    const roster = loadRoster();
    assert.equal(roster.length, 11, "stray .md becomes a roster entry");
    assert.ok(roster.find((e) => e.id === "notes"));
    assert.ok(warned.includes("notes"), `expected a warning naming the stray, got: ${warned}`);
  } finally {
    console.warn = realWarn;
    _resetRosterCacheForTests();
    if (savedDir === undefined) delete process.env.ERGON_AGENTS_DIR;
    else process.env.ERGON_AGENTS_DIR = savedDir;
    rmSync(tmp, { recursive: true, force: true });
  }
});
