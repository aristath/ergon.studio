// Debate verdict/prompt/transcript + runDebate / runParallel on a fake subagent seam.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  parseDebateVerdict,
  debatePrompt,
  renderDebateTranscript,
  outputText,
  runDebate,
  runParallel,
  SPAWN_MAX_DEPTH,
} from "../dist/debate.js";
import { getRosterEntry } from "../dist/roster.js";

// === verdict parsing ===

test("parseDebateVerdict: recognizes all three verdicts (case-insensitive)", () => {
  assert.equal(parseDebateVerdict("blah\nVerdict: AGREE"), "AGREE");
  assert.equal(parseDebateVerdict("Verdict: agree"), "AGREE");
  assert.equal(parseDebateVerdict("Verdict: CONTINUE"), "CONTINUE");
  assert.equal(parseDebateVerdict("Verdict: BLOCKED"), "BLOCKED");
});

test("parseDebateVerdict: uses the LAST line only", () => {
  assert.equal(parseDebateVerdict("Verdict: AGREE\nVerdict: CONTINUE"), "CONTINUE");
  assert.equal(parseDebateVerdict("I say Verdict: AGREE inline\nVerdict: BLOCKED"), "BLOCKED");
});

test("parseDebateVerdict: no match → CONTINUE", () => {
  assert.equal(parseDebateVerdict("no verdict here"), "CONTINUE");
  assert.equal(parseDebateVerdict(""), "CONTINUE");
  assert.equal(parseDebateVerdict("Verdict: MAYBE"), "CONTINUE");
});

// === prompt construction ===

test("debatePrompt: first turn (no peer output)", () => {
  const p = debatePrompt("fix the bug", "coder", "reviewer");
  assert.ok(p.includes("You are coder in a two-agent coding debate with reviewer."));
  assert.ok(p.includes("Task:\nfix the bug"));
  assert.ok(p.includes("Do the first pass."));
  assert.ok(p.includes("Verdict: AGREE, Verdict: CONTINUE, or Verdict: BLOCKED."));
  assert.ok(!p.includes("Your previous response:"));
});

test("debatePrompt: later turn carries own previous + peer latest", () => {
  const p = debatePrompt("fix the bug", "reviewer", "coder", "my prev", "peer latest");
  assert.ok(p.includes("Your previous response:\nmy prev"));
  assert.ok(p.includes("coder's latest response:\npeer latest"));
  assert.ok(p.includes("If you agree it is optimal, say why and use Verdict: AGREE."));
});

test("debatePrompt: own previous defaults to (none)", () => {
  const p = debatePrompt("t", "a", "b", undefined, "peer out");
  assert.ok(p.includes("Your previous response:\n(none)"));
});

// === transcript rendering ===

test("renderDebateTranscript: status, participants, turns, latest, full transcript", () => {
  const t = renderDebateTranscript({
    status: "AGREE",
    agentA: "coder",
    agentB: "reviewer",
    entries: [
      { turn: 1, agent: "coder", text: "did it", verdict: "CONTINUE" },
      { turn: 2, agent: "reviewer", text: "looks good", verdict: "AGREE" },
    ],
  });
  assert.ok(t.includes("Status: AGREE"));
  assert.ok(t.includes("Participants: coder, reviewer"));
  assert.ok(t.includes("Turns: 2"));
  assert.ok(t.includes("### reviewer\n\nlooks good")); // latest
  assert.ok(t.includes("### Turn 1 - coder"));
  assert.ok(t.includes("### Turn 2 - reviewer"));
});

test("renderDebateTranscript: error section when provided", () => {
  const t = renderDebateTranscript({
    status: "FAILED", agentA: "a", agentB: "b", entries: [], error: "boom",
  });
  assert.ok(t.includes("## Error"));
  assert.ok(t.includes("boom"));
});

// === outputText ===

test("outputText: string passthrough", () => {
  assert.equal(outputText("hello"), "hello");
});

test("outputText: content-block array joined", () => {
  const out = outputText([
    { type: "text", text: "one" },
    { type: "tool-call", id: "x", name: "t", arguments: "{}" },
    { type: "text", text: "two" },
  ]);
  assert.equal(out, "one\ntwo");
});

test("outputText: {text} object and junk", () => {
  assert.equal(outputText({ text: "obj" }), "obj");
  assert.equal(outputText(42), "");
  assert.equal(outputText(null), "");
  assert.equal(outputText(undefined), "");
  assert.equal(outputText([]), "");
});

// === fake subagent seam ===

function roster(id) {
  const e = getRosterEntry(id);
  assert.ok(e, `roster entry ${id} should exist`);
  return e;
}

function scriptedSubagents(...parts) {
  // Each part is one spec (string | Error | function) or an array of specs.
  const script = parts.flat();
  const calls = [];
  let i = 0;
  return {
    calls,
    async start(_provider, request) {
      calls.push(request);
      const spec = script[i++ % script.length];
      if (spec instanceof Error) throw spec;
      const output = typeof spec === "function" ? spec(request) : spec;
      return {
        result: Promise.resolve({ output, stopReason: "completed" }),
        dispose() {},
      };
    },
  };
}

test("runDebate: stops on AGREE at turn 2", async () => {
  const subs = scriptedSubagents([
    "made the fix\nVerdict: CONTINUE",
    "looks right\nVerdict: AGREE",
    "SHOULD NOT BE CALLED",
  ]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "fix the bug", maxTurns: 6, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(transcript.includes("Status: AGREE"));
  assert.equal(subs.calls.length, 2);
  // turn 2 prompt carries turn 1's output
  assert.ok(subs.calls[1].prompt[0].text.includes("made the fix"));
  assert.equal(subs.calls[1].persona, roster("reviewer").persona);
});

test("runDebate: spawns carry the absolute depth cap (no harness default applies to these)", () => {
  assert.equal(SPAWN_MAX_DEPTH, 3, "matches the harness generic-subagent default");
  const subs = scriptedSubagents(["x\nVerdict: CONTINUE"], ["y\nVerdict: AGREE"]);
  void runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 6, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  // The first spawn is issued synchronously before the first await, so the
  // call is already recorded here.
  assert.ok(subs.calls.length >= 1, "debate should have started its first spawn");
  assert.equal(subs.calls[0].maxDepth, SPAWN_MAX_DEPTH, "debate spawn must set maxDepth");
});

test("runDebate: MAX_TURNS when nobody agrees", async () => {
  const subs = scriptedSubagents(["still not right\nVerdict: CONTINUE"]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 3, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(transcript.includes("Status: MAX_TURNS"));
  assert.equal(subs.calls.length, 3);
});

test("runDebate: first-turn AGREE/BLOCKED does not stop (needs a review to agree on)", async () => {
  const subs = scriptedSubagents([
    "cannot proceed\nVerdict: BLOCKED",
    "reviewing\nVerdict: CONTINUE",
    "still reviewing\nVerdict: CONTINUE",
  ]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 3, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  // turn 1 blocked → turns 2 and 3 still run → MAX_TURNS
  assert.equal(subs.calls.length, 3);
  assert.ok(transcript.includes("Status: MAX_TURNS"));
});

test("runDebate: BLOCKED on turn>1 stops", async () => {
  const subs = scriptedSubagents(["first\nVerdict: CONTINUE"], ["need a decision\nVerdict: BLOCKED"]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 6, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(transcript.includes("Status: BLOCKED"));
  assert.equal(subs.calls.length, 2);
});

test("runDebate: subagent failure → FAILED with error", async () => {
  const subs = scriptedSubagents([new Error("spawn blew up")]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 2, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(transcript.includes("Status: FAILED"));
  assert.ok(transcript.includes("spawn blew up"));
});

test("runDebate: abort signal → ABORTED", async () => {
  const ctrl = new AbortController();
  ctrl.abort();
  const subs = scriptedSubagents(["x"]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 6, parent: {}, signal: ctrl.signal, subagents: subs,
  });
  assert.ok(transcript.includes("Status: ABORTED"));
  assert.equal(subs.calls.length, 0);
});

test("runDebate: empty output falls back to (stopReason)", async () => {
  const subs = scriptedSubagents(["   "], ["done\nVerdict: AGREE"]);
  const { transcript } = await runDebate({
    agentA: roster("coder"), agentB: roster("reviewer"),
    task: "t", maxTurns: 2, parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(transcript.includes("(completed)"));
});

// === parallel fan-out ===

test("runParallel: joins per-agent sections with ---", async () => {
  const subs = scriptedSubagents(["result A"], ["result B"]);
  const out = await runParallel({
    tasks: [
      { agent: roster("researcher"), brief: "brief A" },
      { agent: roster("critic"), brief: "brief B" },
    ],
    parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(out.includes("## researcher\n\nresult A"));
  assert.ok(out.includes("## critic\n\nresult B"));
  assert.ok(out.includes("---"));
  // briefs passed through as the prompt text block
  assert.equal(subs.calls[0].prompt[0].text, "brief A");
  assert.equal(subs.calls[0].maxDepth, SPAWN_MAX_DEPTH, "parallel spawn must set maxDepth");
  assert.equal(subs.calls[1].maxDepth, SPAWN_MAX_DEPTH, "parallel spawn must set maxDepth");
});

test("runParallel: per-task failure is fail-open, others still reported", async () => {
  const subs = scriptedSubagents(["ok A"], [new Error("boom B")]);
  const out = await runParallel({
    tasks: [
      { agent: roster("researcher"), brief: "A" },
      { agent: roster("critic"), brief: "B" },
    ],
    parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(out.includes("## researcher\n\nok A"));
  assert.ok(out.includes("## critic\n\n⚠️ Task failed: boom B"));
});

test("runParallel: runs concurrently (no barrier between tasks)", async () => {
  let inFlight = 0;
  let maxInFlight = 0;
  const subs = {
    async start() {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 20));
      inFlight--;
      return { result: Promise.resolve({ output: "done", stopReason: "completed" }), dispose() {} };
    },
  };
  await runParallel({
    tasks: [0, 1, 2].map((i) => ({ agent: roster("researcher"), brief: `b${i}` })),
    parent: {}, signal: new AbortController().signal, subagents: subs,
  });
  assert.ok(maxInFlight >= 2, `expected concurrent starts, max in flight was ${maxInFlight}`);
});
