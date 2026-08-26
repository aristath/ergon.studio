// Plugin apply() against a mocked cordis ctx: export shape, effect/tool
// registration, recall flow, save flow, and the three tool executes.

import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, existsSync, readFileSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import * as plugin from "../dist/index.js";
import { _resetStewardDefinitionCacheForTests } from "../dist/steward.js";

const DEFINITION = {
  config: { url: "http://127.0.0.1:18091", model: "steward-test", temperature: 0.3 },
  prompts: { rewrite: "REWRITE", judge: "JUDGE" },
};

// === mock ctx ===

function makeCtx() {
  const listeners = new Map();
  const effects = [];
  const registered = [];
  const contexts = [];
  const warnings = [];
  const ctx = {
    listeners,
    effects,
    _registered: registered,
    contexts,
    warnings,
    on(event, fn) {
      if (!listeners.has(event)) listeners.set(event, []);
      listeners.get(event).push(fn);
    },
    effect(fn, label) {
      effects.push(label);
      fn();
    },
    logger: { warn: (m) => warnings.push(String(m)), info: () => {}, error: () => {}, debug: () => {} },
    tools: { register: (t) => registered.push(t) },
    systemPrompt: { context: (spec) => contexts.push(spec) },
    subagents: null, // set per test
    agents: { get: () => undefined },
    emit(event, ...args) {
      for (const fn of listeners.get(event) ?? []) fn(...args);
    },
  };
  return ctx;
}

function toolsByName(ctx) {
  return new Map(ctx._registered.map((t) => [t.name, t]));
}

async function waitFor(cond, { timeoutMs = 3000, everyMs = 5 } = {}) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const v = await cond();
    if (v) return v;
    await new Promise((r) => setTimeout(r, everyMs));
  }
  throw new Error("waitFor timed out");
}

// === HTTP stub (global fetch) ===

let fetchCalls;
let fetchImpl;
const realFetch = globalThis.fetch;
let fakeHome;
const realDshHome = process.env.DSH_HOME;

beforeEach(() => {
  fakeHome = mkdtempSync(join(tmpdir(), "ergon-dshhome-"));
  process.env.DSH_HOME = fakeHome;
  fetchCalls = [];
  fetchImpl = async (url, init) => {
    const u = String(url);
    const body = init?.body ? JSON.parse(init.body) : undefined;
    fetchCalls.push({ url: u, body });
    if (u.endsWith("/v1/chat/completions")) {
      return chat(body);
    }
    if (u.endsWith("/memory/query")) {
      return new Response(JSON.stringify({ matches: [
        { id: "mem-1", content: "the port is 18091", score: 0.9 },
      ] }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    if (u.endsWith("/memory/add")) {
      return new Response("{}", { status: 200 });
    }
    return new Response("nf", { status: 404 });
  };
  // route chat completions by which prompt is in the system message
  function chat(body) {
    const sys = body?.messages?.[0]?.content ?? "";
    if (sys === "REWRITE") {
      return new Response(JSON.stringify({
        choices: [{ message: { content: "<think>rewrite reasoning</think>\nthe port for the memory steward" } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    if (sys === "JUDGE") {
      return new Response(JSON.stringify({
        choices: [{ message: { content: '{"save": {"content": "the steward runs on 18091"}}' } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response("nf", { status: 404 });
  }
  globalThis.fetch = fetchImpl;
  _resetStewardDefinitionCacheForTests(DEFINITION);
});

afterEach(() => {
  globalThis.fetch = realFetch;
  if (realDshHome === undefined) delete process.env.DSH_HOME;
  else process.env.DSH_HOME = realDshHome;
  try { rmSync(fakeHome, { recursive: true, force: true }); } catch {}
});

const CONFIG = {
  stewardUrl: "http://steward.test",
  stewardModel: "steward-test",
  memoryUrl: "http://mem.test",
  recallTimeoutMs: 2000,
  recallLimit: 5,
};

function mount(subagents) {
  const ctx = makeCtx();
  ctx.subagents = subagents;
  plugin.apply(ctx, CONFIG);
  return ctx;
}

// === export shape ===

test("plugin exports: name, inject, Config, apply", () => {
  assert.equal(plugin.name, "ergon");
  assert.deepEqual(plugin.inject, ["tools", "subagents", "systemPrompt", "agents"]);
  assert.equal(typeof plugin.Config, "function");
  assert.ok(plugin.Config["~standard"], "Config must be a schemastery schema");
  assert.equal(typeof plugin.Config["~standard"].validate, "function");
  assert.equal(typeof plugin.apply, "function");
});

test("apply: registers listeners, effects, and the three tools", () => {
  const ctx = mount({});
  assert.ok(ctx.listeners.has("agent/inbox/inserted"));
  assert.ok(ctx.listeners.has("session/event"));
  const names = [...toolsByName(ctx).keys()].sort();
  assert.deepEqual(names, ["debate", "memory_search", "run_parallel"]);
  // dynamic contexts: memory + scratchpad
  const ctxNames = ctx.contexts.map((c) => c.name).sort();
  assert.deepEqual(ctxNames, ["ergon:memory", "ergon:scratchpad"]);
});

// === self-install (preset + skills into the DSH home) ===

test("self-install: preset + skills installed into a fresh DSH home", () => {
  mount({});
  const preset = join(fakeHome, ".agent-presets", "ergon", "agent.cordis.yml");
  assert.ok(existsSync(preset), `preset should be installed at ${preset}`);
  assert.ok(existsSync(join(fakeHome, ".agent-presets", "ergon", "preset.yml")));
  const skills = join(fakeHome, "skills");
  assert.ok(existsSync(join(skills, "handoff", "SKILL.md")));
  assert.ok(existsSync(join(skills, "scratchpad", "SKILL.md")));
});

test("self-install: idempotent, and an existing user preset is never overwritten", () => {
  mount({});
  const preset = join(fakeHome, ".agent-presets", "ergon", "agent.cordis.yml");
  writeFileSync(preset, "# user-edited preset\n");
  mount({}); // second mount must keep the user copy
  assert.equal(readFileSync(preset, "utf8"), "# user-edited preset\n");
});

test("self-install: package without preset degrades to a warning (fail-open)", () => {
  // Simulate an incomplete package: hide the bundled preset file, mount, and
  // expect a warning with no crash and no partial install.
  const presetFile = join(dirname(fileURLToPath(import.meta.url)), "..", "presets", "ergon", "agent.cordis.yml");
  renameSync(presetFile, `${presetFile}.hidden`);
  try {
    const ctx = mount({});
    assert.ok(!existsSync(join(fakeHome, ".agent-presets", "ergon")));
    assert.ok(ctx.warnings.some((w) => w.includes("preset not found in package")));
  } finally {
    renameSync(`${presetFile}.hidden`, presetFile);
  }
});

// === recall flow ===

const AGENT = {};

test("recall: user message → rewrite → search → context block visible", async () => {
  const ctx = mount({});
  const memoryCtx = ctx.contexts.find((c) => c.name === "ergon:memory");
  assert.equal(memoryCtx.text({ agent: AGENT }), "", "no recall before any user message");

  ctx.emit("agent/inbox/inserted", {
    agent: AGENT,
    message: { content: [{ type: "text", text: "what port does the memory steward use?" }], source: { kind: "user" } },
  });

  const block = await waitFor(() => {
    const t = memoryCtx.text({ agent: AGENT });
    return t || null;
  });
  assert.ok(block.includes("## Relevant prior notes (from memory steward)"));
  assert.ok(block.includes("the port is 18091"));
  // the rewrite stage went to the steward URL with the configured model
  const rewrite = fetchCalls.find((c) => c.url === "http://steward.test/v1/chat/completions");
  assert.ok(rewrite, "steward rewrite call expected");
  assert.equal(rewrite.body.model, "steward-test");
  const query = fetchCalls.find((c) => c.url === "http://mem.test/memory/query");
  assert.equal(query.body.query, "the port for the memory steward");
  assert.equal(query.body.k, 5);
});

test("recall: agent without a message source is treated as user (no kind → allowed)", async () => {
  const ctx = mount({});
  const memoryCtx = ctx.contexts.find((c) => c.name === "ergon:memory");
  ctx.emit("agent/inbox/inserted", {
    agent: AGENT,
    message: { content: [{ type: "text", text: "hello there" }] },
  });
  const block = await waitFor(() => memoryCtx.text({ agent: AGENT }) || null);
  assert.ok(block);
});

test("recall: non-user source is ignored", async () => {
  const ctx = mount({});
  const memoryCtx = ctx.contexts.find((c) => c.name === "ergon:memory");
  ctx.emit("agent/inbox/inserted", {
    agent: AGENT,
    message: { content: [{ type: "text", text: "tool output noise" }], source: { kind: "tool" } },
  });
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(memoryCtx.text({ agent: AGENT }), "");
  assert.equal(fetchCalls.filter((c) => c.url.includes("steward.test")).length, 0);
});

test("recall: identical re-queued message is deduped (no second rewrite)", async () => {
  const ctx = mount({});
  const msg = { content: [{ type: "text", text: "same question" }], source: { kind: "user" } };
  ctx.emit("agent/inbox/inserted", { agent: AGENT, message: msg });
  await waitFor(() => ctx.contexts.find((c) => c.name === "ergon:memory").text({ agent: AGENT }) || null);
  const n1 = fetchCalls.filter((c) => c.url.includes("steward.test")).length;
  ctx.emit("agent/inbox/inserted", { agent: AGENT, message: msg });
  await new Promise((r) => setTimeout(r, 50));
  const n2 = fetchCalls.filter((c) => c.url.includes("steward.test")).length;
  assert.equal(n1, 1);
  assert.equal(n2, 1);
});

test("recall: a newer message supersedes a slow older one (staleness guard)", async () => {
  // Gate the first rewrite so the second user message lands first; make both
  // the rewrite (echoes the user text) and the search (echoes the query)
  // reflect the trigger text so the installed block is attributable.
  let releaseFirst;
  const firstGate = new Promise((r) => { releaseFirst = r; });
  let firstRewriteSeen = false;
  const gateFetch = async (url, init) => {
    const u = String(url);
    const body = init?.body ? JSON.parse(init.body) : undefined;
    fetchCalls.push({ url: u, body });
    if (u.endsWith("/v1/chat/completions")) {
      if (!firstRewriteSeen) {
        firstRewriteSeen = true;
        await firstGate;
      }
      const userMsg = body?.messages?.[1]?.content ?? "?";
      return new Response(JSON.stringify({
        choices: [{ message: { content: userMsg } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    if (u.endsWith("/memory/query")) {
      return new Response(JSON.stringify({ matches: [
        { id: "m", content: `match-for-${body.query}`, score: 0.9 },
      ] }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response("nf", { status: 404 });
  };
  const ctx = makeCtx();
  ctx.subagents = {};
  globalThis.fetch = gateFetch;
  plugin.apply(ctx, CONFIG);

  const memoryCtx = ctx.contexts.find((c) => c.name === "ergon:memory");
  ctx.emit("agent/inbox/inserted", {
    agent: AGENT,
    message: { content: [{ type: "text", text: "slow first question" }], source: { kind: "user" } },
  });
  ctx.emit("agent/inbox/inserted", {
    agent: AGENT,
    message: { content: [{ type: "text", text: "fast second question" }], source: { kind: "user" } },
  });
  await new Promise((r) => setTimeout(r, 20));
  releaseFirst();
  // Let both recalls settle.
  await new Promise((r) => setTimeout(r, 100));
  const block = memoryCtx.text({ agent: AGENT });
  // The stale recall's result must not be installed: the visible block comes
  // from the second (current) trigger text only.
  assert.ok(block.includes("match-for-fast second question"), `block was: ${block}`);
  assert.ok(!block.includes("match-for-slow first question"), `stale block leaked: ${block}`);
});

// === save flow ===

function makeSession(id, turns) {
  const events = [];
  let seq = 0;
  for (const t of turns) {
    events.push({ type: "turn/start", data: { turn: t.turn }, seq: ++seq });
    events.push({
      type: "user/message",
      data: { content: [{ type: "text", text: t.user }], source: { kind: "user" } },
      seq: ++seq,
    });
    events.push({
      type: "assistant/message",
      data: { turn: t.turn, step: 1, message: { content: [{ type: "text", text: t.assistant }] } },
      seq: ++seq,
    });
  }
  return { id, events };
}

test("save: turn/end(completed) → judge → /memory/add with judged content", async () => {
  const ctx = mount({});
  const session = makeSession("s1", [{ turn: 1, user: "remember: steward port is 18091", assistant: "Noted, saved." }]);
  ctx.agents.get = (id) => (id === "s1" ? { session } : undefined);

  ctx.emit("session/event", session, { type: "turn/start", data: { turn: 1 }, seq: 0 });
  ctx.emit("session/event", session, { type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } });

  await waitFor(() => fetchCalls.find((c) => c.url === "http://mem.test/memory/add"));
  const add = fetchCalls.find((c) => c.url === "http://mem.test/memory/add");
  assert.equal(add.body.content, "the steward runs on 18091");
  // judge was called with the exchange
  const judge = fetchCalls.filter((c) => c.url.endsWith("/v1/chat/completions")).at(-1);
  assert.ok(judge.body.messages[1].content.includes("User: remember: steward port is 18091"));
  assert.ok(judge.body.messages[1].content.includes("Assistant: Noted, saved."));
});

test("save: non-completed turn/end does not judge", async () => {
  const ctx = mount({});
  const session = makeSession("s1", [{ turn: 1, user: "u", assistant: "a" }]);
  ctx.agents.get = (id) => (id === "s1" ? { session } : undefined);
  ctx.emit("session/event", session, { type: "turn/end", data: { turn: 1, reason: { kind: "blocked" } } });
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(fetchCalls.filter((c) => c.url.includes("mem.test")).length, 0);
});

test("save: same turn is not judged twice", async () => {
  const ctx = mount({});
  const session = makeSession("s1", [{ turn: 1, user: "u", assistant: "a" }]);
  ctx.agents.get = (id) => (id === "s1" ? { session } : undefined);
  const end = { type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } };
  ctx.emit("session/event", session, end);
  await waitFor(() => fetchCalls.find((c) => c.url === "http://mem.test/memory/add"));
  ctx.emit("session/event", session, end);
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(fetchCalls.filter((c) => c.url === "http://mem.test/memory/add").length, 1);
});

test("save: session with no matching agent is ignored", async () => {
  const ctx = mount({});
  const session = makeSession("ghost", [{ turn: 1, user: "u", assistant: "a" }]);
  ctx.agents.get = () => undefined;
  ctx.emit("session/event", session, { type: "turn/end", data: { turn: 1, reason: { kind: "completed" } } });
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(fetchCalls.filter((c) => c.url.includes("steward.test")).length, 0);
});

// === scratchpad context ===

test("scratchpad context: re-read per assembly, cwd-scoped", () => {
  const cwd = mkdtempSync(join(tmpdir(), "ergon-cwd-"));
  try {
    const ctx = mount({});
    const padCtx = ctx.contexts.find((c) => c.name === "ergon:scratchpad");
    assert.ok(padCtx.text({ agent: { session: { header: { cwd } } } }).includes("No scratchpad yet"));
    mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(join(cwd, ".ergon.studio", "scratchpad.md"), "## Notes\n- discovered thing\n");
    const block = padCtx.text({ agent: { session: { header: { cwd } } } });
    assert.ok(block.includes("discovered thing"));
    // missing cwd → empty string (no context, filtered out by the projection)
    assert.equal(padCtx.text({ agent: {} }), "");
    assert.equal(padCtx.text({}), "");
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

// === tools ===

function fakeSubagents(script) {
  const calls = [];
  let i = 0;
  return {
    calls,
    async start(provider, request) {
      calls.push({ provider, request });
      const spec = script[i++ % script.length];
      if (spec instanceof Error) throw spec;
      return {
        result: Promise.resolve({ output: typeof spec === "function" ? spec(request) : spec, stopReason: "completed" }),
        dispose() {},
      };
    },
  };
}

test("debate tool: runs two roster agents and returns the transcript", async () => {
  const subs = fakeSubagents(["pass 1\nVerdict: CONTINUE", "agree\nVerdict: AGREE"]);
  const ctx = mount(subs);
  const debate = toolsByName(ctx).get("debate");
  const out = await debate.execute(
    { agent_a: "coder", agent_b: "reviewer", task: "fix it", max_turns: 4 },
    { agent: {}, signal: new AbortController().signal },
  );
  assert.ok(out.transcript.includes("Status: AGREE"));
  assert.equal(subs.calls.length, 2);
  assert.equal(subs.calls[0].provider, "spawn");
  assert.match(subs.calls[0].request.label, /coder/);
  assert.ok(subs.calls[0].request.persona.length > 0, "persona should be inlined from the roster");
  // specialist deny list applied
  assert.ok(subs.calls[0].request.toolFilter.deny.includes("ask_user_question"));
});

test("debate tool: unknown agent → FAILED transcript naming the roster", async () => {
  const subs = fakeSubagents(["x"]);
  const ctx = mount(subs);
  const debate = toolsByName(ctx).get("debate");
  const out = await debate.execute(
    { agent_a: "ghost", agent_b: "coder", task: "t" },
    { agent: {}, signal: new AbortController().signal },
  );
  assert.ok(out.transcript.includes("Status: FAILED"));
  assert.ok(out.transcript.includes("scout"));
  assert.equal(subs.calls.length, 0);
});

test("run_parallel tool: combines per-agent output", async () => {
  const subs = fakeSubagents(["out-a", "out-b"]);
  const ctx = mount(subs);
  const rp = toolsByName(ctx).get("run_parallel");
  const out = await rp.execute(
    { tasks: [
      { agent: "researcher", brief: "brief A" },
      { agent: "critic", brief: "brief B" },
    ] },
    { agent: {}, signal: new AbortController().signal },
  );
  assert.ok(out.output.includes("## researcher\n\nout-a"));
  assert.ok(out.output.includes("## critic\n\nout-b"));
  assert.ok(out.output.includes("---"));
});

test("run_parallel tool: invalid tasks → friendly message", async () => {
  const subs = fakeSubagents(["x"]);
  const ctx = mount(subs);
  const rp = toolsByName(ctx).get("run_parallel");
  const out = await rp.execute(
    { tasks: [{ agent: "ghost", brief: "b" }] },
    { agent: {}, signal: new AbortController().signal },
  );
  assert.ok(out.output.includes("No valid tasks"));
});

test("memory_search tool: queries openmemory and counts results", async () => {
  const ctx = mount({});
  const ms = toolsByName(ctx).get("memory_search");
  const out = await ms.execute({ query: "steward port", limit: 3 }, { signal: new AbortController().signal });
  assert.equal(out.count, 1);
  assert.equal(out.results[0].content, "the port is 18091");
  const call = fetchCalls.find((c) => c.url === "http://mem.test/memory/query");
  assert.equal(call.body.query, "steward port");
  assert.equal(call.body.k, 3);
});

test("tool renders: return content blocks", () => {
  const ctx = mount({});
  for (const [name, value] of [
    ["debate", { transcript: "T" }],
    ["run_parallel", { output: "O" }],
    ["memory_search", { count: 0, results: [] }],
  ]) {
    const tool = toolsByName(ctx).get(name);
    const blocks = tool.output.render({}, value);
    assert.ok(Array.isArray(blocks), `${name} render must return ContentBlock[]`);
    assert.ok(blocks.every((b) => b.type === "text" && typeof b.text === "string"), `${name} block shape`);
  }
});
