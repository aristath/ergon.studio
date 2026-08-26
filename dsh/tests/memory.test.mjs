// openmemory HTTP client behavior (fetch injected).

import { test } from "node:test";
import assert from "node:assert/strict";

import { createMemoryClient } from "../dist/memory.js";

const BASE = "http://mem.test";

function jsonFetch(routes, calls = []) {
  return async (url, init) => {
    const u = String(url);
    calls.push({ url: u, body: JSON.parse(init.body) });
    for (const [suffix, handler] of routes) {
      if (u.endsWith(suffix)) return handler(JSON.parse(init.body), u);
    }
    return new Response("not found", { status: 404 });
  };
}

const okJson = (obj) =>
  new Response(JSON.stringify(obj), { status: 200, headers: { "Content-Type": "application/json" } });

test("recall: returns id/content/score from matches", async () => {
  const calls = [];
  const c = createMemoryClient({
    baseURL: BASE,
    fetch: jsonFetch([["/memory/query", (body) => okJson({ query: body.query, matches: [
      { id: "m1", content: "note one", score: 0.9, salience: 1, sectors: [] },
      { id: "m2", content: "note two", score: 0.5 },
      { content: "no id" },
      { id: "m3", content: "", score: 0.1 },
    ] })]], calls),
  });
  const items = await c.recall("ports", 3);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].body.query, "ports");
  assert.equal(calls[0].body.k, 3);
  assert.deepEqual(items, [
    { id: "m1", content: "note one", score: 0.9 },
    { id: "m2", content: "note two", score: 0.5 },
    { id: "", content: "no id", score: undefined },
  ]);
});

test("recall: accepts legacy `memories` key", async () => {
  const c = createMemoryClient({
    baseURL: BASE,
    fetch: jsonFetch([["/memory/query", () => okJson({ memories: [{ id: "a", content: "legacy" }] })]]),
  });
  assert.deepEqual(await c.recall("q"), [{ id: "a", content: "legacy", score: undefined }]);
});

test("recall: empty query short-circuits without fetch", async () => {
  let touched = false;
  const c = createMemoryClient({ baseURL: BASE, fetch: () => { touched = true; } });
  assert.deepEqual(await c.recall(""), []);
  assert.deepEqual(await c.recall("   "), []);
  assert.equal(touched, false);
});

test("recall: non-ok and network failure → [] (fail open)", async () => {
  const c1 = createMemoryClient({
    baseURL: BASE,
    fetch: () => new Response("err", { status: 500 }),
  });
  assert.deepEqual(await c1.recall("q"), []);
  const c2 = createMemoryClient({ baseURL: BASE, fetch: () => { throw new Error("down"); } });
  assert.deepEqual(await c2.recall("q"), []);
});

test("recall: default limit applied when omitted", async () => {
  const calls = [];
  const c = createMemoryClient({
    baseURL: BASE,
    defaultLimit: 7,
    fetch: jsonFetch([["/memory/query", (b) => okJson({ matches: [] })]], calls),
  });
  await c.recall("q");
  assert.equal(calls[0].body.k, 7);
});

test("recall: userID filter applied", async () => {
  const calls = [];
  const c = createMemoryClient({
    baseURL: BASE,
    userID: "alice",
    fetch: jsonFetch([["/memory/query", () => okJson({ matches: [] })]], calls),
  });
  await c.recall("q");
  assert.deepEqual(calls[0].body.filters, { user_id: "alice" });
});

test("save: posts trimmed content; errors swallowed", async () => {
  const calls = [];
  const c = createMemoryClient({ baseURL: BASE, fetch: jsonFetch([["/memory/add", () => okJson({})]], calls) });
  await c.save("  durable fact  ");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, `${BASE}/memory/add`);
  assert.equal(calls[0].body.content, "durable fact");

  // empty content: no fetch
  const c2calls = [];
  const c2 = createMemoryClient({ baseURL: BASE, fetch: jsonFetch([], c2calls) });
  await c2.save("");
  await c2.save("   ");
  assert.equal(c2calls.length, 0);

  // network error: swallowed (fail open)
  const c3 = createMemoryClient({ baseURL: BASE, fetch: () => { throw new Error("down"); } });
  await c3.save("x"); // must not throw
});

test("save: non-ok response swallowed", async () => {
  const c = createMemoryClient({
    baseURL: BASE,
    fetch: () => new Response("err", { status: 503 }),
  });
  await c.save("x"); // must not throw
});

test("trailing slash on baseURL normalized", async () => {
  const calls = [];
  const c = createMemoryClient({
    baseURL: "http://mem.test/",
    fetch: jsonFetch([["/memory/query", () => okJson({ matches: [] })]], calls),
  });
  await c.recall("q");
  assert.equal(calls[0].url, "http://mem.test/memory/query");
});
