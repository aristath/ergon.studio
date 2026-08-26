// Steward definition parsing + client behavior (all fetch injected).

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  parseStewardMd,
  createStewardClient,
  _resetStewardDefinitionCacheForTests,
} from "../dist/steward.js";

const DEFINITION = {
  config: { url: "http://127.0.0.1:18091", model: "ergon-studio-memory-steward", temperature: 0.3 },
  prompts: { rewrite: "REWRITE_PROMPT", judge: "JUDGE_PROMPT" },
};

test("parseStewardMd: frontmatter config + ## sections", () => {
  const def = parseStewardMd(
    "---\nurl: http://x:1 # comment\nmodel: 'my-model'\ntemperature: 0.3\n---\n\n## Rewrite\nDo the rewrite.\n\n## Judge\nReturn JSON.\n",
  );
  assert.equal(def.config.url, "http://x:1");
  assert.equal(def.config.model, "my-model");
  assert.equal(def.config.temperature, 0.3);
  assert.equal(def.prompts.rewrite, "Do the rewrite.");
  assert.equal(def.prompts.judge, "Return JSON.");
});

test("parseStewardMd: throws without frontmatter", () => {
  assert.throws(() => parseStewardMd("no frontmatter here"));
});

function chatFetch(responseText, { calls = [], ok = true } = {}) {
  return async (url, init) => {
    calls.push({ url: String(url), body: JSON.parse(init.body) });
    if (!ok) return new Response("boom", { status: 500 });
    return new Response(
      JSON.stringify({ choices: [{ message: { content: responseText } }] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };
}

test("createStewardClient: available with injected definition", () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch("x") });
  assert.equal(c.available, true);
});

test("createStewardClient: unavailable without prompts (fail open)", async () => {
  // Inject a definition missing both prompts: the client must report
  // unavailable and no-op every call without touching fetch.
  _resetStewardDefinitionCacheForTests({ config: {}, prompts: {} });
  let touched = false;
  const c = createStewardClient({ fetch: () => { touched = true; throw new Error("must not be called"); } });
  assert.equal(c.available, false);
  const [q, s] = await Promise.all([c.rewriteQuery("q"), c.judgeSave("u", "a")]);
  assert.equal(q, null);
  assert.equal(s, null);
  assert.equal(touched, false);
});

test("rewriteQuery: strips think block, passes model + temperature", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const calls = [];
  const c = createStewardClient({
    fetch: chatFetch(" <think>\nhmm\n</think>\n\nmy search query", { calls }),
  });
  const q = await c.rewriteQuery("what did we decide about ports again?");
  assert.equal(q, "my search query");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://127.0.0.1:18091/v1/chat/completions");
  assert.equal(calls[0].body.model, "ergon-studio-memory-steward");
  assert.equal(calls[0].body.temperature, 0.3);
  assert.equal(calls[0].body.messages[0].role, "system");
  assert.equal(calls[0].body.messages[0].content, "REWRITE_PROMPT");
  assert.equal(calls[0].body.messages[1].content, "what did we decide about ports again?");
});

test("rewriteQuery: NONE → null", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch("NONE") });
  assert.equal(await c.rewriteQuery("random chitchat"), null);
});

test("rewriteQuery: fetch failure → null (fail open)", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch("x", { ok: false }) });
  assert.equal(await c.rewriteQuery("q"), null);
  const c2 = createStewardClient({ fetch: () => { throw new Error("down"); } });
  assert.equal(await c2.rewriteQuery("q"), null);
});

test("judgeSave: parses {save:{content}}", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch('{"save": {"content": "user prefers pnpm"}}') });
  assert.equal(await c.judgeSave("use pnpm next time", "ok"), "user prefers pnpm");
});

test("judgeSave: {save:false} → null", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch('{"save": false}') });
  assert.equal(await c.judgeSave("hi", "hello"), null);
});

test("judgeSave: code-fenced JSON accepted", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch('```json\n{"save": {"content": "x"}}\n```') });
  assert.equal(await c.judgeSave("u", "a"), "x");
});

test("judgeSave: malformed JSON → null (fail open)", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const c = createStewardClient({ fetch: chatFetch("not json at all") });
  assert.equal(await c.judgeSave("u", "a"), null);
});

test("judgeSave: sends the exchange to the judge prompt", async () => {
  _resetStewardDefinitionCacheForTests(DEFINITION);
  const calls = [];
  const c = createStewardClient({ fetch: chatFetch('{"save": false}', { calls }) });
  await c.judgeSave("user line", "assistant line");
  assert.equal(calls[0].body.messages[0].content, "JUDGE_PROMPT");
  assert.equal(calls[0].body.messages[1].content, "User: user line\nAssistant: assistant line");
});
