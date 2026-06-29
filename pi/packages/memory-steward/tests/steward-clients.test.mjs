import assert from "node:assert/strict";
import test from "node:test";

import { createEmbedderClient } from "../dist/src/embedder.js";
import { createStewardClient, parseStewardMd } from "../dist/src/steward.js";

test("parses steward frontmatter and prompt sections", () => {
	const parsed = parseStewardMd(`---
url: http://127.0.0.1:18091
temperature: 0.25 # comment
model: "steward"
---

## rewrite

Rewrite prompt.

## judge

Judge prompt.
`);

	assert.deepEqual(parsed.config, {
		url: "http://127.0.0.1:18091",
		temperature: 0.25,
		model: "steward",
	});
	assert.equal(parsed.prompts.rewrite, "Rewrite prompt.");
	assert.equal(parsed.prompts.judge, "Judge prompt.");
});

test("steward client strips thinking tags and handles NONE", async () => {
	const calls = [];
	const client = createStewardClient({
		baseURL: "http://local",
		model: "steward",
		rewritePrompt: "rewrite",
		judgePrompt: "judge",
		fetch: async (_url, init) => {
			calls.push(JSON.parse(init.body));
			return {
				ok: true,
				async json() {
					return {
						choices: [
							{ message: { content: "<think>hidden</think>Rust edition" } },
						],
					};
				},
			};
		},
	});

	assert.equal(await client.rewriteQuery("debug rust"), "Rust edition");
	assert.equal(calls[0].model, "steward");
});

test("steward client parses save judgments", async () => {
	const client = createStewardClient({
		baseURL: "http://local",
		model: "steward",
		rewritePrompt: "rewrite",
		judgePrompt: "judge",
		fetch: async () => ({
			ok: true,
			async json() {
				return {
					choices: [
						{
							message: {
								content:
									'```json\n{ "save": { "content": "Use uv, not pip" } }\n```',
							},
						},
					],
				};
			},
		}),
	});

	assert.equal(
		await client.judgeSave("use uv", "Switching to uv"),
		"Use uv, not pip",
	);
});

test("embedder client normalizes vector dimensions", async () => {
	const calls = [];
	const client = createEmbedderClient({
		baseURL: "http://local",
		dimensions: 3,
		fetch: async (_url, init) => {
			calls.push(JSON.parse(init.body));
			return {
				ok: true,
				async json() {
					return { data: [{ embedding: [1, 2, 3, 4] }] };
				},
			};
		},
	});

	assert.deepEqual(await client.embed("hello"), [1, 2, 3]);
	assert.equal(calls[0].model, "granite-embedding-311m");
	assert.equal(calls[0].input, "hello");
});

test("embedder client pads short vectors and preserves batch ordering", async () => {
	const calls = [];
	const client = createEmbedderClient({
		baseURL: "http://local",
		model: "custom-embedder",
		dimensions: 3,
		fetch: async (_url, init) => {
			calls.push(JSON.parse(init.body));
			return {
				ok: true,
				async json() {
					return { data: [{ embedding: [1] }, { embedding: [2, 3, 4] }] };
				},
			};
		},
	});

	assert.deepEqual(await client.embedBatch(["a", "b"]), [
		[1, 0, 0],
		[2, 3, 4],
	]);
	assert.equal(calls[0].model, "custom-embedder");
	assert.deepEqual(calls[0].input, ["a", "b"]);
});
