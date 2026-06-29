import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createMemoryStore } from "../dist/src/memory-store.js";

function tempDir() {
	const dir = mkdtempSync(join(tmpdir(), "ergon-memory-store-"));
	return {
		dir,
		cleanup() {
			rmSync(dir, { recursive: true, force: true });
		},
	};
}

test("saves, deduplicates, recalls, lists, and deletes memories", () => {
	const project = tempDir();
	try {
		const store = createMemoryStore({
			dbPath: join(project.dir, "memory.sqlite"),
			embeddingDimensions: 3,
		});

		assert.equal(store.isAvailable(), true);
		store.save("New Rust projects default to edition 2024", [0.1, 0.2, 0.3]);
		store.save("New Rust projects default to edition 2024", [0.1, 0.2, 0.3]);
		store.save(
			"Use uv instead of pip for Python dependencies",
			[0.9, 0.1, 0.1],
		);

		const listed = store.list();
		assert.equal(listed.length, 2);

		const hits = store.recall([0.1, 0.2, 0.3], 2, "Rust edition");
		assert.ok(hits.some((hit) => hit.content.includes("Rust")));

		store.delete(listed[0].id);
		assert.equal(store.list().length, 1);

		store.close();
		assert.equal(store.isAvailable(), false);
	} finally {
		project.cleanup();
	}
});

test("returns a no-op store for incompatible vector dimensions", () => {
	const project = tempDir();
	try {
		const dbPath = join(project.dir, "memory.sqlite");
		const store = createMemoryStore({ dbPath, embeddingDimensions: 3 });
		assert.equal(store.isAvailable(), true);
		store.close();

		const originalConsoleError = console.error;
		console.error = () => {};
		try {
			const mismatched = createMemoryStore({ dbPath, embeddingDimensions: 4 });
			assert.equal(mismatched.isAvailable(), false);
			assert.deepEqual(mismatched.recall([0, 0, 0, 0]), []);
		} finally {
			console.error = originalConsoleError;
		}
	} finally {
		project.cleanup();
	}
});
