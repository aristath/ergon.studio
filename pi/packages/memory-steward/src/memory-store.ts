// Memory store: SQLite + sqlite-vec + FTS5.
//
// Replaces openmemory-js. Stores memories with their 768-dim embedding
// vectors and supports vector similarity search + keyword (FTS) hybrid search.
//
// Single file: ~/.local/share/ergon-memory-steward/memory.sqlite (configurable).

import { createHash } from "node:crypto";
import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import Database, { type Statement } from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";

// ── Types ────────────────────────────────────────────────────────────────────

export interface MemoryItem {
	id: string;
	content: string;
	score?: number; // Combined score (lower = closer)
	createdAt?: number;
	accessCount?: number;
}

export interface MemoryStore {
	/**
	 * Search by vector similarity (optionally combined with keyword FTS).
	 * Returns top-k results.
	 */
	recall(queryVector: number[], k?: number, queryText?: string): MemoryItem[];

	/** Save a memory with its embedding vector. Dedup by content hash. */
	save(content: string, vector: number[]): void;

	/** Delete a memory by ID. */
	delete(id: string): void;

	/** List all memories (for debugging/inspection). */
	list(limit?: number): MemoryItem[];

	/** Close the database connection. */
	close(): void;

	/** Whether this store has a real, usable database behind it. */
	isAvailable(): boolean;
}

export interface MemoryStoreOptions {
	dbPath?: string;
	embeddingDimensions?: number;
}

// ── Defaults ─────────────────────────────────────────────────────────────────

export const DEFAULT_DB_PATH = resolve(
	homedir(),
	".local/share/ergon-memory-steward/memory.sqlite",
);
const DEFAULT_EMBEDDING_DIMENSIONS = 768;

// ── Schema ───────────────────────────────────────────────────────────────────

function schemaSql(embeddingDimensions: number): string {
	return `
	  -- Vector search table (sqlite-vec vec0 virtual table)
	  CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
	    embedding float[${embeddingDimensions}]
	  );

	  -- Memory entries
  CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    vector_rowid INTEGER,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER,
    access_count INTEGER DEFAULT 0,
    tags TEXT
  );

  -- FTS index for keyword search
  CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid'
  );

  -- Triggers to keep FTS in sync
  CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
  END;

  CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.rowid, old.content);
  END;

  CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
  END;

	  -- Index for fast hash lookups
	  CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(hash);

	  -- Store schema-sensitive settings so incompatible vector dimensions fail at startup.
	  CREATE TABLE IF NOT EXISTS memory_meta (
	    key TEXT PRIMARY KEY,
	    value TEXT NOT NULL
	  );
	`;
}

// ── Prepared statements (initialized on open) ────────────────────────────────

function normalizeEmbeddingDimensions(value: number): number {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(
			`embeddingDimensions must be a positive integer, got ${value}`,
		);
	}
	return value;
}

function hasTable(db: ReturnType<typeof Database>, tableName: string): boolean {
	const row = db
		.prepare(
			`SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?`,
		)
		.get(tableName);
	return row !== undefined;
}

function validateEmbeddingDimensions(
	db: ReturnType<typeof Database>,
	embeddingDimensions: number,
	hadVectorTable: boolean,
): void {
	const getMeta = db.prepare(
		`SELECT value FROM memory_meta WHERE key = 'embedding_dimensions'`,
	);
	const row = getMeta.get() as { value: string } | undefined;

	if (row) {
		const storedDimensions = Number(row.value);
		if (storedDimensions !== embeddingDimensions) {
			throw new Error(
				`memory DB embedding dimension mismatch: database uses ${storedDimensions}, config wants ${embeddingDimensions}`,
			);
		}
		return;
	}

	const storedDimensions = hadVectorTable
		? DEFAULT_EMBEDDING_DIMENSIONS
		: embeddingDimensions;
	db.prepare(
		`INSERT INTO memory_meta (key, value) VALUES ('embedding_dimensions', ?)`,
	).run(String(storedDimensions));

	if (storedDimensions !== embeddingDimensions) {
		throw new Error(
			`memory DB embedding dimension mismatch: legacy database uses ${storedDimensions}, config wants ${embeddingDimensions}`,
		);
	}
}

interface PreparedStmts {
	insertMemory: Statement;
	insertVector: Statement;
	vectorSearch: Statement;
	ftsSearch: Statement;
	findMemoryByHash: Statement;
	maxVectorRowid: Statement;
	updateVectorRowid: Statement;
	updateAccess: Statement;
	deleteMemory: Statement;
	deleteVector: Statement;
	listAll: Statement;
}

// ── Factory ──────────────────────────────────────────────────────────────────

export function createMemoryStore(opts: MemoryStoreOptions = {}): MemoryStore {
	const dbPath = opts.dbPath ?? DEFAULT_DB_PATH;
	const embeddingDimensions = normalizeEmbeddingDimensions(
		opts.embeddingDimensions ?? DEFAULT_EMBEDDING_DIMENSIONS,
	);

	// Ensure parent directory exists
	mkdirSync(dirname(dbPath), { recursive: true });

	let db: ReturnType<typeof Database>;
	let stmts: PreparedStmts;

	try {
		db = new Database(dbPath);
		db.pragma("journal_mode = WAL");
		db.pragma("foreign_keys = ON");

		// Load sqlite-vec extension
		sqliteVec.load(db);

		const hadVectorTable = hasTable(db, "memory_vectors");

		// Initialize schema
		db.exec(schemaSql(embeddingDimensions));
		validateEmbeddingDimensions(db, embeddingDimensions, hadVectorTable);

		// Prepare statements
		stmts = {
			insertMemory: db.prepare(`
					INSERT OR IGNORE INTO memories (id, content, hash, vector_rowid, created_at)
					VALUES (?, ?, ?, NULL, ?)
				`),
			insertVector: db.prepare(`
					INSERT INTO memory_vectors (rowid, embedding)
					VALUES (?, ?)
				`),
			vectorSearch: db.prepare(`
					SELECT m.id, m.content, m.created_at, m.access_count, v.distance
					FROM memory_vectors v
					JOIN memories m ON m.vector_rowid = v.rowid
					WHERE v.embedding MATCH ? AND k = ?
				`),
			ftsSearch: db.prepare(`
					SELECT m.id, m.content, m.created_at, m.access_count, memory_fts.rank
					FROM memory_fts
					JOIN memories m ON m.rowid = memory_fts.rowid
					WHERE memory_fts MATCH ?
					ORDER BY rank
					LIMIT ?
				`),
			findMemoryByHash: db.prepare(`
					SELECT rowid, vector_rowid FROM memories WHERE hash = ?
				`),
			maxVectorRowid: db.prepare(`
					SELECT COALESCE(MAX(rowid), 0) AS maxRow FROM memory_vectors
				`),
			updateVectorRowid: db.prepare(`
					UPDATE memories SET vector_rowid = ? WHERE hash = ?
				`),
			updateAccess: db.prepare(`
					UPDATE memories
					SET last_accessed_at = ?, access_count = access_count + 1
					WHERE id = ?
				`),
			deleteMemory: db.prepare(`DELETE FROM memories WHERE id = ?`),
			deleteVector: db.prepare(`
					DELETE FROM memory_vectors
					WHERE rowid = (SELECT vector_rowid FROM memories WHERE id = ?)
				`),
			listAll: db.prepare(`
					SELECT id, content, created_at, access_count FROM memories
					ORDER BY created_at DESC LIMIT ?
				`),
		};
	} catch (err) {
		// If DB can't be opened, return a no-op store
		console.error(`[ergon-memory] Failed to open database at ${dbPath}:`, err);
		return createNoopStore();
	}

	const saveMemory = db.transaction((content: string, vector: number[]) => {
		const id = randomUUID();
		const hash = createHash("sha256").update(content).digest("hex");
		const now = Date.now();

		stmts.insertMemory.run(id, content, hash, now);

		const memoryRow = stmts.findMemoryByHash.get(hash) as
			{ rowid: number; vector_rowid: number | null } | undefined;

		if (!memoryRow) return;

		// Also repairs rows left without vectors by older failed saves.
		if (!memoryRow.vector_rowid) {
			const maxRow = stmts.maxVectorRowid.get() as
				{ maxRow: number | bigint } | undefined;
			const nextRowid = BigInt(maxRow?.maxRow ?? 0) + 1n;

			stmts.insertVector.run(nextRowid, floatArrayToBuffer(vector));
			stmts.updateVectorRowid.run(nextRowid, hash);
		}
	});

	return {
		recall(
			queryVector: number[],
			k: number = 5,
			queryText?: string,
		): MemoryItem[] {
			try {
				const vectorResults = runVectorSearch(stmts, queryVector, k);
				const ftsResults = queryText ? runFtsSearch(stmts, queryText, k) : [];

				// If we have both, do reciprocal rank fusion
				if (vectorResults.length > 0 && ftsResults.length > 0) {
					const results = reciprocalRankFusion(vectorResults, ftsResults, k);
					updateAccessStats(stmts, results);
					return results;
				}

				// Prefer vector results, fall back to FTS
				const results = vectorResults.length > 0 ? vectorResults : ftsResults;

				// Update access stats for hits
				updateAccessStats(stmts, results);

				return results;
			} catch (err) {
				console.error("[ergon-memory] recall failed:", err);
				return [];
			}
		},

		save(content: string, vector: number[]): void {
			try {
				saveMemory(content, vector);
			} catch (err) {
				console.error("[ergon-memory] save failed:", err);
			}
		},

		delete(id: string): void {
			try {
				stmts.deleteVector.run(id);
				stmts.deleteMemory.run(id);
			} catch (err) {
				console.error("[ergon-memory] delete failed:", err);
			}
		},

		list(limit: number = 100): MemoryItem[] {
			try {
				const rows = stmts.listAll.all(limit) as Array<{
					id: string;
					content: string;
					created_at: number;
					access_count: number;
				}>;
				return rows.map((row) => ({
					id: row.id,
					content: row.content,
					createdAt: row.created_at,
					accessCount: row.access_count,
				}));
			} catch (err) {
				console.error("[ergon-memory] list failed:", err);
				return [];
			}
		},

		close(): void {
			try {
				db.close();
			} catch {
				// ignore
			}
		},

		isAvailable(): boolean {
			return db.open;
		},
	};
}

// ── Vector search ────────────────────────────────────────────────────────────

function runVectorSearch(
	stmts: PreparedStmts,
	queryVector: number[],
	k: number,
): MemoryItem[] {
	const vectorBlob = floatArrayToBuffer(queryVector);
	const rows = stmts.vectorSearch.all(vectorBlob, k) as Array<{
		id: string;
		content: string;
		created_at: number;
		access_count: number;
		distance: number;
	}>;

	return rows.map((row) => ({
		id: row.id,
		content: row.content,
		score: row.distance,
		createdAt: row.created_at,
		accessCount: row.access_count,
	}));
}

// ── FTS keyword search ───────────────────────────────────────────────────────

function runFtsSearch(
	stmts: PreparedStmts,
	queryText: string,
	k: number,
): MemoryItem[] {
	const ftsQuery = buildFtsQuery(queryText);
	if (!ftsQuery) return [];

	const rows = stmts.ftsSearch.all(ftsQuery, k) as Array<{
		id: string;
		content: string;
		created_at: number;
		access_count: number;
		rank: number;
	}>;

	return rows.map((row) => ({
		id: row.id,
		content: row.content,
		score: row.rank,
		createdAt: row.created_at,
		accessCount: row.access_count,
	}));
}

function updateAccessStats(stmts: PreparedStmts, results: MemoryItem[]): void {
	const now = Date.now();
	for (const item of results) {
		stmts.updateAccess.run(now, item.id);
	}
}

function buildFtsQuery(queryText: string): string {
	return queryText
		.split(/\s+/)
		.map((term) => term.trim())
		.filter(Boolean)
		.map((term) => `"${term.replace(/"/g, '""')}"`)
		.join(" ");
}

// ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

/**
 * Combine vector and keyword results using reciprocal rank fusion.
 * The same item appearing in both lists gets a significantly boosted score.
 *
 * RRF score = sum(1 / (k + rank)) for each list the item appears in.
 * Higher score = better. We negate for consistency with "lower = closer".
 */
function reciprocalRankFusion(
	vectorResults: MemoryItem[],
	ftsResults: MemoryItem[],
	k: number,
): MemoryItem[] {
	const kConstant = 60; // standard RRF constant

	const scores = new Map<string, { item: MemoryItem; score: number }>();

	const addToMap = (results: MemoryItem[]) => {
		results.forEach((item, rank) => {
			const entry = scores.get(item.id);
			const rrfScore = 1 / (kConstant + rank + 1); // rank is 0-based
			if (entry) {
				entry.score += rrfScore;
			} else {
				scores.set(item.id, { item, score: rrfScore });
			}
		});
	};

	addToMap(vectorResults);
	addToMap(ftsResults);

	// Sort by score descending (higher = better), negate for "lower = closer" convention
	const merged = Array.from(scores.values())
		.sort((a, b) => b.score - a.score)
		.slice(0, k)
		.map((entry) => ({
			...entry.item,
			score: -entry.score, // negate so lower = better, consistent with distance
		}));

	return merged;
}

// ── No-op store (when DB can't be opened) ────────────────────────────────────

function createNoopStore(): MemoryStore {
	return {
		recall(): MemoryItem[] {
			return [];
		},
		save(): void {},
		delete(): void {},
		list(): MemoryItem[] {
			return [];
		},
		close(): void {},
		isAvailable(): boolean {
			return false;
		},
	};
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Convert a float32 array to a Node Buffer for sqlite-vec. */
function floatArrayToBuffer(floats: number[]): Buffer {
	const buf = new ArrayBuffer(floats.length * 4);
	const view = new Float32Array(buf);
	for (let i = 0; i < floats.length; i++) {
		view[i] = floats[i];
	}
	return Buffer.from(buf);
}
