// Embedder HTTP client.
//
// Talks to a dedicated llama-server running the Granite 311M multilingual r2
// embedding model. Produces 768-dimensional L2-normalized vectors.
//
// Config is read from prompts/steward.md frontmatter (embedder_url,
// embedder_model, embedder_dimensions).

import { parseStewardMd } from "./steward.js";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// ── Defaults from frontmatter ────────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function packageFilePath(...parts: string[]): string {
	const candidates = [
		join(__dirname, "..", ...parts),
		join(__dirname, "..", "..", ...parts),
	];
	return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
}

function loadEmbedderConfig() {
	const absPath = packageFilePath("prompts", "steward.md");
	const raw = readFileSync(absPath, "utf8");
	return parseStewardMd(raw);
}

const def = loadEmbedderConfig();

export const DEFAULT_EMBEDDER_URL: string = String(
	process.env.ERGON_EMBEDDER_URL ??
		def.config.embedder_url ??
		"http://127.0.0.1:18092",
);
export const DEFAULT_EMBEDDER_MODEL: string = String(
	process.env.ERGON_EMBEDDER_MODEL ??
		def.config.embedder_model ??
		"granite-embedding-311m",
);
export const DEFAULT_EMBEDDER_MODEL_PATH: string = expandHomePath(
	String(
		process.env.ERGON_EMBEDDER_MODEL_PATH ??
			def.config.embedder_model_path ??
			"",
	),
);
export const DEFAULT_EMBEDDING_DIMENSIONS: number = numberConfig(
	process.env.ERGON_EMBEDDER_DIMENSIONS,
	def.config.embedder_dimensions,
	768,
);

// ── Client types ─────────────────────────────────────────────────────────────

export interface EmbedderClient {
	/** Embed a single text string, returning a 768-dim L2-normalized vector. */
	embed(text: string, opts?: EmbedderRequestOptions): Promise<number[] | null>;

	/** Embed multiple texts in a single request. Returns parallel array (null on failure). */
	embedBatch(
		texts: string[],
		opts?: EmbedderRequestOptions,
	): Promise<(number[] | null)[] | null>;
}

export interface EmbedderRequestOptions {
	signal?: AbortSignal;
}

export interface EmbedderOptions {
	baseURL?: string;
	model?: string;
	dimensions?: number;
	fetch?: typeof fetch;
}

// ── Client factory ───────────────────────────────────────────────────────────

export function createEmbedderClient(
	opts: EmbedderOptions = {},
): EmbedderClient {
	const baseURL = (opts.baseURL ?? DEFAULT_EMBEDDER_URL).replace(/\/+$/, "");
	const model = opts.model ?? DEFAULT_EMBEDDER_MODEL;
	const dimensions = opts.dimensions ?? DEFAULT_EMBEDDING_DIMENSIONS;
	const fetchImpl = opts.fetch ?? fetch;

	async function doEmbed(
		input: string | string[],
		signal?: AbortSignal,
	): Promise<(number[] | null)[] | null> {
		try {
			const response = await fetchImpl(`${baseURL}/v1/embeddings`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
					signal,
					body: JSON.stringify({
						model,
						input,
					}),
				});
			if (!response.ok) return null;

			const data: any = await response.json();
			const items = data?.data;
			if (!Array.isArray(items)) return null;

			const vectors: (number[] | null)[] = [];
			for (const item of items) {
				const emb = item?.embedding;
				if (!Array.isArray(emb) || emb.length === 0) {
					vectors.push(null);
					continue;
				}
				// Truncate or pad to requested dimensions (Matryoshka support)
				let vec = emb.slice(0, dimensions);
				if (vec.length < dimensions) {
					vec = [...vec, ...new Array(dimensions - vec.length).fill(0)];
				}
				vectors.push(vec);
			}
			return vectors;
		} catch {
			return null;
		}
	}

	return {
		async embed(
			text: string,
			requestOpts: EmbedderRequestOptions = {},
		): Promise<number[] | null> {
			const result = await doEmbed(text, requestOpts.signal);
			if (!result || result.length === 0) return null;
			return result[0];
		},

		async embedBatch(
			texts: string[],
			requestOpts: EmbedderRequestOptions = {},
		): Promise<(number[] | null)[] | null> {
			if (texts.length === 0) return [];
			const result = await doEmbed(texts, requestOpts.signal);
			if (!result) return null;
			return result;
		},
	};
}

function expandHomePath(path: string): string {
	if (path === "~") return homedir();
	if (path.startsWith("~/")) return join(homedir(), path.slice(2));
	return path;
}

function parseNumber(value: string | undefined): number | undefined {
	if (!value) return undefined;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : undefined;
}

function numberConfig(
	envValue: string | undefined,
	fileValue: string | number | undefined,
	fallback: number,
): number {
	return (
		parseNumber(envValue) ??
		(typeof fileValue === "number" ? fileValue : fallback)
	);
}
