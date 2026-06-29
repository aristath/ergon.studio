// Ergon Memory Steward — Pi extension
//
// Hooks into the Pi agent lifecycle to provide cross-session memory:
//
//   before_agent_start      — recall path (rewrite → embed → search → inject)
//   turn_end                — save path   (judge → embed → store)
//   session_start           — startup health checks
//   session_shutdown        — cleanup
//
// Note: Scratchpad injection has moved to @ergon.studio/pi-scratchpad.

import type {
	BeforeAgentStartEvent,
	ExtensionAPI,
	ExtensionContext,
	SessionShutdownEvent,
	SessionStartEvent,
	TurnEndEvent,
} from "@earendil-works/pi-coding-agent";

import {
	createStewardClient,
	type StewardClient,
	DEFAULT_STEWARD_URL,
	DEFAULT_MEMORY_DB_PATH,
	DEFAULT_RECALL_LIMIT,
} from "../src/steward.js";

import {
	createEmbedderClient,
	type EmbedderClient,
	DEFAULT_EMBEDDER_URL,
	DEFAULT_EMBEDDER_MODEL_PATH,
	DEFAULT_EMBEDDING_DIMENSIONS,
} from "../src/embedder.js";

import { createMemoryStore, type MemoryStore } from "../src/memory-store.js";
import { existsSync } from "node:fs";

// ── Runtime state ────────────────────────────────────────────────────────────

let steward: StewardClient | null = null;
let embedder: EmbedderClient | null = null;
let memoryStore: MemoryStore | null = null;

// Per-session dedup: prevents judging the same exchange twice
const lastJudgedAssistantId = new Map<string, string>();
const pendingSaves = new Set<Promise<void>>();

// Component health flags
const health = {
	steward: false,
	embedder: false,
	memory: false,
	stewardReason: "",
	embedderReason: "",
	memoryReason: "",
};

// ── Timeout utility ──────────────────────────────────────────────────────────

const RECALL_TIMEOUT_MS = 5000;
const SAVE_TIMEOUT_MS = 10000;
const SHUTDOWN_SAVE_DRAIN_MS = SAVE_TIMEOUT_MS * 2 + 2000;

async function runWithTimeout<T>(
	run: (signal: AbortSignal) => Promise<T>,
	timeoutMs: number,
	label: string,
): Promise<{ ok: true; value: T } | { ok: false; reason: string }> {
	const controller = new AbortController();
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		const TIMEOUT_SENTINEL = Symbol("timeout");
		const value = await Promise.race<T | typeof TIMEOUT_SENTINEL>([
			run(controller.signal),
			new Promise<typeof TIMEOUT_SENTINEL>((resolve) => {
				timer = setTimeout(() => {
					controller.abort(`${label} timeout after ${timeoutMs}ms`);
					resolve(TIMEOUT_SENTINEL);
				}, timeoutMs);
			}),
		]);
		if (value === TIMEOUT_SENTINEL) {
			return { ok: false, reason: `${label} timeout after ${timeoutMs}ms` };
		}
		return { ok: true, value: value as T };
	} catch (err) {
		return {
			ok: false,
			reason: err instanceof Error ? err.message : String(err),
		};
	} finally {
		if (timer) clearTimeout(timer);
	}
}

// ── Recall ───────────────────────────────────────────────────────────────────

async function recall(prompt: string): Promise<string | null> {
	if (!steward || !embedder || !memoryStore) return null;
	if (!prompt || prompt.trim().length === 0) return null;
	const stewardClient = steward;
	const embedderClient = embedder;
	const store = memoryStore;

	// 1. Rewrite
	const rewriteResult = await runWithTimeout(
		(signal) => stewardClient.rewriteQuery(prompt, { signal }),
		RECALL_TIMEOUT_MS,
		"steward.rewriteQuery",
	);
	if (!rewriteResult.ok) return null;
	const query = rewriteResult.value;
	if (!query) return null;

	// 2. Embed query
	const embedResult = await runWithTimeout(
		(signal) => embedderClient.embed(query, { signal }),
		RECALL_TIMEOUT_MS,
		"embedder.embed",
	);
	if (!embedResult.ok) return null;
	const vector = embedResult.value;
	if (!vector) return null;

	// 3. Search
	const memories = store.recall(vector, DEFAULT_RECALL_LIMIT, query);
	if (memories.length === 0) return null;

	return (
		"## Relevant prior notes (from memory steward)\n\n" +
		memories.map((m) => `- ${m.content}`).join("\n")
	);
}

// ── Save ─────────────────────────────────────────────────────────────────────

async function save(
	sessionID: string,
	userText: string,
	assistantText: string,
	assistantId: string,
): Promise<void> {
	if (!steward || !embedder || !memoryStore) return;
	const stewardClient = steward;
	const embedderClient = embedder;
	const store = memoryStore;

	// Dedup: skip if this exact assistant message was already judged
	if (assistantId) {
		if (lastJudgedAssistantId.get(sessionID) === assistantId) return;
		lastJudgedAssistantId.set(sessionID, assistantId);
	}

	// 1. Judge
	const judgeResult = await runWithTimeout(
		(signal) => stewardClient.judgeSave(userText, assistantText, { signal }),
		SAVE_TIMEOUT_MS,
		"steward.judgeSave",
	);
	if (!judgeResult.ok) return;
	const content = judgeResult.value;
	if (!content) return;

	// 2. Embed
	const embedResult = await runWithTimeout(
		(signal) => embedderClient.embed(content, { signal }),
		SAVE_TIMEOUT_MS,
		"embedder.embed",
	);
	if (!embedResult.ok) return;
	const vector = embedResult.value;
	if (!vector) return;

	// 3. Store (hash dedup: INSERT OR IGNORE)
	store.save(content, vector);
}

function trackSave(job: Promise<void>): void {
	pendingSaves.add(job);
	job.finally(() => pendingSaves.delete(job)).catch(() => {});
}

async function drainPendingSaves(): Promise<void> {
	if (pendingSaves.size === 0) return;
	await runWithTimeout(
		() => Promise.allSettled([...pendingSaves]).then(() => undefined),
		SHUTDOWN_SAVE_DRAIN_MS,
		"memory-steward.pendingSaves",
	);
}

// ── Health checks ────────────────────────────────────────────────────────────

function initializeComponents(): void {
	if (!steward) {
		try {
			steward = createStewardClient();
		} catch {
			steward = null;
		}
	}

	if (!embedder) {
		try {
			embedder = createEmbedderClient();
		} catch {
			embedder = null;
		}
	}

	if (!memoryStore) {
		try {
			memoryStore = createMemoryStore({
				dbPath: DEFAULT_MEMORY_DB_PATH,
				embeddingDimensions: DEFAULT_EMBEDDING_DIMENSIONS,
			});
		} catch {
			memoryStore = null;
		}
	}
}

async function checkHealth(): Promise<void> {
	initializeComponents();

	try {
		const res = await fetch(`${DEFAULT_STEWARD_URL}/v1/models`, {
			signal: AbortSignal.timeout(3000),
		});
		health.steward = res.ok;
		health.stewardReason = res.ok ? "" : "steward unhealthy";
	} catch {
		health.steward = false;
		health.stewardReason = "steward unreachable";
	}

	try {
		const res = await fetch(`${DEFAULT_EMBEDDER_URL}/v1/models`, {
			signal: AbortSignal.timeout(3000),
		});
		health.embedder = res.ok;
		health.embedderReason = res.ok ? "" : "embedder unhealthy";
	} catch {
		health.embedder = false;
		health.embedderReason =
			DEFAULT_EMBEDDER_MODEL_PATH && !existsSync(DEFAULT_EMBEDDER_MODEL_PATH)
				? "embedder model missing"
				: "embedder unreachable";
	}

	health.memory = memoryStore?.isAvailable() ?? false;
	health.memoryReason = health.memory ? "" : "memory db unavailable";
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Extract plain text from a user or assistant message */
type MessageContent = string | Array<{ type: string; text?: string }>;
function extractText(msg: { content: MessageContent }): string {
	const parts =
		typeof msg.content === "string"
			? msg.content
			: msg.content
					.filter(
						(p: {
							type: string;
							text?: string;
						}): p is { type: "text"; text: string } =>
							p.type === "text" && typeof p.text === "string",
					)
					.map((p: { type: "text"; text: string }) => p.text)
					.join("\n");
	return parts.trim();
}

// ── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI): void {
	// ── before_agent_start: recall injection ──────────────────────────────────

	pi.on("before_agent_start", async (event: BeforeAgentStartEvent) => {
		initializeComponents();
		const recallBlock = await recall(event.prompt);
		if (!recallBlock) return;

		return {
			systemPrompt: (event.systemPrompt ?? "") + "\n\n" + recallBlock,
		};
	});

	// ── turn_end: save path (fire-and-forget) ─────────────────────────────────

	pi.on("turn_end", async (event: TurnEndEvent, ctx: ExtensionContext) => {
		initializeComponents();

		const sessionFile = ctx.sessionManager.getSessionFile();
		const sessionID = sessionFile ?? "default";

		// event.message is the final message of the turn — typically an assistant message
		const msg = event.message;
		if (msg.role !== "assistant") return;

		const assistantText = extractText(msg);
		const assistantId = msg.responseId ?? "";
		if (!assistantText) return;

		// Walk session entries backwards to find the preceding user message
		const entries = ctx.sessionManager.getEntries();
		let userText = "";

		for (let i = entries.length - 1; i >= 0; i--) {
			const entry = entries[i];
			if (entry.type === "message" && entry.message.role === "user") {
				userText = extractText(entry.message);
				break;
			}
		}

		if (!userText) return;

		trackSave(save(sessionID, userText, assistantText, assistantId));
	});

	// ── session_start: notify ─────────────────────────────────────────────────

	pi.on(
		"session_start",
		async (_event: SessionStartEvent, ctx: ExtensionContext) => {
			await checkHealth();

			const parts: string[] = [];
			if (!health.steward) {
				parts.push(health.stewardReason || "steward unreachable");
			}
			if (!health.embedder) {
				parts.push(health.embedderReason || "embedder unreachable");
			}
			if (!health.memory) parts.push(health.memoryReason);

			if (parts.length > 0) {
				ctx.ui.setStatus("memory-steward", `⚠ memory: ${parts.join(", ")}`);
			} else {
				ctx.ui.setStatus("memory-steward", "✓ memory active");
			}
		},
	);

	// ── session_shutdown: cleanup ─────────────────────────────────────────────

	pi.on(
		"session_shutdown",
		async (_event: SessionShutdownEvent, ctx: ExtensionContext) => {
			const sessionFile = ctx.sessionManager.getSessionFile();
			if (sessionFile) {
				lastJudgedAssistantId.delete(sessionFile);
			} else {
				lastJudgedAssistantId.clear();
			}
			await drainPendingSaves();
			memoryStore?.close();
			memoryStore = null;
			health.memory = false;
			health.memoryReason = "";
			ctx.ui.setStatus("memory-steward", undefined);
		},
	);
}
