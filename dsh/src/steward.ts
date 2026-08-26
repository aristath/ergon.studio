// Memory steward HTTP client (ported from ergon-studio for DeepSeek Harness).
//
// The steward is a small (4B) LLM running in its own llama-server instance,
// permanently resident (ergon-steward.service, port 18091). It does two
// narrow jobs:
//
//   1. rewriteQuery: turn a noisy user message into a tight search query
//      for openmemory. Returns null if the message has no searchable intent.
//
//   2. judgeSave:    look at one exchange (user message + assistant response)
//      and decide whether anything durable is worth remembering. Returns the
//      content to save, or null.
//
// Client config (URL, model, temperature) and the two prompts live in
// `prompts/steward.md` at the package root.
//
// DSH port notes:
// - The definition loads LAZILY and FAILS OPEN. The OpenCode client threw at
//   module scope; in DSH a plugin that throws during apply() fails the whole
//   preset mount. Memory is auxiliary: if the steward is unavailable, recall
//   and save simply stop happening, and the agent runs on.
// - This module is dependency-injectable: callers can pass a custom fetch
//   implementation (for tests) and override prompts.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// === Steward definition loader ===

export interface StewardDefinition {
  config: Record<string, string | number>;
  prompts: Record<string, string>;
}

/**
 * Parses a `steward.md`-style file: YAML frontmatter followed by a body
 * containing one or more `## <name>` sections. Frontmatter supports simple
 * `key: value` pairs (strings and numbers), with `#` for comments.
 */
export function parseStewardMd(content: string): StewardDefinition {
  const match = content.match(/^---[ \t]*\n([\s\S]*?)\n---[ \t]*\n?([\s\S]*)$/);
  if (!match) {
    throw new Error("steward.md: missing or malformed YAML frontmatter");
  }
  const frontmatterRaw = match[1];
  const body = match[2];

  const config: Record<string, string | number> = {};
  for (const rawLine of frontmatterRaw.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    let value: string | number = line.slice(colonIdx + 1).trim();
    value = value.replace(/\s+#.*$/, "").trim();
    value = value.replace(/^["']|["']$/g, "");
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      value = Number(value);
    }
    config[key] = value;
  }

  const prompts: Record<string, string> = {};
  const parts = body.split(/^##\s+/m);
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i];
    const firstNewline = part.indexOf("\n");
    if (firstNewline === -1) continue;
    const name = part.slice(0, firstNewline).trim().toLowerCase().replace(/\s+/g, "-");
    const content = part.slice(firstNewline + 1).trim();
    if (name && content) {
      prompts[name] = content;
    }
  }

  return { config, prompts };
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let definitionCache: StewardDefinition | null = null;
let definitionError: Error | null = null;

/**
 * Load (and cache) the steward definition. Returns null on failure — the
 * caller degrades to no-op behavior. The first failure is retained so later
 * calls don't re-probe the filesystem every time.
 */
function loadDefinition(): StewardDefinition | null {
  if (definitionCache) return definitionCache;
  if (definitionError) return null;
  // dist/steward.js → package root is "..", then prompts/steward.md
  const absPath = join(__dirname, "..", "prompts", "steward.md");
  try {
    const raw = readFileSync(absPath, "utf8");
    definitionCache = parseStewardMd(raw);
    return definitionCache;
  } catch (err) {
    definitionError = err instanceof Error ? err : new Error(String(err));
    return null;
  }
}

// Test hook: reset the cache (and allow injecting a definition).
export function _resetStewardDefinitionCacheForTests(
  definition: StewardDefinition | null = null,
): void {
  definitionCache = definition;
  definitionError = null;
}

function numberConfig(...values: Array<unknown>): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return 0;
}

// === Exported client types ===

export interface StewardClient {
  /** True once the steward definition loaded; false when recall/save are no-ops. */
  readonly available: boolean;
  rewriteQuery(userMessage: string): Promise<string | null>;
  judgeSave(userMessage: string, assistantResponse: string): Promise<string | null>;
}

export interface StewardOptions {
  baseURL?: string;
  model?: string;
  fetch?: typeof fetch;
  rewritePrompt?: string;
  judgePrompt?: string;
  temperature?: number;
}

// === Defaults (env var → steward.md frontmatter → hard default) ===

const FALLBACK_URL = "http://127.0.0.1:18091";
const FALLBACK_MODEL = "ergon-studio-memory-steward";
const FALLBACK_TEMPERATURE = 0.3;

function definitionConfig(): Record<string, string | number> {
  return loadDefinition()?.config ?? {};
}
function definitionPrompt(name: string): string | null {
  return loadDefinition()?.prompts[name] ?? null;
}

/** Resolved steward URL (env > steward.md > default). */
export function stewardUrlOverride(): string | undefined {
  return process.env.ERGON_STEWARD_URL ?? (definitionConfig().url as string | undefined);
}
/** Resolved steward model name. */
export function stewardModelOverride(): string | undefined {
  return process.env.ERGON_STEWARD_MODEL ?? (definitionConfig().model as string | undefined);
}

export function createStewardClient(opts: StewardOptions = {}): StewardClient {
  const cfg = definitionConfig();
  const baseURL = opts.baseURL ?? process.env.ERGON_STEWARD_URL ?? String(cfg.url ?? FALLBACK_URL);
  const model = opts.model ?? process.env.ERGON_STEWARD_MODEL ?? String(cfg.model ?? FALLBACK_MODEL);
  const temperature = opts.temperature ?? numberConfig(process.env.ERGON_STEWARD_TEMPERATURE, cfg.temperature, FALLBACK_TEMPERATURE);
  const fetchImpl = opts.fetch ?? fetch;
  const rewritePrompt = opts.rewritePrompt ?? definitionPrompt("rewrite");
  const judgePrompt = opts.judgePrompt ?? definitionPrompt("judge");
  const available = rewritePrompt !== null && judgePrompt !== null;

  async function complete(systemPrompt: string, userContent: string): Promise<string | null> {
    try {
      const response = await fetchImpl(`${baseURL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          temperature,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userContent },
          ],
        }),
      });
      if (!response.ok) return null;
      const data = (await response.json()) as any;
      const text = data?.choices?.[0]?.message?.content;
      if (typeof text !== "string") return null;
      // Strip any leading <think>...</think> block. Qwen 3.5 with
      // enable_thinking=false still emits empty </think> tags; other
      // thinking-capable models may emit real reasoning we want to
      // discard before parsing the actual answer.
      return text.replace(/^\s*<think>[\s\S]*?<\/think>\s*/i, "").trim();
    } catch {
      return null;
    }
  }

  return {
    available,

    async rewriteQuery(userMessage: string): Promise<string | null> {
      if (!available || !rewritePrompt) return null;
      const text = await complete(rewritePrompt, userMessage);
      if (text === null) return null;
      if (text === "NONE" || text === "") return null;
      return text;
    },

    async judgeSave(userMessage: string, assistantResponse: string): Promise<string | null> {
      if (!available || !judgePrompt) return null;
      const exchange = `User: ${userMessage}\nAssistant: ${assistantResponse}`;
      const text = await complete(judgePrompt, exchange);
      if (text === null) return null;
      try {
        const stripped = text.replace(/^```(?:json)?\s*|\s*```$/g, "").trim();
        const parsed = JSON.parse(stripped);
        if (parsed?.save && typeof parsed.save.content === "string") {
          return parsed.save.content;
        }
        return null;
      } catch {
        return null;
      }
    },
  };
}
