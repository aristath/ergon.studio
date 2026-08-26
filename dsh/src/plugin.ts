// Ergon for DeepSeek Harness — the cordis plugin.
//
// Mounted once per session by the ergon agent preset (see presets/ergon/),
// this plugin contributes:
//
//   1. `debate`        — two-agent alternating coding debate (subagent seam).
//   2. `run_parallel`  — N specialist agents in parallel (subagent seam).
//   3. `memory_search` — explicit semantic search of the openmemory corpus.
//   4. Memory steward recall — user message queued → 4B rewrite → openmemory
//      query → result stashed per agent → surfaced through a dynamic prompt
//      context (append-only runtime-context snapshot, KV-cache friendly).
//   5. Memory steward save — turn/end (completed) → 4B judge → conditional
//      openmemory store. Fire-and-forget, off the critical path.
//   6. Scratchpad context — .ergon.studio/scratchpad.md (+ HANDOFF.md)
//      re-read on every prompt assembly; the runtime-context projection
//      re-appends the snapshot when the file changes, which is also how the
//      scratchpad survives compaction without a compaction hook.
//
// Every external dependency is fail-open: a dead steward or memory service
// degrades to no memory, never to a broken session.

import z from "@deepseek-ai/schemastery";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { createStewardClient, type StewardClient } from "./steward.js";
import { createMemoryClient, type MemoryClient } from "./memory.js";
import { loadRoster, getRosterEntry, type RosterEntry } from "./roster.js";
import { scratchpadBlock } from "./scratchpad.js";
import { runDebate, runParallel, type SubagentsService } from "./debate.js";
import { ensureErgonAssets, dshHome } from "./install.js";

// === Cordis plugin identity ===

/** Cordis plugin name. */
export const name = "ergon";

/** Hard dependencies (all present in the dsh-base host composition). */
export const inject = ["tools", "subagents", "systemPrompt", "agents"];

/** Runtime schema for the ergon plugin row. */
export const Config = z.object({
  /** Steward base URL. Default: $ERGON_STEWARD_URL → prompts/steward.md → http://127.0.0.1:18091. */
  stewardUrl: z.string(),
  /** Steward model name. Default: $ERGON_STEWARD_MODEL → prompts/steward.md → ergon-studio-memory-steward. */
  stewardModel: z.string(),
  /** openmemory base URL. Default: $ERGON_MEMORY_URL → http://127.0.0.1:8082. */
  memoryUrl: z.string(),
  /** Per-stage timeout (ms) for recall's external calls. */
  recallTimeoutMs: z.number().default(5000),
  /** Max memories returned per recall / memory_search. */
  recallLimit: z.number().default(5),
});

// === internal types (structural — no import from dsh internals) ===

interface AssemblyContext {
  agent?: {
    session?: {
      header?: { cwd?: unknown };
    };
  };
  [key: string]: unknown;
}

interface UserMessageLike {
  content?: Array<{ type?: string; text?: string }>;
  source?: { kind?: string };
}

function messageText(message: UserMessageLike | undefined | null): string {
  if (!message || !Array.isArray(message.content)) return "";
  return message.content
    .filter((b) => b && b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("\n")
    .trim();
}

/** Race a promise against a timeout. Always settles. */
const TIMEOUT = Symbol("ergon-timeout");
async function raceWithTimeout<T>(
  inner: Promise<T>,
  timeoutMs: number,
): Promise<{ ok: true; value: T } | { ok: false }> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const value = await Promise.race<T | typeof TIMEOUT>([
      inner,
      new Promise<typeof TIMEOUT>((resolve) => {
        timer = setTimeout(() => resolve(TIMEOUT), timeoutMs);
      }),
    ]);
    return value === TIMEOUT ? { ok: false } : { ok: true, value };
  } catch {
    return { ok: false };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// === apply ===

export function apply(ctx: any, config: Schemastery.TypeT<typeof Config>): void {
  // The ergon bundle patch makes this preset the default for new sessions, so
  // the preset must exist in the user preset root by the time the first
  // session resolves it. Install from the package on first mount (idempotent,
  // install-only-when-missing, fail-open — see install.ts).
  try {
    ensureErgonAssets(dshHome(), (msg) => ctx.logger?.warn?.(msg));
  } catch (err) {
    ctx.logger?.warn?.(`ergon: self-install failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  const steward: StewardClient = createStewardClient({
    baseURL: config.stewardUrl,
    model: config.stewardModel,
  });
  const memory: MemoryClient = createMemoryClient({
    baseURL: config.memoryUrl,
    defaultLimit: config.recallLimit,
  });

  // Per-agent recall cache: agent → rendered recall block (absent = none).
  const recallCache = new Map<object, string>();
  // Last user text that triggered a recall per agent. Doubles as a dedup for
  // re-queued identical messages and as a staleness guard: a result whose
  // trigger text was superseded mid-flight is dropped, not installed.
  const lastRecalledText = new Map<object, string>();
  // Save dedup: session object → last turn number processed.
  const savedTurns = new Map<object, number>();

  // ── recall ──────────────────────────────────────────────────────────────

  async function recallFor(agent: object, userText: string): Promise<void> {
    if (!steward.available) return;
    const q = await raceWithTimeout(steward.rewriteQuery(userText), config.recallTimeoutMs);
    if (!q.ok || !q.value) return;
    const r = await raceWithTimeout(memory.recall(q.value, config.recallLimit), config.recallTimeoutMs);
    if (!r.ok) return;
    const items = r.value;
    if (items.length === 0) {
      recallCache.delete(agent);
      return;
    }
    const block =
      "## Relevant prior notes (from memory steward)\n\n" +
      items.map((m) => `- ${m.content}`).join("\n");
    // Only install if this agent didn't recall something newer meanwhile.
    if (lastRecalledText.get(agent) === userText) {
      recallCache.set(agent, block);
    }
  }

  ctx.on("agent/inbox/inserted", (payload: { agent?: object; message?: UserMessageLike }) => {
    const agent = payload.agent;
    if (!agent) return;
    const text = messageText(payload.message);
    if (!text) return;
    if (payload.message?.source?.kind && payload.message.source.kind !== "user") return;
    if (lastRecalledText.get(agent) === text) return;
    lastRecalledText.set(agent, text);
    void recallFor(agent, text).catch(() => {
      /* recall is auxiliary — never surface failures */
    });
  });

  // Dynamic prompt context: recall block. Re-read from the cache on every
  // assembly; the runtime-context projection re-appends the snapshot only
  // when the rendered text actually changes (append-only history).
  ctx.effect(
    () =>
      ctx.systemPrompt.context({
        name: "ergon:memory",
        order: 95,
        text: (c: AssemblyContext) => (c.agent ? (recallCache.get(c.agent) ?? "") : ""),
      }),
    "ergon.memory.context()",
  );

  // Dynamic prompt context: scratchpad (+ handoff) for the agent's workspace.
  ctx.effect(
    () =>
      ctx.systemPrompt.context({
        name: "ergon:scratchpad",
        order: 90,
        text: (c: AssemblyContext) => {
          const cwd = c.agent?.session?.header?.cwd;
          if (typeof cwd !== "string" || cwd.length === 0) return "";
          return scratchpadBlock(cwd);
        },
      }),
    "ergon.scratchpad.context()",
  );

  // ── save ────────────────────────────────────────────────────────────────

  function collectTurnText(session: any, turn: number): { user: string; assistant: string } {
    const events: Array<{ type: string; data: any; seq: number }> = session?.events ?? [];
    // Boundary: this turn's turn/start seq (events after it belong to the turn).
    let startSeq = -1;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e?.type === "turn/start" && e.data?.turn === turn) {
        startSeq = e.seq;
        break;
      }
    }
    if (startSeq === -1) return { user: "", assistant: "" };
    let user = "";
    let assistant = "";
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (!e || e.seq <= startSeq) break;
      if (e.type === "assistant/message" && e.data?.turn === turn) {
        const blocks: Array<{ type?: string; text?: string }> = e.data?.message?.content ?? [];
        const text = blocks
          .filter((b) => b && b.type === "text" && typeof b.text === "string")
          .map((b) => b.text as string)
          .join("\n")
          .trim();
        if (text && !assistant) assistant = text;
      } else if (e.type === "user/message") {
        const blocks: Array<{ type?: string; text?: string }> = e.data?.content ?? [];
        const text = blocks
          .filter((b) => b && b.type === "text" && typeof b.text === "string")
          .map((b) => b.text as string)
          .join("\n")
          .trim();
        const sourceKind = e.data?.source?.kind;
        if (text && sourceKind === "user" && !user) user = text;
      }
    }
    return { user, assistant };
  }

  function handleTurnEnd(session: any, turn: number): void {
    void (async () => {
      if (!steward.available) return;
      const { user, assistant } = collectTurnText(session, turn);
      if (!user || !assistant) return;
      const saved = await steward.judgeSave(user, assistant);
      if (saved) await memory.save(saved);
    })().catch(() => {
      /* save is auxiliary — never surface failures */
    });
  }

  ctx.on("session/event", (session: any, event: any) => {
    if (event?.type !== "turn/end") return;
    if (event.data?.reason?.kind !== "completed") return;
    const turn = event.data?.turn;
    if (typeof turn !== "number") return;
    const agent = ctx.agents?.get?.(session.id);
    if (agent === void 0 || agent.session !== session) return;
    const last = savedTurns.get(session);
    if (last !== undefined && last >= turn) return;
    savedTurns.set(session, turn);
    handleTurnEnd(session, turn);
  });

  // ── tools ───────────────────────────────────────────────────────────────

  const subagents = ctx.subagents as SubagentsService;

  // debate
  ctx.effect(
    () =>
      ctx.tools.register(
        defineTool({
          name: "debate",
          description:
            "Run two agents in an alternating coding debate. " +
            "Agent A does the first pass, Agent B reviews or improves it, then they alternate until one agrees, blocks, or max_turns is reached. " +
            "Use this when you want intentional cross-review and convergence, not isolated parallel opinions.",
          parameters: {
            agent_a: {
              type: "string",
              required: true,
              description: `First agent (takes the first turn). One of: ${loadRoster().map((r) => r.id).join(", ")}.`,
            },
            agent_b: {
              type: "string",
              required: true,
              description: "Second agent (reviews or improves the first turn). One of the roster names.",
            },
            task: {
              type: "string",
              required: true,
              description: "Short, specific task or question for the debate",
            },
            max_turns: {
              type: "integer",
              description: "Maximum total agent turns, including the first turn. Defaults to 6, capped at 12.",
            },
          },
          output: {
            schema: {
              type: "object",
              additionalProperties: false,
              properties: { transcript: { type: "string", required: true } },
            },
            render: (_args: any, value: any) => [
              { type: "text", text: `# Debate result\n\n${value?.transcript ?? ""}` },
            ],
          },
          async execute(args: any, exec: any) {
            const a = getRosterEntry(args.agent_a);
            const b = getRosterEntry(args.agent_b);
            if (!a || !b) {
              const known = loadRoster().map((r) => r.id).join(", ");
              return {
                transcript: `# Debate result\n\nStatus: FAILED\n\nUnknown agent name(s). The ergon roster is: ${known}`,
              };
            }
            const maxTurns = Math.min(Math.max(args.max_turns ?? 6, 2), 12);
            const { transcript } = await runDebate({
              agentA: a,
              agentB: b,
              task: args.task,
              maxTurns,
              parent: exec.agent,
              signal: exec.signal,
              subagents,
            });
            return { transcript };
          },
        }),
      ),
    "ergon.tool.debate",
  );

  // run_parallel
  ctx.effect(
    () =>
      ctx.tools.register(
        defineTool({
          name: "run_parallel",
          description:
            "Run multiple ergon agents in parallel and return their combined output. " +
            "Each task specifies an agent name and a brief. All tasks execute concurrently. " +
            "This tool delegates to LLM agents — it is NOT a way to run shell commands or built-in tools. " +
            "Avoid using write-capable agents (e.g. coder) in parallel — they may conflict on shared files.",
          parameters: {
            tasks: {
              type: "array",
              required: true,
              description: "List of agent+brief pairs to run in parallel (at least one)",
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  agent: {
                    type: "string",
                    required: true,
                    description: `Agent name. One of: ${loadRoster().map((r) => r.id).join(", ")}.`,
                  },
                  brief: {
                    type: "string",
                    required: true,
                    description: "Full brief to send to the agent",
                  },
                },
              },
            },
          },
          output: {
            schema: {
              type: "object",
              additionalProperties: false,
              properties: { output: { type: "string", required: true } },
            },
            render: (_args: any, value: any) => [
              { type: "text", text: `# Parallel agents\n\n${value?.output ?? ""}` },
            ],
          },
          async execute(args: any, exec: any) {
            const rawTasks: unknown[] = Array.isArray(args.tasks) ? (args.tasks as unknown[]) : [];
            const tasks = rawTasks
              .map((t): { agent: RosterEntry; brief: string } | null => {
                const rec = t as { agent?: unknown; brief?: unknown } | null | undefined;
                const entry = getRosterEntry(typeof rec?.agent === "string" ? rec.agent : "");
                return entry && typeof rec?.brief === "string" ? { agent: entry, brief: rec.brief } : null;
              })
              .filter((t): t is { agent: RosterEntry; brief: string } => t !== null);
            if (tasks.length === 0) {
              return { output: "No valid tasks. Each task needs an agent from the ergon roster and a brief." };
            }
            const output = await runParallel({
              tasks,
              parent: exec.agent,
              signal: exec.signal,
              subagents,
            });
            return { output };
          },
        }),
      ),
    "ergon.tool.run_parallel",
  );

  // memory_search
  ctx.effect(
    () =>
      ctx.tools.register(
        defineTool({
          name: "memory_search",
          description:
            "Semantic search over the project's long-term memory (openmemory). " +
            "Use for explicit lookups: past decisions, user preferences, constraints, gotchas — things worth re-reading before acting.",
          parameters: {
            query: {
              type: "string",
              required: true,
              description: "Natural-language search query",
            },
            limit: {
              type: "integer",
              description: "Maximum results (default 5, max 20)",
            },
          },
          output: {
            schema: {
              type: "object",
              additionalProperties: false,
              properties: {
                count: { type: "integer", required: true },
                results: {
                  type: "array",
                  required: true,
                  items: {
                    type: "object",
                    additionalProperties: false,
                    properties: {
                      id: { type: "string", required: true },
                      content: { type: "string", required: true },
                      score: { type: "number" },
                    },
                  },
                },
              },
            },
            render: (_args: any, value: any) => {
              const results: Array<{ content: string; score?: number }> = value?.results ?? [];
              return [
                {
                  type: "text",
                  text:
                    results.length === 0
                      ? "No matching memories."
                      : results.map((r) => `- ${r.content}`).join("\n"),
                },
              ];
            },
          },
          async execute(args: any) {
            const limit = Math.min(Math.max(args.limit ?? config.recallLimit, 1), 20);
            const results = await memory.recall(args.query, limit);
            return { count: results.length, results };
          },
        }),
      ),
    "ergon.tool.memory_search",
  );

  if (!steward.available) {
    ctx.logger?.warn?.(
      "ergon: memory steward definition unavailable (prompts/steward.md missing?) — recall and save disabled",
    );
  }
}
