import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import { existsSync, readFileSync } from "fs"
import { join } from "path"
import { createStewardClient, type StewardClient } from "./steward.js"
import { createMemoryClient, type MemoryClient } from "./memory.js"

export interface ErgonPluginDeps {
  steward?: StewardClient
  memory?: MemoryClient
  /**
   * Maximum time (ms) to wait for opencode's GET /agent endpoint at plugin
   * init before falling back to a permissive run_parallel schema. Defaults
   * to 15000ms. Tests inject small values to keep the suite fast.
   */
  agentLookupTimeoutMs?: number
  /**
   * Maximum time (ms) to wait for an external recall call (steward rewrite +
   * memory recall) inside the chat.message hook. Defaults to 5000ms. The
   * hook is awaited by opencode, so an unbounded call stalls the user's
   * turn — this timeout protects against a hung steward or memory backend.
   */
  chatMessageTimeoutMs?: number
}

/**
 * Race a promise against a timeout. Resolves with `{ ok: true, value }` if
 * the inner promise wins, or `{ ok: false, reason }` if the timer fires.
 * Always clears the timer so it doesn't keep the event loop alive on the
 * happy path.
 */
async function raceWithTimeout<T>(
  inner: Promise<T>,
  timeoutMs: number,
  label: string,
): Promise<{ ok: true; value: T } | { ok: false; reason: string }> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    const value = await Promise.race<T | typeof TIMEOUT_SENTINEL>([
      inner,
      new Promise<typeof TIMEOUT_SENTINEL>((resolve) => {
        timer = setTimeout(() => resolve(TIMEOUT_SENTINEL), timeoutMs)
      }),
    ])
    if (value === TIMEOUT_SENTINEL) {
      return { ok: false, reason: `${label} timeout after ${timeoutMs}ms` }
    }
    return { ok: true, value: value as T }
  } catch (err) {
    return { ok: false, reason: err instanceof Error ? err.message : String(err) }
  } finally {
    if (timer) clearTimeout(timer)
  }
}

const TIMEOUT_SENTINEL = Symbol("ergon-plugin-timeout")

function extractText(parts: any[]): string {
  if (!Array.isArray(parts)) return ""
  return parts
    .filter((p) => p && p.type === "text" && typeof p.text === "string")
    .map((p) => p.text)
    .join("\n")
    .trim()
}

async function handleSessionIdle(
  sessionID: string,
  client: any,
  steward: StewardClient,
  memory: MemoryClient,
  lastJudgedAssistantId: Map<string, string>,
): Promise<void> {
  let result: any
  try {
    result = await client.session.messages({ path: { id: sessionID } })
  } catch {
    return
  }
  const messages = result?.data
  if (!Array.isArray(messages) || messages.length < 2) return

  // Walk backwards to find the most recent assistant response and the
  // user message that triggered it.
  let lastAssistantIdx = -1
  let lastUserIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i]?.info?.role
    if (lastAssistantIdx === -1 && role === "assistant") {
      lastAssistantIdx = i
      continue
    }
    if (lastAssistantIdx !== -1 && role === "user") {
      lastUserIdx = i
      break
    }
  }
  if (lastUserIdx === -1 || lastAssistantIdx === -1) return
  if (lastAssistantIdx < lastUserIdx) return

  // Dedup: skip if this exact assistant message has already been judged
  // for this session. session.idle normally fires once per turn, but a
  // defensive re-fire (or any future opencode change) would otherwise
  // re-submit the same pair to the steward — wasted LLM cost and risk
  // of duplicate memory writes.
  const assistantId = messages[lastAssistantIdx]?.info?.id
  if (typeof assistantId === "string" && assistantId.length > 0) {
    if (lastJudgedAssistantId.get(sessionID) === assistantId) return
    lastJudgedAssistantId.set(sessionID, assistantId)
  }

  const userText = extractText(messages[lastUserIdx].parts)
  const assistantText = extractText(messages[lastAssistantIdx].parts)
  if (!userText || !assistantText) return

  const saved = await steward.judgeSave(userText, assistantText)
  if (saved) await memory.save(saved)
}

export function createErgonPlugin(deps: ErgonPluginDeps = {}): Plugin {
  return async ({ client, directory }) => {
    const steward = deps.steward ?? createStewardClient()

    // Memory is an HTTP client against an external openmemory-js HTTP
    // service (see prompts/openmemory.service). Construction is just
    // building a thin fetch wrapper — no I/O, no module loading, no
    // ambient noise. We cache it once per plugin instance.
    let memoryCache: MemoryClient | null = null
    function getMemory(): MemoryClient {
      if (deps.memory) return deps.memory
      if (memoryCache) return memoryCache
      memoryCache = createMemoryClient()
      return memoryCache
    }

    // Cross-hook state for the recall path. The chat.message hook is the
    // only place we have access to the user's message text, but injecting
    // recall content there as an extra TextPart causes opencode to send
    // a multi-content user message which the Qwen3.5 Jinja chat template
    // rejects ("System message must be at the beginning"). Instead, the
    // chat.message hook stashes the pre-rendered recall block here, keyed
    // by sessionID, and experimental.chat.system.transform consumes it on
    // the same turn — putting the recalled notes into the system prompt
    // alongside the scratchpad, where they actually belong.
    const pendingRecall = new Map<string, string>()

    // Per-session memo of the assistant message id we most recently judged.
    // session.idle can re-fire for the same exchange (defensive idle set in
    // opencode); without this, the steward LLM would be called repeatedly
    // for identical input. Cleared on session.deleted with pendingRecall.
    const lastJudgedAssistantId = new Map<string, string>()

    function readScratchpad(): string | null {
      const p = join(directory, ".ergon.studio", "scratchpad.md")
      return existsSync(p) ? readFileSync(p, "utf8") : null
    }

    // Fetch the live agent list once at plugin init so the run_parallel
    // schema can constrain to whatever opencode would actually accept.
    // Without this, the LLM hallucinates names like "bash" (confusing
    // run_parallel with the bash built-in tool) and the call surfaces as
    // an `Agent not found: "bash"` error from inside opencode. By querying
    // GET /agent up front and building a Zod enum from the result, the
    // bad call gets rejected at the harness boundary with a schema error
    // the LLM can self-correct from.
    //
    // CRITICAL: opencode loads plugins synchronously at startup. If this
    // call hangs (server still warming up, slow MCP, network hiccup),
    // opencode's entire startup blocks with it — the user sees a frozen
    // process with no error. Always bound the lookup with a timeout and
    // fall back to the permissive string schema if it expires, fails, or
    // the harness doesn't expose app.agents at all.
    const AGENT_LOOKUP_TIMEOUT_MS = deps.agentLookupTimeoutMs ?? 15000
    const chatMessageTimeoutMs = deps.chatMessageTimeoutMs ?? 5000
    let agentNames: string[] | null = null
    let fallbackReason: string | null = null
    try {
      const anyClient = client as any
      if (typeof anyClient?.app?.agents !== "function") {
        fallbackReason = "client.app.agents unavailable"
      } else {
        let timer: ReturnType<typeof setTimeout> | undefined
        const timeoutPromise = new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error(`agent lookup timeout after ${AGENT_LOOKUP_TIMEOUT_MS}ms`)),
            AGENT_LOOKUP_TIMEOUT_MS,
          )
        })
        try {
          const res: any = await Promise.race([anyClient.app.agents(), timeoutPromise])
          const list = res?.data
          if (Array.isArray(list)) {
            const names = list
              .map((a: any) => a?.name)
              .filter((n: any): n is string => typeof n === "string" && n.length > 0)
            if (names.length > 0) agentNames = names
            else fallbackReason = "agent lookup returned no usable names"
          } else {
            fallbackReason = "agent lookup returned non-array data"
          }
        } finally {
          if (timer) clearTimeout(timer)
        }
      }
    } catch (err) {
      fallbackReason = err instanceof Error ? err.message : String(err)
    }

    if (fallbackReason) {
      // Make the fallback observable. Without this the user sees an
      // unconstrained schema and has no idea why — silent degradation
      // is the worst kind.
      try {
        await client.app.log({
          body: {
            service: "ergon-plugin",
            level: "warn",
            message: `run_parallel agent enum disabled (fallback to string): ${fallbackReason}`,
          },
        })
      } catch {
        /* logging itself is best-effort */
      }
    }

    const agentArg = agentNames
      ? tool.schema
          .enum(agentNames as [string, ...string[]])
          .describe("Agent name to run")
      : tool.schema.string().describe("Agent name to run")

    const validAgentList = agentNames
      ? `Valid agent names: ${agentNames.join(", ")}. `
      : ""

    return {
      event: async ({ event }) => {
        if (event.type === "session.created") {
          await client.app.log({
            body: {
              service: "ergon-plugin",
              level: "info",
              message: "Ergon session started",
            },
          })
        }

        if (event.type === "session.idle") {
          // Save path: fire-and-forget so this hook never blocks the loop.
          // Failures are intentionally swallowed inside handleSessionIdle.
          const { sessionID } = (event as any).properties ?? {}
          if (typeof sessionID === "string" && sessionID.length > 0) {
            void handleSessionIdle(sessionID, client, steward, getMemory(), lastJudgedAssistantId).catch(() => {})
          }
        }

        if (event.type === "session.deleted") {
          // pendingRecall is populated by chat.message and consumed by
          // experimental.chat.system.transform on the same turn. If a turn
          // is aborted/errored before transform runs, the entry is orphaned.
          // Purge on session.deleted so the maps can't grow unbounded across
          // a long-lived opencode process. Same goes for the dedup memo.
          const { sessionID } = (event as any).properties ?? {}
          if (typeof sessionID === "string" && sessionID.length > 0) {
            pendingRecall.delete(sessionID)
            lastJudgedAssistantId.delete(sessionID)
          }
        }
      },

      // Recall path, half 1: see the user's message, rewrite it via the
      // steward, query openmemory, and stash the rendered recall block in
      // pendingRecall. The actual prompt injection happens in
      // experimental.chat.system.transform below — see the comment on
      // pendingRecall above for why we can't push directly here.
      //
      // CRITICAL: opencode awaits this hook (per sst/opencode#16879 the
      // session.idle path is fire-and-forget but chat.message is awaited),
      // so a hung steward or memory backend stalls every user turn. Both
      // external calls are bounded with chatMessageTimeoutMs (default 5s).
      // On timeout we log and skip recall — losing one turn's memory is
      // strictly better than blocking the user indefinitely.
      "chat.message": async (input, output) => {
        const userText = extractText(output.parts as any[])
        if (!userText) return

        const queryResult = await raceWithTimeout(
          steward.rewriteQuery(userText),
          chatMessageTimeoutMs,
          "steward.rewriteQuery",
        )
        if (!queryResult.ok) {
          try {
            await client.app.log({
              body: {
                service: "ergon-plugin",
                level: "warn",
                message: `chat.message recall disabled (fallback): ${queryResult.reason}`,
              },
            })
          } catch { /* logging is best-effort */ }
          return
        }
        const query = queryResult.value
        if (!query) return

        const memory = getMemory()
        const recallResult = await raceWithTimeout(
          memory.recall(query),
          chatMessageTimeoutMs,
          "memory.recall",
        )
        if (!recallResult.ok) {
          try {
            await client.app.log({
              body: {
                service: "ergon-plugin",
                level: "warn",
                message: `chat.message recall disabled (fallback): ${recallResult.reason}`,
              },
            })
          } catch { /* logging is best-effort */ }
          return
        }
        const memories = recallResult.value
        if (memories.length === 0) return

        const block =
          "## Relevant prior notes (from memory steward)\n\n" +
          memories.map((m) => `- ${m.content}`).join("\n")

        pendingRecall.set(input.sessionID, block)

        // Surface the recall in opencode logs so it's observable even
        // though the content lands in the system prompt (invisible in TUI).
        try {
          await client.app.log({
            body: {
              service: "ergon-plugin",
              level: "info",
              message: `memory recall: query="${query}" hits=${memories.length}`,
            },
          })
        } catch {
          /* logging is best-effort, never break the hook */
        }
      },

      // System-prompt augmentation: scratchpad + memory-steward recall.
      //
      // CRITICAL: we must NOT push new entries to output.system. Opencode's
      // llm.ts maps each entry in `system: string[]` to its own role-system
      // message in the chat completion request, and strict chat templates
      // (Qwen 3.5 in particular) reject any conversation with more than one
      // system message or with a system message at index ≥ 1. Instead, we
      // collect our additions and append them to the LAST existing system
      // entry — keeping everything in a single role-system message at index 0.
      "experimental.chat.system.transform": async (input, output) => {
        const additions: string[] = []

        const scratchpad = readScratchpad()
        if (scratchpad) {
          additions.push(`## Project Scratchpad\n\n${scratchpad}`)
        } else {
          additions.push(
            `## Project Scratchpad\n\nNo scratchpad yet for this project. ` +
            `When you discover something worth keeping (a constraint, a gotcha, a decision and why), ` +
            `create \`.ergon.studio/scratchpad.md\` with \`## Conventions\`, \`## Notes\`, and \`## Decisions\` sections.`
          )
        }

        // Recall path, half 2: pick up whatever the chat.message hook
        // stashed for this session.
        const sessionID = (input as any)?.sessionID
        if (typeof sessionID === "string" && pendingRecall.has(sessionID)) {
          additions.push(pendingRecall.get(sessionID)!)
          pendingRecall.delete(sessionID)
        }

        if (additions.length === 0) return

        const preExisting = output.system.length
        const combined = additions.join("\n\n")

        if (preExisting > 0) {
          // Append to the last entry — no new system message gets created.
          const lastIdx = preExisting - 1
          output.system[lastIdx] = `${output.system[lastIdx]}\n\n${combined}`
        } else {
          output.system.push(combined)
        }

        // Diagnostic: surface what we observed so we can verify in opencode
        // logs that we never end up with more than one system message.
        try {
          await client.app.log({
            body: {
              service: "ergon-plugin",
              level: "info",
              message: `system.transform: pre-existing=${preExisting}, additions=${additions.length}, final=${output.system.length}`,
            },
          })
        } catch {
          /* logging is best-effort */
        }
      },

      // Re-inject scratchpad when context is compacted so it survives long sessions.
      "experimental.session.compacting": async (_input, output) => {
        const scratchpad = readScratchpad()
        if (scratchpad) {
          output.context.push(`## Project Scratchpad\n\n${scratchpad}`)
        } else {
          output.context.push(
            `## Project Scratchpad\n\nNo scratchpad yet. ` +
            `Create \`.ergon.studio/scratchpad.md\` with \`## Conventions\`, \`## Notes\`, and \`## Decisions\` sections when you have something worth keeping.`
          )
        }
      },

      tool: {
        run_parallel: tool({
          description:
            "Run multiple agents in parallel and return their combined output. " +
            "Each task specifies an agent name and a brief. All tasks execute concurrently. " +
            validAgentList +
            "This tool delegates to LLM agents — it is NOT a way to run shell commands or built-in tools. " +
            "Avoid using write-capable agents (e.g. coder) in parallel — they may conflict on shared files.",
          args: {
            tasks: tool.schema
              .array(
                tool.schema.object({
                  agent: agentArg,
                  brief: tool.schema.string().describe("Full brief to send to the agent"),
                })
              )
              .min(1)
              .describe("List of agent+brief pairs to run in parallel (at least one)"),
          },
          async execute(args, context) {
            const results = await Promise.all(
              args.tasks.map(async (task) => {
                let id: string | undefined
                try {
                  const created = await client.session.create({
                    body: {
                      title: `${task.agent} (parallel)`,
                      parentID: context.sessionID,
                    },
                  })
                  id = created.data?.id
                  if (!id) throw new Error("Failed to create session")

                  const response = await client.session.prompt({
                    path: { id },
                    body: {
                      agent: task.agent,
                      parts: [{ type: "text", text: task.brief }],
                    },
                  })

                  await client.session.delete({ path: { id } })

                  const text = response.data?.parts
                    ?.filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
                    .map((p) => p.text)
                    .join("\n") ?? ""

                  return `## ${task.agent}\n\n${text}`
                } catch (err) {
                  if (id) {
                    try { await client.session.delete({ path: { id } }) } catch {}
                  }
                  const message = err instanceof Error ? err.message : String(err)
                  return `## ${task.agent}\n\n⚠️ Task failed: ${message}`
                }
              })
            )

            return results.join("\n\n---\n\n")
          },
        }),
      },
    }
  }
}

// Default plugin export — uses real steward + openmemory-js by default.
export const ErgonPlugin: Plugin = createErgonPlugin()
