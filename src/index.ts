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
   * Maximum time (ms) to wait for each external stage in the chat.message
   * recall path. The steward rewrite and memory recall each receive this full
   * budget, so the complete path may take about twice this value. Defaults to
   * 5000ms. The hook is awaited by opencode, so these per-stage timeouts
   * protect against a hung steward or memory backend.
   */
  chatMessageTimeoutMs?: number
  /**
   * Maximum time (ms) to wait for a provider's models endpoint when
   * validating agent model references in the config hook. Defaults to 3000ms.
   * The config hook is awaited during opencode startup, so a slow or dead
   * provider must not stall the launch — on timeout we leave the reference
   * untouched. Tests inject small values to keep the suite fast.
   */
  providerLookupTimeoutMs?: number
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

function parseModelRef(model: unknown): { providerID: string; modelID: string } | null {
  if (typeof model !== "string") return null
  const slashIdx = model.indexOf("/")
  if (slashIdx <= 0 || slashIdx === model.length - 1) return null
  return {
    providerID: model.slice(0, slashIdx),
    modelID: model.slice(slashIdx + 1),
  }
}

async function logBestEffort(client: any, level: "info" | "warn", message: string): Promise<void> {
  try {
    await client.app.log({
      body: {
        service: "ergon-plugin",
        level,
        message,
      },
    })
  } catch {
    /* logging itself is best-effort */
  }
}

// Ask a provider's own endpoint which models it actually serves. The
// provider's coordinates (baseURL, optional apiKey) come straight from the
// opencode config we're handed — so this works for any OpenAI-compatible
// provider and never calls back into opencode. (Asking opencode's own
// provider API from the config hook re-enters its boot and deadlocks the
// launch.) Returns the set of model ids the provider reports, or null if we
// couldn't reach it / it answered with something unusable.
async function fetchProviderModels(
  config: any,
  providerID: string,
  timeoutMs: number,
): Promise<Set<string> | null> {
  const provider = config?.provider?.[providerID]
  const baseURL = provider?.options?.baseURL
  if (typeof baseURL !== "string" || baseURL.length === 0) return null

  const url = `${baseURL.replace(/\/+$/, "")}/models`
  const headers: Record<string, string> = { accept: "application/json" }
  const apiKey = provider?.options?.apiKey
  if (typeof apiKey === "string" && apiKey.length > 0) {
    headers.authorization = `Bearer ${apiKey}`
  }

  const r = await raceWithTimeout(
    fetch(url, { headers }).then((res) => (res.ok ? res.json() : null)),
    timeoutMs,
    `models@${providerID}`,
  )
  if (!r.ok) return null

  // OpenAI-compatible shape: { data: [{ id }, ...] }. Tolerate a bare array
  // and bare string ids too.
  const value = r.value as any
  const list = Array.isArray(value) ? value : value?.data
  if (!Array.isArray(list)) return null

  const ids = new Set<string>()
  for (const m of list) {
    const id = typeof m === "string" ? m : m?.id
    if (typeof id === "string" && id.length > 0) ids.add(id)
  }
  return ids
}

async function sanitizeAgentModels(
  config: any,
  client: any,
  timeoutMs: number,
): Promise<void> {
  const agentGroups = [
    ["agent", config?.agent],
    ["mode", config?.mode],
  ] as const

  // Collect every agent/mode that pins a model. A reference that isn't even
  // "provider/model" can't be checked against any provider, so drop it now —
  // opencode falls back to the default model.
  const refs: Array<{
    group: string
    agent: string
    model: string
    parsed: { providerID: string; modelID: string }
    target: any
  }> = []

  for (const [group, agents] of agentGroups) {
    if (!agents || typeof agents !== "object") continue
    for (const [agent, target] of Object.entries(agents)) {
      if (!target || typeof target !== "object") continue
      if (!Object.prototype.hasOwnProperty.call(target, "model")) continue
      const model = (target as any).model
      const parsed = parseModelRef(model)
      if (!parsed) {
        delete (target as any).model
        await logBestEffort(
          client,
          "warn",
          `agent "${agent}" ${group}.model ignored (invalid model reference); using default model`,
        )
        continue
      }
      refs.push({ group, agent, model, parsed, target })
    }
  }

  if (refs.length === 0) return

  // One probe per distinct provider — ask each provider what it actually has.
  const modelsByProvider = new Map<string, Set<string> | null>()
  for (const providerID of new Set(refs.map((r) => r.parsed.providerID))) {
    modelsByProvider.set(providerID, await fetchProviderModels(config, providerID, timeoutMs))
  }

  for (const ref of refs) {
    const available = modelsByProvider.get(ref.parsed.providerID)
    if (!available) {
      // Provider unreachable / no usable list — don't guess. Only act when we
      // actually know the model isn't there.
      await logBestEffort(
        client,
        "warn",
        `agent "${ref.agent}" ${ref.group}.model "${ref.model}" not validated; provider "${ref.parsed.providerID}" returned no model list`,
      )
      continue
    }
    if (available.has(ref.parsed.modelID)) continue
    delete ref.target.model
    await logBestEffort(
      client,
      "warn",
      `agent "${ref.agent}" ${ref.group}.model "${ref.model}" not found on provider; using default model`,
    )
  }
}

function extractText(parts: any[]): string {
  if (!Array.isArray(parts)) return ""
  return parts
    .filter((p) => p && p.type === "text" && typeof p.text === "string")
    .map((p) => p.text)
    .join("\n")
    .trim()
}

type DebateVerdict = "AGREE" | "CONTINUE" | "BLOCKED"

function parseDebateVerdict(text: string): DebateVerdict {
  const lastLine = text.trim().split(/\r?\n/).at(-1)?.trim() ?? ""
  const match = lastLine.match(/^Verdict:\s*(AGREE|CONTINUE|BLOCKED)$/i)
  if (!match) return "CONTINUE"
  return match[1].toUpperCase() as DebateVerdict
}

function debatePrompt(
  task: string,
  selfAgent: string,
  peerAgent: string,
  peerOutput?: string,
): string {
  const verdict =
    "End with exactly one line: Verdict: AGREE, Verdict: CONTINUE, or Verdict: BLOCKED."

  if (!peerOutput) {
    return [
      `You are ${selfAgent} in a two-agent coding debate with ${peerAgent}.`,
      "",
      "Task:",
      task,
      "",
      "Do the first pass. If code changes are needed, make the changes you believe are right.",
      "Keep the response focused: what you did or propose, why, and what needs review.",
      "Use Verdict: CONTINUE unless you cannot proceed without the user.",
      verdict,
    ].join("\n")
  }

  return [
    `You are ${selfAgent} in a two-agent coding debate with ${peerAgent}.`,
    "",
    "Task:",
    task,
    "",
    `${peerAgent}'s latest response:`,
    peerOutput,
    "",
    "Review it directly. If you agree it is optimal, say why and use Verdict: AGREE.",
    "If it needs work and you can improve it, make the changes or propose the correction, then use Verdict: CONTINUE.",
    "If the user needs to decide something before progress is possible, use Verdict: BLOCKED and name the decision.",
    verdict,
  ].join("\n")
}

function renderDebateTranscript(input: {
  status: DebateVerdict | "FAILED" | "MAX_TURNS"
  agentA: string
  agentB: string
  entries: Array<{ turn: number; agent: string; text: string; verdict: DebateVerdict }>
  error?: string
}): string {
  const lines = [
    "# Debate result",
    "",
    `Status: ${input.status}`,
    `Participants: ${input.agentA}, ${input.agentB}`,
    `Turns: ${input.entries.length}`,
  ]

  if (input.error) {
    lines.push("", "## Error", "", input.error)
  }

  const latest = input.entries.at(-1)
  if (latest) {
    lines.push("", "## Latest response", "", `### ${latest.agent}`, "", latest.text)
  }

  lines.push("", "## Transcript")
  for (const entry of input.entries) {
    lines.push("", `### Turn ${entry.turn} - ${entry.agent}`, "", entry.text)
  }

  return lines.join("\n")
}

async function handleSessionIdle(
  sessionID: string,
  client: any,
  steward: StewardClient,
  memory: MemoryClient,
  lastAttemptedAssistantId: Map<string, string>,
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

  // Dedup: skip if this exact assistant message already had a save attempt
  // for this session. session.idle normally fires once per turn, but a
  // defensive re-fire (or any future opencode change) would otherwise
  // re-submit the same pair to the steward — wasted LLM cost and risk
  // of duplicate memory writes.
  const assistantId = messages[lastAssistantIdx]?.info?.id
  if (typeof assistantId === "string" && assistantId.length > 0) {
    if (lastAttemptedAssistantId.get(sessionID) === assistantId) return
    lastAttemptedAssistantId.set(sessionID, assistantId)
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

    // Per-session memo of the assistant message id whose save we last attempted.
    // session.idle can re-fire for the same exchange (defensive idle set in
    // opencode); without this, the steward LLM would be called repeatedly
    // for identical input. Cleared on session.deleted with pendingRecall.
    const lastAttemptedAssistantId = new Map<string, string>()

    function readScratchpad(): string | null {
      const p = join(directory, ".ergon.studio", "scratchpad.md")
      return existsSync(p) ? readFileSync(p, "utf8") : null
    }

    const chatMessageTimeoutMs = deps.chatMessageTimeoutMs ?? 5000
    const providerLookupTimeoutMs = deps.providerLookupTimeoutMs ?? 3000
    const agentArg = tool.schema.string().describe("Agent name to run")

    return {
      config: async (config) => {
        await sanitizeAgentModels(config, client, providerLookupTimeoutMs)
      },

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
            void handleSessionIdle(sessionID, client, steward, getMemory(), lastAttemptedAssistantId).catch(() => {})
          }
        }

        if (event.type === "session.deleted") {
          // pendingRecall is populated by chat.message and consumed by
          // experimental.chat.system.transform on the same turn. If a turn
          // is aborted/errored before transform runs, the next chat.message
          // invalidates it. Purge here too for sessions that never receive
          // another message. Same goes for the dedup memo.
          const { sessionID } = (event as any).properties ?? {}
          if (typeof sessionID === "string" && sessionID.length > 0) {
            pendingRecall.delete(sessionID)
            lastAttemptedAssistantId.delete(sessionID)
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
      // external calls are each bounded with chatMessageTimeoutMs (default
      // 5s per stage, or about 10s for both stages in the worst case).
      // On timeout we log and skip recall — losing one turn's memory is
      // strictly better than blocking the user indefinitely.
      "chat.message": async (input, output) => {
        // A prior turn may have stopped after stashing recall but before
        // system.transform consumed it. Never let that orphaned context cross
        // into this user message, including when the current recall exits early.
        pendingRecall.delete(input.sessionID)

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
        debate: tool({
          description:
            "Run two agents in an alternating coding debate. " +
            "Agent A does the first pass, Agent B reviews or improves it, then they alternate until one agrees, blocks, or max_turns is reached. " +
            "Use this when you want intentional cross-review and convergence, not isolated parallel opinions.",
          args: {
            agent_a: agentArg.describe("First agent. This agent takes the first turn."),
            agent_b: agentArg.describe("Second agent. This agent reviews or improves the first turn."),
            task: tool.schema.string().describe("Short, specific task or question for the debate"),
            max_turns: tool.schema
              .number()
              .int()
              .min(2)
              .max(12)
              .optional()
              .describe("Maximum total agent turns, including the first turn. Defaults to 6."),
          },
          async execute(args, context) {
            const maxTurns = args.max_turns ?? 6
            const entries: Array<{ turn: number; agent: string; text: string; verdict: DebateVerdict }> = []
            const sessions: Array<{ id: string; agent: string }> = []

            try {
              for (const agent of [args.agent_a, args.agent_b]) {
                const created = await client.session.create({
                  body: {
                    title: `${agent} (debate)`,
                    parentID: context.sessionID,
                  },
                })
                const id = created.data?.id
                if (!id) throw new Error(`Failed to create debate session for ${agent}`)
                sessions.push({ id, agent })
              }

              let latestText = ""
              let status: DebateVerdict | "MAX_TURNS" = "MAX_TURNS"

              for (let turn = 1; turn <= maxTurns; turn++) {
                const current = sessions[(turn - 1) % 2]
                const peer = sessions[turn % 2]
                const prompt = debatePrompt(
                  args.task,
                  current.agent,
                  peer.agent,
                  turn === 1 ? undefined : latestText,
                )

                const response = await client.session.prompt({
                  path: { id: current.id },
                  body: {
                    agent: current.agent,
                    parts: [{ type: "text", text: prompt }],
                  },
                })

                const text = extractText(response.data?.parts as any[])
                const verdict = parseDebateVerdict(text)
                entries.push({ turn, agent: current.agent, text, verdict })
                latestText = text

                if (turn > 1 && (verdict === "AGREE" || verdict === "BLOCKED")) {
                  status = verdict
                  break
                }
              }

              return renderDebateTranscript({
                status,
                agentA: args.agent_a,
                agentB: args.agent_b,
                entries,
              })
            } catch (err) {
              return renderDebateTranscript({
                status: "FAILED",
                agentA: args.agent_a,
                agentB: args.agent_b,
                entries,
                error: err instanceof Error ? err.message : String(err),
              })
            } finally {
              await Promise.all(
                sessions.map(async ({ id }) => {
                  try { await client.session.delete({ path: { id } }) } catch {}
                })
              )
            }
          },
        }),
        run_parallel: tool({
          description:
            "Run multiple agents in parallel and return their combined output. " +
            "Each task specifies an agent name and a brief. All tasks execute concurrently. " +
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
