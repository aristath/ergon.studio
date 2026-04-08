import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import { existsSync, readFileSync } from "fs"
import { join } from "path"
import { createStewardClient, type StewardClient } from "./steward.js"
import { createMemoryClient, type MemoryClient } from "./memory.js"

export interface ErgonPluginDeps {
  steward?: StewardClient
  memory?: MemoryClient
}

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

    function readScratchpad(): string | null {
      const p = join(directory, ".ergon.studio", "scratchpad.md")
      return existsSync(p) ? readFileSync(p, "utf8") : null
    }

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
            void handleSessionIdle(sessionID, client, steward, getMemory()).catch(() => {})
          }
        }
      },

      // Recall path, half 1: see the user's message, rewrite it via the
      // steward, query openmemory, and stash the rendered recall block in
      // pendingRecall. The actual prompt injection happens in
      // experimental.chat.system.transform below — see the comment on
      // pendingRecall above for why we can't push directly here.
      "chat.message": async (input, output) => {
        const userText = extractText(output.parts as any[])
        if (!userText) return

        const query = await steward.rewriteQuery(userText)
        if (!query) return

        const memory = getMemory()
        const memories = await memory.recall(query)
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
            "Each task specifies an agent and a brief. All tasks execute concurrently. " +
            "Avoid using write-capable agents (e.g. coder) in parallel — they may conflict on shared files.",
          args: {
            tasks: tool.schema.array(
              tool.schema.object({
                agent: tool.schema.string().describe("Agent name to run"),
                brief: tool.schema.string().describe("Full brief to send to the agent"),
              })
            ).describe("List of agent+brief pairs to run in parallel"),
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
