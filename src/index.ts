import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"
import { existsSync, readFileSync } from "fs"
import { join } from "path"

export const ErgonPlugin: Plugin = async ({ client, directory }) => {
  function readConventions(): string | null {
    const p = join(directory, ".ergon.studio", "scratchpads", "conventions.md")
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
    },

    // Inject project conventions into every agent's system prompt automatically.
    // Agents see the conventions without needing to load the scratchpad skill.
    "experimental.chat.system.transform": async (_input, output) => {
      const conventions = readConventions()
      if (conventions) {
        output.system.push(
          `## Project Conventions\n\nThe following conventions apply to this project. Follow them without being asked.\n\n${conventions}`
        )
      }
    },

    // Re-inject conventions when context is compacted so they survive long sessions.
    "experimental.session.compacting": async (_input, output) => {
      const conventions = readConventions()
      if (conventions) {
        output.context.push(`## Project Conventions\n\n${conventions}`)
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
