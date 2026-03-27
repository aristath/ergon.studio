import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

export const ErgonPlugin: Plugin = async ({ client }) => {
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
              const created = await client.session.create({
                body: {
                  title: `${task.agent} (parallel)`,
                  parentID: context.sessionID,
                },
              })
              const id = created.data?.id
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
            })
          )

          return results.join("\n\n---\n\n")
        },
      }),
    },
  }
}

