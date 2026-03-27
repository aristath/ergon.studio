import type { Plugin } from "@opencode-ai/plugin"

/**
 * Minimal Opencode plugin.
 * Registers an `event` handler that logs when a new session is created.
 */
export const ErgonPlugin: Plugin = async ({ client }) => {
  return {
    // Fires on every Opencode event. We only act on session creation.
    event: async ({ event }) => {
      if (event.type === "session.created") {
        // Structured log entry – useful for debugging and monitoring.
        await client.app.log({
          body: {
            service: "ergon-plugin",
            level: "info",
            message: "Ergon session started",
          },
        })
      }
    },
  }
}

