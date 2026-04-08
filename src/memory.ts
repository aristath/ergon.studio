// HTTP client for openmemory-js running as a standalone service.
//
// This module deliberately does NOT import openmemory-js as a library.
// openmemory-js's library entry (`dist/index.js`) eagerly loads its
// dashboard HTTP server module, which prints [CONFIG]/[SERVER]/[DB]/
// [Vector]/[decay-2.0] lines to stdout/stderr at module-load and on
// every search/add call. In a long-running process like opencode that
// trashes the TUI. The clean fix is to run openmemory-js as its own
// systemd-managed HTTP service (see prompts/openmemory.service) and
// talk to it over the wire — its noise then goes to a log file and
// never touches anyone else's terminal.
//
// HTTP API surface (verified live against openmemory-js v1.3.3 serving):
//
//   POST /memory/add    { content, user_id?, tags? }
//   POST /memory/query  { query, k, filters? }  →  { query, matches: [...] }
//
// Each match has at least { id, content, score, salience, sectors,
// primary_sector }. We pull only id/content/score; the rest is for
// future use. Failures of either call are non-fatal: recall returns []
// and save silently does nothing. The plugin's recall and save paths
// already degrade gracefully when the memory client returns nothing.

export interface MemoryItem {
  id: string
  content: string
  score?: number
}

export interface MemoryClient {
  recall(query: string, limit?: number): Promise<MemoryItem[]>
  save(content: string): Promise<void>
}

export interface MemoryClientOptions {
  baseURL?: string
  userID?: string
  defaultLimit?: number
  fetch?: typeof fetch
}

export const DEFAULT_MEMORY_URL = "http://127.0.0.1:8082"
export const DEFAULT_RECALL_LIMIT = 5

export function createMemoryClient(opts: MemoryClientOptions = {}): MemoryClient {
  const baseURL = (opts.baseURL ?? DEFAULT_MEMORY_URL).replace(/\/$/, "")
  const userID = opts.userID
  const defaultLimit = opts.defaultLimit ?? DEFAULT_RECALL_LIMIT
  const fetchImpl = opts.fetch ?? fetch

  return {
    async recall(query: string, limit?: number): Promise<MemoryItem[]> {
      if (!query || query.trim().length === 0) return []
      try {
        const body: Record<string, unknown> = { query, k: limit ?? defaultLimit }
        if (userID) body.filters = { user_id: userID }
        const res = await fetchImpl(`${baseURL}/memory/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        })
        if (!res.ok) return []
        const data: any = await res.json()
        // openmemory-js returns matches under `matches`, not `memories`.
        // Accept either key for forward/backward compatibility.
        const items = Array.isArray(data?.matches)
          ? data.matches
          : Array.isArray(data?.memories)
          ? data.memories
          : null
        if (!items) return []
        return items
          .filter((m: any) => m && typeof m === "object" && typeof m.content === "string" && m.content.length > 0)
          .map((m: any) => ({
            id: String(m.id ?? ""),
            content: String(m.content),
            score: typeof m.score === "number" ? m.score : undefined,
          }))
      } catch {
        return []
      }
    },

    async save(content: string): Promise<void> {
      if (!content || content.trim().length === 0) return
      try {
        const body: Record<string, unknown> = { content: content.trim() }
        if (userID) body.user_id = userID
        await fetchImpl(`${baseURL}/memory/add`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        })
      } catch {
        // Save errors are non-fatal — the steward fires post-turn and
        // we never want a memory write failure to surface to the user.
      }
    },
  }
}
