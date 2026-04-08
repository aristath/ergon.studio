// Memory steward HTTP client.
//
// The steward is a small (2B) LLM running in its own llama-server instance,
// permanently resident on a dedicated GPU. It does two narrow jobs:
//
//   1. rewriteQuery: turn a noisy user message into a tight search query
//      for openmemory. Returns null if the message has no searchable intent.
//
//   2. judgeSave:    look at one exchange (user message + assistant response)
//      and decide whether anything durable is worth remembering. Returns the
//      content to save, or null.
//
// Both the client config (URL, model, temperature) and the two prompts live
// in a single file: `prompts/steward.md`. Its YAML frontmatter holds the
// configuration, and its body has `## rewrite` and `## judge` sections with
// the prompts. A companion bash script `scripts/run-steward.sh` parses the
// *same* frontmatter at runtime to invoke llama-server — so editing
// prompts/steward.md changes both the service runtime and the ergon client
// in lockstep. That's the single source of truth.
//
// This module is dependency-injectable: callers can pass a custom fetch
// implementation (for tests) and override prompts.

import { readFileSync } from "fs"
import { fileURLToPath } from "url"
import { dirname, join } from "path"

// === Steward definition loader ===

interface StewardDefinition {
  config: Record<string, string | number>
  prompts: Record<string, string>
}

/**
 * Parses a `steward.md`-style file: YAML frontmatter followed by a body
 * containing one or more `## <name>` sections. Frontmatter supports simple
 * `key: value` pairs (strings and numbers), with `#` for comments.
 */
export function parseStewardMd(content: string): StewardDefinition {
  const match = content.match(/^---[ \t]*\n([\s\S]*?)\n---[ \t]*\n?([\s\S]*)$/)
  if (!match) {
    throw new Error("steward.md: missing or malformed YAML frontmatter")
  }
  const frontmatterRaw = match[1]
  const body = match[2]

  const config: Record<string, string | number> = {}
  for (const rawLine of frontmatterRaw.split("\n")) {
    const line = rawLine.trim()
    if (!line || line.startsWith("#")) continue
    const colonIdx = line.indexOf(":")
    if (colonIdx === -1) continue
    const key = line.slice(0, colonIdx).trim()
    let value: string | number = line.slice(colonIdx + 1).trim()
    // Strip trailing inline comments
    value = value.replace(/\s+#.*$/, "").trim()
    // Strip surrounding quotes
    value = value.replace(/^["']|["']$/g, "")
    // Parse numbers (integers and decimals)
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      value = Number(value)
    }
    config[key] = value
  }

  // Split body on `^## ` headings. parts[0] is the prologue (discarded);
  // parts[1..] each start with the heading text on line 1 followed by body.
  const prompts: Record<string, string> = {}
  const parts = body.split(/^##\s+/m)
  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]
    const firstNewline = part.indexOf("\n")
    if (firstNewline === -1) continue
    const name = part.slice(0, firstNewline).trim().toLowerCase().replace(/\s+/g, "-")
    const content = part.slice(firstNewline + 1).trim()
    if (name && content) {
      prompts[name] = content
    }
  }

  return { config, prompts }
}

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

function loadDefinition(): StewardDefinition {
  // dist/steward.js → project root is "..", then prompts/steward.md
  const absPath = join(__dirname, "..", "prompts", "steward.md")
  let raw: string
  try {
    raw = readFileSync(absPath, "utf8")
  } catch (err) {
    throw new Error(
      `ergon steward: cannot read ${absPath}. This file should ship with ergon-studio. ` +
      `If it is missing, your install is incomplete. (${(err as Error).message})`
    )
  }
  return parseStewardMd(raw)
}

const definition = loadDefinition()

function requirePrompt(name: string): string {
  const content = definition.prompts[name]
  if (!content) {
    throw new Error(`prompts/steward.md: missing '## ${name}' section in body`)
  }
  return content
}

// === Exported client types ===

export interface StewardClient {
  rewriteQuery(userMessage: string): Promise<string | null>
  judgeSave(userMessage: string, assistantResponse: string): Promise<string | null>
}

export interface StewardOptions {
  baseURL?: string
  model?: string
  fetch?: typeof fetch
  rewritePrompt?: string
  judgePrompt?: string
  temperature?: number
}

// === Defaults loaded from prompts/steward.md frontmatter ===

export const DEFAULT_STEWARD_URL: string = String(
  definition.config.url ?? "http://127.0.0.1:8081"
)
export const DEFAULT_STEWARD_MODEL: string = String(
  definition.config.model ?? "ergon-studio-memory-steward"
)
export const DEFAULT_TEMPERATURE: number =
  typeof definition.config.temperature === "number" ? definition.config.temperature : 0.3

// === Prompts loaded from prompts/steward.md body ===

export const REWRITE_PROMPT: string = requirePrompt("rewrite")
export const JUDGE_PROMPT: string = requirePrompt("judge")

export function createStewardClient(opts: StewardOptions = {}): StewardClient {
  const baseURL = opts.baseURL ?? DEFAULT_STEWARD_URL
  const model = opts.model ?? DEFAULT_STEWARD_MODEL
  const temperature = opts.temperature ?? DEFAULT_TEMPERATURE
  const fetchImpl = opts.fetch ?? fetch
  const rewritePrompt = opts.rewritePrompt ?? REWRITE_PROMPT
  const judgePrompt = opts.judgePrompt ?? JUDGE_PROMPT

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
      })
      if (!response.ok) return null
      const data: any = await response.json()
      const text = data?.choices?.[0]?.message?.content
      if (typeof text !== "string") return null
      // Strip any leading <think>...</think> block. Qwen 3.5 with
      // enable_thinking=false still emits empty <think></think> tags;
      // other thinking-capable models may emit real reasoning we want
      // to discard before parsing the actual answer.
      return text.replace(/^\s*<think>[\s\S]*?<\/think>\s*/i, "").trim()
    } catch {
      return null
    }
  }

  return {
    async rewriteQuery(userMessage: string): Promise<string | null> {
      const text = await complete(rewritePrompt, userMessage)
      if (text === null) return null
      if (text === "NONE" || text === "") return null
      return text
    },

    async judgeSave(userMessage: string, assistantResponse: string): Promise<string | null> {
      const exchange = `User: ${userMessage}\nAssistant: ${assistantResponse}`
      const text = await complete(judgePrompt, exchange)
      if (text === null) return null
      try {
        // Strip markdown code fences if the model wrapped the JSON
        const stripped = text.replace(/^```(?:json)?\s*|\s*```$/g, "").trim()
        const parsed = JSON.parse(stripped)
        if (parsed?.save && typeof parsed.save.content === "string") {
          return parsed.save.content
        }
        return null
      } catch {
        return null
      }
    },
  }
}
