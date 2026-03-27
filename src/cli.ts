#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { join, resolve } from "path"

type AgentConfig = { tools?: Record<string, boolean>; [key: string]: unknown }
type Config = {
  $schema?: string
  plugins?: string[]
  default_agent?: string
  agent?: Record<string, AgentConfig>
  [key: string]: unknown
}

const command = process.argv[2]

if (command === "init") {
  const agentsSource = join(__dirname, "..", "agents")

  if (!existsSync(agentsSource)) {
    console.error(`Error: agents directory not found at ${agentsSource}`)
    process.exit(1)
  }

  // Install agent files
  const agentsDest = resolve(process.cwd(), ".opencode", "agents")
  mkdirSync(agentsDest, { recursive: true })
  cpSync(agentsSource, agentsDest, { recursive: true })
  console.log(`Ergon agents installed to ${agentsDest}`)

  // Merge opencode.json
  const configPath = resolve(process.cwd(), "opencode.json")
  const config: Config = existsSync(configPath)
    ? (JSON.parse(readFileSync(configPath, "utf8")) as Config)
    : { $schema: "https://opencode.ai/config.json" }

  // Add plugin if not present
  config.plugins ??= []
  if (!config.plugins.includes("ergon-studio")) {
    config.plugins.push("ergon-studio")
  }

  // Set default agent if not already set
  config.default_agent ??= "orchestrator"

  // Apply tool restrictions for read-only agents
  config.agent ??= {}
  config.agent.architect = {
    ...config.agent.architect,
    tools: { write: false, edit: false, bash: false },
  }
  config.agent.reviewer = {
    ...config.agent.reviewer,
    tools: { write: false, edit: false },
  }
  config.agent.critic = {
    ...config.agent.critic,
    tools: { write: false, edit: false },
  }
  config.agent.researcher = {
    ...config.agent.researcher,
    tools: { write: false, edit: false },
  }

  writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n")
  console.log(`opencode.json updated at ${configPath}`)
} else {
  console.log("Usage: ergon init")
}
