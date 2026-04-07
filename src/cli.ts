#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { homedir } from "os"
import { join, dirname } from "path"
import { fileURLToPath } from "url"

const __dirname = dirname(fileURLToPath(import.meta.url))

type AgentConfig = { permission?: Record<string, string>; [key: string]: unknown }
type Config = {
  $schema?: string
  plugin?: string[]
  default_agent?: string
  agent?: Record<string, AgentConfig>
  [key: string]: unknown
}

function getOpencodeConfigDir(): string {
  const xdg = process.env.XDG_CONFIG_HOME
  return xdg ? join(xdg, "opencode") : join(homedir(), ".config", "opencode")
}

function installAgentsAndSkills(configDir: string): void {
  const agentsSource = join(__dirname, "..", "agents")
  if (!existsSync(agentsSource)) {
    console.error(`Error: agents directory not found at ${agentsSource}`)
    process.exit(1)
  }
  const agentsDest = join(configDir, "agents")
  mkdirSync(agentsDest, { recursive: true })
  cpSync(agentsSource, agentsDest, { recursive: true })
  console.log(`Ergon agents installed to ${agentsDest}`)

  const skillsSource = join(__dirname, "..", "skills")
  if (existsSync(skillsSource)) {
    const skillsDest = join(configDir, "skills")
    mkdirSync(skillsDest, { recursive: true })
    cpSync(skillsSource, skillsDest, { recursive: true })
    console.log(`Ergon skills installed to ${skillsDest}`)
  }
}

const command = process.argv[2]

if (command === "update") {
  const configDir = getOpencodeConfigDir()
  installAgentsAndSkills(configDir)
} else if (command === "init") {
  const configDir = getOpencodeConfigDir()

  installAgentsAndSkills(configDir)

  // Merge global opencode.json
  const configPath = join(configDir, "opencode.json")
  const config: Config = existsSync(configPath)
    ? (JSON.parse(readFileSync(configPath, "utf8")) as Config)
    : { $schema: "https://opencode.ai/config.json" }

  // Add plugin if not present
  config.plugin ??= []
  if (!config.plugin.includes("ergon-studio")) {
    config.plugin.push("ergon-studio")
  }

  // Set default agent if not already set
  config.default_agent ??= "orchestrator"

  // Apply permission restrictions for read-only agents
  config.agent ??= {}
  config.agent.architect = {
    ...config.agent.architect,
    permission: { edit: "deny", bash: "deny" },
  }
  config.agent.reviewer = {
    ...config.agent.reviewer,
    permission: { edit: "deny" },
  }
  config.agent.critic = {
    ...config.agent.critic,
    permission: { edit: "deny" },
  }
  config.agent.researcher = {
    ...config.agent.researcher,
    permission: { edit: "deny" },
  }
  config.agent.scout = {
    ...config.agent.scout,
    permission: { bash: "deny" },
  }

  writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n")
  console.log(`Global opencode.json updated at ${configPath}`)
} else {
  console.error("Usage: ergon <init|update>")
  process.exit(1)
}
