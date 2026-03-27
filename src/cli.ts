#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync } from "fs"
import { join, resolve } from "path"

const command = process.argv[2]

if (command === "init") {
  const agentsSource = join(__dirname, "..", "agents")
  const agentsDest = resolve(process.cwd(), ".opencode", "agents")

  if (!existsSync(agentsDest)) {
    mkdirSync(agentsDest, { recursive: true })
  }

  cpSync(agentsSource, agentsDest, { recursive: true })
  console.log(`Ergon agents installed to ${agentsDest}`)
  console.log(
    `Add "ergon-studio" to your opencode.json plugins to activate the plugin.`,
  )
} else {
  console.log("Usage: ergon init")
}
