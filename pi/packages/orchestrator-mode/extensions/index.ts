import {
  type BeforeAgentStartEvent,
  type ExtensionAPI,
  type ExtensionCommandContext,
  type ExtensionContext,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import type { ChildProcess, SpawnOptions } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "typebox";

type OrchestratorAction = "start" | "finish" | "cancel";

type OrchestratorMarker = {
  action: OrchestratorAction;
  startedAt?: number;
};

type OrchestratorState = {
  active: boolean;
  startedAt?: number;
  supersededByMode: boolean;
};

type AgentDefinition = {
  name: string;
  description: string;
  prompt: string;
  tools: string[];
  model?: string;
};

type ChildResult = {
  agent: string;
  brief: string;
  exitCode: number;
  output: string;
  stderr: string;
  error?: string;
};

const ORCHESTRATOR_ENTRY_TYPE = "ergon-orchestrator-state";
const PLAN_ENTRY_TYPE = "ergon-plan-state";
const BRAINSTORM_ENTRY_TYPE = "ergon-brainstorm-state";

const DEFAULT_TOOLS = ["read", "find", "grep", "ls", "bash", "edit", "write"];

const AGENT_TOOL_OVERRIDES: Record<string, string[]> = {
  architect: ["read", "find", "grep", "ls"],
  coder: ["read", "find", "grep", "ls", "bash", "edit", "write"],
  critic: ["read", "find", "grep", "ls", "bash"],
  design_reviewer: ["read", "find", "grep", "ls"],
  quality_controller: ["read", "find", "grep", "ls", "task"],
  researcher: ["read", "find", "grep", "ls", "bash"],
  reviewer: ["read", "find", "grep", "ls", "bash"],
  tester: ["read", "find", "grep", "ls", "bash"],
};

const CONTINUE_OPTION = "Continue orchestrating";
const FINISH_OPTION = "Finish orchestrating";
const CANCEL_OPTION = "Cancel orchestrating";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

type SpawnFunction = (
  command: string,
  args: string[],
  options: SpawnOptions,
) => ChildProcess;

let spawnProcess: SpawnFunction = spawn;
let childKillGraceMs = 5000;

export function setSpawnProcessForTest(spawnOverride?: SpawnFunction): void {
  spawnProcess = spawnOverride ?? spawn;
}

export function setChildKillGraceMsForTest(timeoutMs = 5000): void {
  childKillGraceMs = timeoutMs;
}

function packageFilePath(...parts: string[]): string {
  const candidates = [
    join(__dirname, "..", ...parts),
    join(__dirname, "..", "..", ...parts),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
}

function stripFrontmatter(content: string): string {
  return content.replace(/^---[ \t]*\n[\s\S]*?\n---[ \t]*\n?/, "").trim();
}

function parseFrontmatter(content: string): Record<string, string> {
  const match = content.match(/^---[ \t]*\n([\s\S]*?)\n---[ \t]*\n?/);
  if (!match) return {};

  const frontmatter: Record<string, string> = {};
  for (const line of match[1].split("\n")) {
    const separator = line.indexOf(":");
    if (separator <= 0) continue;
    frontmatter[line.slice(0, separator).trim()] = line
      .slice(separator + 1)
      .trim()
      .replace(/^["']|["']$/g, "");
  }
  return frontmatter;
}

function agentFilePath(agentName: string): string {
  return packageFilePath("agents", `${agentName}.md`);
}

function listAgentNames(): string[] {
  return Object.keys(AGENT_TOOL_OVERRIDES).sort();
}

export function loadAgentDefinition(agentName: string): AgentDefinition | null {
  const filePath = agentFilePath(agentName);
  if (!existsSync(filePath)) return null;

  const content = readFileSync(filePath, "utf8");
  const frontmatter = parseFrontmatter(content);
  const prompt = stripFrontmatter(content);
  return {
    name: agentName,
    description: frontmatter.description ?? agentName,
    prompt,
    tools: AGENT_TOOL_OVERRIDES[agentName] ?? DEFAULT_TOOLS,
    model: frontmatter.model,
  };
}

const ORCHESTRATOR_LEGACY_PROMPT =
  loadAgentDefinition("orchestrator")?.prompt ?? "";

const PI_ORCHESTRATOR_BOUNDARY = `## Pi /orchestrator Mode Boundary

The text above is the authoritative legacy orchestrator workflow. Preserve its wording and behavior.

Pi-specific mechanics:
- The legacy task tool is registered as the \`task\` tool in this package.
- The legacy run_parallel tool is registered as the \`run_parallel\` tool in this package.
- Pi's active tool selection remains unchanged when this mode starts or ends.
- These tools run the bundled legacy agent prompts from this package.
- The quality gate remains agent-owned: invoke \`quality_controller\` after code work, and follow its APPROVED/REJECTED loop exactly.
- Do not replace the quality controller's judgment with extension state or deterministic parsing.`;

export const ORCHESTRATOR_SYSTEM_PROMPT = `${ORCHESTRATOR_LEGACY_PROMPT}\n\n${PI_ORCHESTRATOR_BOUNDARY}`;

export function inactiveState(): OrchestratorState {
  return {
    active: false,
    supersededByMode: false,
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function customAction(entry: SessionEntry, customType: string): string | null {
  if (entry.type !== "custom" || entry.customType !== customType) {
    return null;
  }
  return isPlainObject(entry.data) && typeof entry.data.action === "string"
    ? entry.data.action
    : null;
}

function getOrchestratorMarker(entry: SessionEntry): OrchestratorMarker | null {
  if (entry.type !== "custom" || entry.customType !== ORCHESTRATOR_ENTRY_TYPE) {
    return null;
  }

  return isPlainObject(entry.data) ? (entry.data as OrchestratorMarker) : null;
}

export function deriveStateFromBranch(
  branch: SessionEntry[],
): OrchestratorState {
  let state = inactiveState();
  let blockerActive = false;

  for (const entry of branch) {
    const planAction = customAction(entry, PLAN_ENTRY_TYPE);
    const brainstormAction = customAction(entry, BRAINSTORM_ENTRY_TYPE);

    if (planAction === "start" || brainstormAction === "start") {
      blockerActive = true;
      state = {
        ...inactiveState(),
        supersededByMode: true,
      };
      continue;
    }

    if (
      planAction === "finish" ||
      planAction === "cancel" ||
      brainstormAction === "done" ||
      brainstormAction === "cancel"
    ) {
      blockerActive = false;
      if (state.supersededByMode) {
        state = inactiveState();
      }
      continue;
    }

    const marker = getOrchestratorMarker(entry);
    if (!marker) continue;

    if (marker.action === "start") {
      if (blockerActive) {
        state = {
          ...inactiveState(),
          supersededByMode: true,
        };
        continue;
      }

      state = {
        active: true,
        startedAt:
          typeof marker.startedAt === "number" ? marker.startedAt : Date.now(),
        supersededByMode: false,
      };
    }

    if (marker.action === "finish" || marker.action === "cancel") {
      state = inactiveState();
    }
  }

  return state;
}

function notify(
  ctx: ExtensionContext,
  message: string,
  level: "info" | "warning" | "error" = "info",
) {
  if (ctx.hasUI) {
    ctx.ui.notify(message, level);
  } else if (level !== "info") {
    console.error(`[orchestrator] ${level}: ${message}`);
  }
}

function setOrchestratorUi(ctx: ExtensionContext, state: OrchestratorState) {
  if (!ctx.hasUI) {
    return;
  }

  if (!state.active) {
    ctx.ui.setStatus("orchestrator", undefined);
    ctx.ui.setWidget("orchestrator", undefined);
    return;
  }

  ctx.ui.setStatus("orchestrator", "orchestrator");
  ctx.ui.setWidget("orchestrator", [
    "Orchestrator mode active",
    "delegation workflow; quality gate is agent-owned",
    "after code work: invoke quality_controller",
  ]);
}

function writeTempPrompt(agentName: string, prompt: string): string {
  const dir = mkdtempSync(join(tmpdir(), "ergon-pi-agent-"));
  const safeName = agentName.replace(/[^\w.-]+/g, "_");
  const filePath = join(dir, `${safeName}.md`);
  writeFileSync(filePath, prompt, { encoding: "utf8", mode: 0o600 });
  return filePath;
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  const isBunVirtualScript = currentScript?.startsWith("/$bunfs/root/");
  if (currentScript && !isBunVirtualScript && existsSync(currentScript)) {
    return { command: process.execPath, args: [currentScript, ...args] };
  }

  const execName = basename(process.execPath).toLowerCase();
  const isGenericRuntime = /^(node|bun)(\.exe)?$/.test(execName);
  if (!isGenericRuntime) {
    return { command: process.execPath, args };
  }

  return { command: "pi", args };
}

function extractEventText(event: unknown): string {
  if (!isPlainObject(event) || !isPlainObject(event.message)) {
    return "";
  }

  const content = event.message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((part) =>
      isPlainObject(part) &&
      part.type === "text" &&
      typeof part.text === "string"
        ? part.text
        : "",
    )
    .filter(Boolean)
    .join("\n");
}

async function runBundledAgent(
  agentName: string,
  brief: string,
  cwd: string,
  signal?: AbortSignal,
): Promise<ChildResult> {
  const definition = loadAgentDefinition(agentName);
  if (!definition) {
    return {
      agent: agentName,
      brief,
      exitCode: 1,
      output: "",
      stderr: "",
      error: `Unknown agent: "${agentName}". Available agents: ${listAgentNames()
        .map((name) => `"${name}"`)
        .join(", ")}.`,
    };
  }

  const promptPath = writeTempPrompt(
    definition.name,
    `${definition.prompt}\n\n${PI_ORCHESTRATOR_BOUNDARY}`,
  );
  const args = ["--mode", "json", "-p", "--no-session"];
  if (definition.model) args.push("--model", definition.model);
  if (definition.tools.length > 0)
    args.push("--tools", definition.tools.join(","));
  args.push("--append-system-prompt", promptPath);
  args.push(`Task: ${brief}`);

  const invocation = getPiInvocation(args);
  let output = "";
  let stderr = "";
  let buffer = "";
  let wasAborted = false;

  try {
    const exitCode = await new Promise<number>((resolve) => {
      const child = spawnProcess(invocation.command, invocation.args, {
        cwd,
        shell: false,
        stdio: ["ignore", "pipe", "pipe"],
      });
      let closed = false;

      const processLine = (line: string) => {
        if (!line.trim()) return;
        try {
          const event = JSON.parse(line);
          const text = extractEventText(event);
          if (text) output = text;
        } catch {
          /* ignore non-JSON output */
        }
      };

      child.stdout?.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) processLine(line);
      });

      child.stderr?.on("data", (chunk) => {
        stderr += chunk.toString();
      });

      child.on("close", (code) => {
        closed = true;
        if (buffer.trim()) processLine(buffer);
        resolve(code ?? 0);
      });

      child.on("error", () => resolve(1));

      const abort = () => {
        wasAborted = true;
        child.kill("SIGTERM");
        setTimeout(() => {
          if (!closed) child.kill("SIGKILL");
        }, childKillGraceMs);
      };

      if (signal?.aborted) abort();
      else signal?.addEventListener("abort", abort, { once: true });
    });

    return {
      agent: agentName,
      brief,
      exitCode,
      output,
      stderr,
      error: wasAborted ? "Agent task aborted." : undefined,
    };
  } finally {
    rmSync(dirname(promptPath), { recursive: true, force: true });
  }
}

function renderTaskResult(result: ChildResult): string {
  if (result.error) {
    return `## ${result.agent}\n\nTask failed: ${result.error}`;
  }

  if (result.exitCode !== 0) {
    const detail = result.output || result.stderr || "No output.";
    return `## ${result.agent}\n\nTask failed with exit code ${result.exitCode}.\n\n${detail}`;
  }

  return `## ${result.agent}\n\n${result.output || "(no output)"}`;
}

const TaskParams = Type.Object({
  agent: Type.String({ description: "Bundled legacy agent to invoke" }),
  brief: Type.String({ description: "Full brief to send to the agent" }),
});

const RunParallelTask = Type.Object({
  agent: Type.String({ description: "Bundled legacy agent to invoke" }),
  brief: Type.String({ description: "Full brief to send to the agent" }),
});

const RunParallelParams = Type.Object({
  tasks: Type.Array(RunParallelTask, {
    minItems: 1,
    description: "List of agent+brief pairs to run in parallel",
  }),
});

export default function orchestratorExtension(pi: ExtensionAPI): void {
  let state = inactiveState();

  pi.registerTool({
    name: "task",
    label: "Task",
    description:
      "Delegate one task to a bundled Ergon legacy specialist agent. Use this for quality_controller, reviewer, design_reviewer, coder, architect, critic, researcher, or tester.",
    parameters: TaskParams,
    execute: async (_toolCallId, params, signal, _onUpdate, ctx) => {
      const result = await runBundledAgent(
        params.agent,
        params.brief,
        ctx.cwd,
        signal,
      );
      return {
        content: [{ type: "text", text: renderTaskResult(result) }],
        details: result,
        isError: Boolean(result.error) || result.exitCode !== 0,
      };
    },
  });

  pi.registerTool({
    name: "run_parallel",
    label: "Run Parallel",
    description:
      "Run multiple bundled Ergon legacy specialist agents concurrently. Do not use write-capable agents in parallel.",
    parameters: RunParallelParams,
    execute: async (_toolCallId, params, signal, _onUpdate, ctx) => {
      const results = await Promise.all(
        params.tasks.map((task) =>
          runBundledAgent(task.agent, task.brief, ctx.cwd, signal),
        ),
      );
      return {
        content: [
          {
            type: "text",
            text: results
              .map((result) => renderTaskResult(result))
              .join("\n\n---\n\n"),
          },
        ],
        details: { results },
        isError: results.some(
          (result) => result.error || result.exitCode !== 0,
        ),
      };
    },
  });

  const syncStateFromBranch = (ctx: ExtensionContext) => {
    const nextState = deriveStateFromBranch(ctx.sessionManager.getBranch());
    state = nextState;
    setOrchestratorUi(ctx, state);
  };

  const otherModeIsActive = (ctx: ExtensionContext) =>
    deriveStateFromBranch(ctx.sessionManager.getBranch()).supersededByMode;

  const startOrchestrator = (ctx: ExtensionContext) => {
    syncStateFromBranch(ctx);

    if (otherModeIsActive(ctx)) {
      notify(
        ctx,
        "Another Ergon mode is active. Finish or cancel it before starting /orchestrator.",
        "warning",
      );
      return;
    }

    if (state.active) {
      notify(ctx, "Orchestrator mode is already active.", "warning");
      return;
    }

    const nextState: OrchestratorState = {
      active: true,
      startedAt: Date.now(),
      supersededByMode: false,
    };

    pi.appendEntry<OrchestratorMarker>(ORCHESTRATOR_ENTRY_TYPE, {
      action: "start",
      startedAt: nextState.startedAt,
    });

    state = nextState;
    setOrchestratorUi(ctx, state);
    notify(ctx, "Orchestrator mode started.");
  };

  const stopOrchestrator = (
    action: Exclude<OrchestratorAction, "start">,
    ctx: ExtensionContext,
  ) => {
    if (!state.active) {
      notify(ctx, "Orchestrator mode is not active.", "warning");
      return;
    }

    pi.appendEntry<OrchestratorMarker>(ORCHESTRATOR_ENTRY_TYPE, {
      action,
      startedAt: state.startedAt,
    });

    state = inactiveState();
    setOrchestratorUi(ctx, state);
  };

  const openOrchestratorMenu = async (ctx: ExtensionCommandContext) => {
    if (!ctx.hasUI) {
      stopOrchestrator("finish", ctx);
      notify(ctx, "Orchestrator mode finished.", "info");
      return;
    }

    const choice = await ctx.ui.select("Orchestrator mode", [
      CONTINUE_OPTION,
      FINISH_OPTION,
      CANCEL_OPTION,
    ]);

    if (!choice || choice === CONTINUE_OPTION) {
      return;
    }

    if (choice === FINISH_OPTION) {
      stopOrchestrator("finish", ctx);
      notify(ctx, "Orchestrator mode finished.", "info");
      return;
    }

    stopOrchestrator("cancel", ctx);
    notify(ctx, "Orchestrator mode cancelled.", "info");
  };

  pi.registerCommand("orchestrator", {
    description: "Start or manage Ergon orchestrator mode",
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      syncStateFromBranch(ctx);

      if (!state.active) {
        startOrchestrator(ctx);
        return;
      }

      await openOrchestratorMenu(ctx);
    },
  });

  pi.on("before_agent_start", async (event: BeforeAgentStartEvent, ctx) => {
    syncStateFromBranch(ctx);

    if (!state.active) {
      return;
    }

    return {
      systemPrompt: [event.systemPrompt ?? "", ORCHESTRATOR_SYSTEM_PROMPT]
        .filter(Boolean)
        .join("\n\n"),
    };
  });

  pi.on("session_start", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_tree", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    setOrchestratorUi(ctx, inactiveState());
  });
}
