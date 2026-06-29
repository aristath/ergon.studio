import {
  convertToLlm,
  type BeforeAgentStartEvent,
  type ExtensionAPI,
  type ExtensionCommandContext,
  type ExtensionContext,
  type SessionEntry,
  type ToolCallEvent,
} from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type PlanAction = "start" | "finish" | "cancel";

type PlanMarker = {
  action: PlanAction;
  topic?: string;
  startedAt?: number;
  previousTools?: string[];
  handoffPath?: string;
};

type PlanState = {
  active: boolean;
  topic?: string;
  startedAt?: number;
  previousTools: string[];
  startIndex: number;
};

const PLAN_ENTRY_TYPE = "ergon-plan-state";
const DEFAULT_TOOLS = ["read", "find", "grep", "ls", "bash", "edit", "write"];
const PLAN_TOOLS = [
  "read",
  "find",
  "grep",
  "ls",
  "ask_user_question",
  "subagent",
  "get_subagent_result",
  "edit",
  "write",
];
const HANDOFF_PATH = ".ergon.studio/HANDOFF.md";
const SCRATCHPAD_PATH = ".ergon.studio/scratchpad.md";

const CONTINUE_OPTION = "Continue planning";
const FINISH_OPTION = "Finish and write HANDOFF";
const CANCEL_OPTION = "Cancel planning";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function packageFilePath(...parts: string[]): string {
  const candidates = [
    join(__dirname, "..", ...parts),
    join(__dirname, "..", "..", ...parts),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
}

export function stripAgentFrontmatter(content: string): string {
  return content.replace(/^---[ \t]*\n[\s\S]*?\n---[ \t]*\n?/, "").trim();
}

export const LEGACY_SCOUT_PROMPT = readFileSync(
  packageFilePath("prompts", "scout.md"),
  "utf8",
);

const PI_PLAN_BOUNDARY = `## Pi /plan Mode Boundary

The text above is the authoritative legacy Scout workflow. Preserve its wording and behavior.

Because the user invoked /plan, treat that as the explicit shift into Planning Mode.

Pi-specific mechanics:
- You may inspect the project only through the read-only tools allowed by this extension.
- You may edit or write only .ergon.studio/scratchpad.md when the legacy Scout workflow tells you to record conventions or decisions.
- Do not use implementation tools or write any other files directly during planning.
- The extension writes .ergon.studio/HANDOFF.md only when the user runs /plan finish and reviews the handoff.`;

export const SCOUT_PLAN_PROMPT = `${stripAgentFrontmatter(LEGACY_SCOUT_PROMPT)}\n\n${PI_PLAN_BOUNDARY}`;

export function inactiveState(): PlanState {
  return {
    active: false,
    previousTools: DEFAULT_TOOLS,
    startIndex: -1,
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

function getPlanMarker(entry: SessionEntry): PlanMarker | null {
  if (entry.type !== "custom" || entry.customType !== PLAN_ENTRY_TYPE) {
    return null;
  }

  return isPlainObject(entry.data) ? (entry.data as PlanMarker) : null;
}

export function deriveStateFromBranch(branch: SessionEntry[]): PlanState {
  let state = inactiveState();

  for (let i = 0; i < branch.length; i++) {
    const entry = branch[i];
    const marker = getPlanMarker(entry);
    if (!marker) continue;

    if (marker.action === "start") {
      state = {
        active: true,
        topic:
          typeof marker.topic === "string" && marker.topic.trim()
            ? marker.topic.trim()
            : undefined,
        startedAt:
          typeof marker.startedAt === "number" ? marker.startedAt : Date.now(),
        previousTools: isStringArray(marker.previousTools)
          ? marker.previousTools
          : DEFAULT_TOOLS,
        startIndex: i,
      };
    }

    if (marker.action === "finish" || marker.action === "cancel") {
      state = inactiveState();
    }
  }

  return state;
}

export function getPlanTools(pi: Pick<ExtensionAPI, "getAllTools">): string[] {
  const available = new Set(pi.getAllTools().map((tool) => tool.name));
  return PLAN_TOOLS.filter((tool) => available.has(tool));
}

function notify(
  ctx: ExtensionContext,
  message: string,
  level: "info" | "warning" | "error" = "info",
) {
  if (ctx.hasUI) {
    ctx.ui.notify(message, level);
  } else if (level !== "info") {
    console.error(`[plan] ${level}: ${message}`);
  }
}

function setPlanUi(ctx: ExtensionContext, state: PlanState) {
  if (!ctx.hasUI) {
    return;
  }

  if (!state.active) {
    ctx.ui.setStatus("plan", undefined);
    ctx.ui.setWidget("plan", undefined);
    return;
  }

  const topic = state.topic ? ` • ${state.topic}` : "";
  ctx.ui.setStatus("plan", `plan${topic}`);
  ctx.ui.setWidget("plan", [
    `Plan mode active${topic}`,
    "Scout workflow: optimal → strip down → compare current → plan → attack the plan",
    "read-only investigation; scratchpad writes only",
    "finish: /plan finish   cancel: /plan cancel",
  ]);
}

function applyState(
  pi: Pick<ExtensionAPI, "setActiveTools" | "getAllTools">,
  ctx: ExtensionContext,
  currentState: PlanState,
  nextState: PlanState,
) {
  if (nextState.active) {
    pi.setActiveTools(getPlanTools(pi));
  } else if (currentState.active) {
    pi.setActiveTools(
      currentState.previousTools.length > 0
        ? currentState.previousTools
        : DEFAULT_TOOLS,
    );
  }

  setPlanUi(ctx, nextState);
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((part) => {
      if (
        part &&
        typeof part === "object" &&
        "type" in part &&
        part.type === "text" &&
        "text" in part &&
        typeof part.text === "string"
      ) {
        return part.text;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export function latestAssistantTextSince(
  branch: SessionEntry[],
  startIndex: number,
): string | null {
  const messages = branch
    .slice(Math.max(startIndex + 1, 0))
    .filter(
      (entry): entry is SessionEntry & { type: "message" } =>
        entry.type === "message",
    )
    .map((entry) => entry.message);

  for (const message of convertToLlm(messages).slice().reverse()) {
    if (message.role !== "assistant") {
      continue;
    }

    const text = textFromContent(message.content).trim();
    if (text) {
      return text;
    }
  }

  return null;
}

function defaultHandoff(state: PlanState): string {
  const topic = state.topic ?? "Planning handoff";
  const started = state.startedAt
    ? new Date(state.startedAt).toISOString()
    : new Date().toISOString();

  return [
    `# Handoff: ${topic}`,
    "",
    `Started: ${started}`,
    "",
    "## Goal",
    "",
    "- ",
    "",
    "## Decisions",
    "",
    "- ",
    "",
    "## Plan",
    "",
    "1. ",
    "",
    "## Files And Changes",
    "",
    "- ",
    "",
    "## Verification",
    "",
    "- ",
    "",
    "## Friction Points",
    "",
    "- ",
    "",
    "## Assume You're Wrong",
    "",
    "- ",
    "",
  ].join("\n");
}

function handoffDraft(ctx: ExtensionCommandContext, state: PlanState): string {
  return (
    latestAssistantTextSince(
      ctx.sessionManager.getBranch(),
      state.startIndex,
    ) ?? defaultHandoff(state)
  );
}

function readProjectFile(cwd: string, relativePath: string): string | null {
  const absolutePath = join(cwd, relativePath);
  try {
    return existsSync(absolutePath) ? readFileSync(absolutePath, "utf8") : null;
  } catch {
    return null;
  }
}

async function writeHandoff(
  ctx: ExtensionCommandContext,
  state: PlanState,
): Promise<string | null> {
  if (!ctx.hasUI) {
    notify(
      ctx,
      "Cannot review HANDOFF without a dialog-capable UI.",
      "warning",
    );
    return null;
  }

  const reviewed = await ctx.ui.editor(
    "Review planning handoff",
    handoffDraft(ctx, state),
  );
  if (reviewed === undefined) {
    return null;
  }

  const existingHandoff = readProjectFile(ctx.cwd, HANDOFF_PATH);
  if (existingHandoff !== null) {
    const overwrite = await ctx.ui.confirm(
      "Overwrite existing HANDOFF?",
      `${HANDOFF_PATH} already exists and was included in /plan context. Replace it with the reviewed handoff?`,
    );
    if (!overwrite) {
      return null;
    }
  }

  const absolutePath = join(ctx.cwd, HANDOFF_PATH);
  await mkdir(dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, reviewed, "utf8");
  return HANDOFF_PATH;
}

function isScratchpadPath(cwd: string, rawPath: unknown): boolean {
  if (typeof rawPath !== "string" || !rawPath.trim()) {
    return false;
  }

  return resolve(cwd, rawPath) === resolve(cwd, SCRATCHPAD_PATH);
}

function isAllowedToolCall(
  event: ToolCallEvent,
  allowedTools: string[],
  cwd: string,
): boolean {
  if (!allowedTools.includes(event.toolName)) {
    return false;
  }

  if (event.toolName === "write" || event.toolName === "edit") {
    return isScratchpadPath(cwd, (event.input as Record<string, unknown>).path);
  }

  if (event.toolName !== "subagent") {
    return true;
  }

  const subagentType = (event.input as Record<string, unknown>).subagent_type;
  return (
    typeof subagentType === "string" && subagentType.toLowerCase() === "explore"
  );
}

export default function planExtension(pi: ExtensionAPI): void {
  let state = inactiveState();

  const syncStateFromBranch = (ctx: ExtensionContext) => {
    const nextState = deriveStateFromBranch(ctx.sessionManager.getBranch());
    applyState(pi, ctx, state, nextState);
    state = nextState;
  };

  const startPlan = (topic: string | undefined, ctx: ExtensionContext) => {
    if (state.active) {
      notify(ctx, "Plan mode is already active.", "warning");
      return;
    }

    const nextState: PlanState = {
      active: true,
      topic,
      startedAt: Date.now(),
      previousTools: pi.getActiveTools(),
      startIndex: ctx.sessionManager.getBranch().length,
    };

    pi.appendEntry<PlanMarker>(PLAN_ENTRY_TYPE, {
      action: "start",
      topic,
      startedAt: nextState.startedAt,
      previousTools: nextState.previousTools,
    });

    pi.setActiveTools(getPlanTools(pi));
    state = nextState;
    setPlanUi(ctx, state);
    notify(
      ctx,
      `Plan mode started${topic ? `: ${topic}` : ""}. Work through Scout's phases; use /plan finish when the handoff is ready.`,
    );
  };

  const stopPlan = (
    action: Exclude<PlanAction, "start">,
    ctx: ExtensionContext,
    handoffPath?: string,
  ) => {
    if (!state.active) {
      notify(ctx, "Plan mode is not active.", "warning");
      return;
    }

    const previousTools =
      state.previousTools.length > 0 ? state.previousTools : DEFAULT_TOOLS;

    pi.appendEntry<PlanMarker>(PLAN_ENTRY_TYPE, {
      action,
      topic: state.topic,
      startedAt: state.startedAt,
      previousTools,
      handoffPath,
    });

    pi.setActiveTools(previousTools);
    state = inactiveState();
    setPlanUi(ctx, state);
  };

  const finishPlan = async (ctx: ExtensionCommandContext) => {
    if (!state.active) {
      notify(ctx, "Plan mode is not active.", "warning");
      return;
    }

    const handoffPath = await writeHandoff(ctx, state);
    if (!handoffPath) {
      notify(ctx, "Plan finish cancelled. Plan mode is still active.", "info");
      return;
    }

    stopPlan("finish", ctx, handoffPath);
    notify(ctx, `Plan saved to ${handoffPath}.`, "info");
  };

  const openPlanMenu = async (ctx: ExtensionCommandContext) => {
    const choice = await ctx.ui.select("Plan mode", [
      CONTINUE_OPTION,
      FINISH_OPTION,
      CANCEL_OPTION,
    ]);

    if (!choice || choice === CONTINUE_OPTION) {
      return;
    }

    if (choice === FINISH_OPTION) {
      await finishPlan(ctx);
      return;
    }

    stopPlan("cancel", ctx);
    notify(ctx, "Plan mode cancelled.", "info");
  };

  const handlePlanCommand = async (
    args: string,
    ctx: ExtensionCommandContext,
  ) => {
    const trimmed = args.trim();

    if (!state.active) {
      if (trimmed === "finish" || trimmed === "done" || trimmed === "cancel") {
        notify(ctx, "Plan mode is not active.", "warning");
        return;
      }
      startPlan(trimmed || undefined, ctx);
      return;
    }

    if (trimmed === "cancel") {
      stopPlan("cancel", ctx);
      notify(ctx, "Plan mode cancelled.", "info");
      return;
    }

    if (trimmed === "finish" || trimmed === "done") {
      await finishPlan(ctx);
      return;
    }

    if (!ctx.hasUI) {
      notify(
        ctx,
        "Plan mode is active. Use /plan finish or /plan cancel.",
        "info",
      );
      return;
    }

    await openPlanMenu(ctx);
  };

  pi.registerCommand("plan", {
    description: "Start or finish Scout-style read-only planning mode",
    handler: handlePlanCommand,
  });

  pi.on("before_agent_start", async (event: BeforeAgentStartEvent, ctx) => {
    if (!state.active) {
      return;
    }

    const topicLine = state.topic
      ? `Current planning topic: ${state.topic}`
      : "";
    const existingHandoff = readProjectFile(ctx.cwd, HANDOFF_PATH);
    const handoffContext = existingHandoff
      ? `Existing ${HANDOFF_PATH}:\n\n${existingHandoff}`
      : "";

    return {
      systemPrompt: [
        event.systemPrompt ?? "",
        SCOUT_PLAN_PROMPT,
        topicLine,
        handoffContext,
      ]
        .filter(Boolean)
        .join("\n\n"),
    };
  });

  pi.on("tool_call", async (event, ctx) => {
    if (!state.active) {
      return;
    }

    const allowedTools = getPlanTools(pi);
    if (isAllowedToolCall(event, allowedTools, ctx.cwd)) {
      return;
    }

    const allowed = allowedTools.map((tool) => `\`${tool}\``).join(", ");
    if (event.toolName === "subagent") {
      return {
        block: true,
        reason:
          '/plan only allows subagent with subagent_type: "Explore". Finish or cancel /plan to use implementation agents.',
      };
    }

    if (event.toolName === "write" || event.toolName === "edit") {
      return {
        block: true,
        reason: `/plan only allows ${event.toolName} for ${SCRATCHPAD_PATH}. Finish or cancel /plan to edit other files.`,
      };
    }

    return {
      block: true,
      reason: `/plan mode is read-only and only permits ${allowed}. Finish or cancel /plan to use implementation tools.`,
    };
  });

  pi.on("session_start", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_tree", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    setPlanUi(ctx, inactiveState());
  });
}
