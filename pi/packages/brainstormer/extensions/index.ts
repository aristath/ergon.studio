import {
  type BeforeAgentStartEvent,
  type ExtensionAPI,
  type ExtensionCommandContext,
  type ExtensionContext,
  type SessionEntry,
  type ToolCallEvent,
} from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

type BrainstormAction = "start" | "done" | "cancel";

type BrainstormMarker = {
  action: BrainstormAction;
  startedAt?: number;
  previousTools?: string[];
};

type PlanAction = "start" | "finish" | "cancel";

type PlanMarker = {
  action: PlanAction;
};

type BrainstormState = {
  active: boolean;
  startedAt?: number;
  previousTools: string[];
  supersededByPlan: boolean;
};

const BRAINSTORM_ENTRY_TYPE = "ergon-brainstorm-state";
const PLAN_ENTRY_TYPE = "ergon-plan-state";
const DEFAULT_TOOLS = ["read", "find", "grep", "ls", "bash", "edit", "write"];
const BRAINSTORM_TOOLS = [
  "read",
  "find",
  "grep",
  "ls",
  "ask_user_question",
  "subagent",
  "get_subagent_result",
];

const CONTINUE_OPTION = "Continue brainstorming";
const DONE_OPTION = "Done brainstorming";
const CANCEL_OPTION = "Cancel brainstorming";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function packageFilePath(...parts: string[]): string {
  const candidates = [
    join(__dirname, "..", ...parts),
    join(__dirname, "..", "..", ...parts),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
}

export const BRAINSTORM_PROMPT = readFileSync(
  packageFilePath("prompts", "brainstorm.md"),
  "utf8",
).trim();

const PI_BRAINSTORM_BOUNDARY = `## Pi /brainstorm Mode Boundary

Because the user invoked /brainstorm, stay in freeform brainstorm mode.

Pi-specific mechanics:
- There is no required topic. Treat the session as open-ended exploration.
- Do not write implementation code or artifacts.
- Use only read-only exploration tools allowed by this extension.
- If the idea converges into something concrete enough to plan, suggest that the user run /plan. Do not switch modes yourself.`;

export const BRAINSTORM_SYSTEM_PROMPT = `${BRAINSTORM_PROMPT}\n\n${PI_BRAINSTORM_BOUNDARY}`;

export function inactiveState(): BrainstormState {
  return {
    active: false,
    previousTools: DEFAULT_TOOLS,
    supersededByPlan: false,
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

function getBrainstormMarker(entry: SessionEntry): BrainstormMarker | null {
  if (entry.type !== "custom" || entry.customType !== BRAINSTORM_ENTRY_TYPE) {
    return null;
  }

  return isPlainObject(entry.data) ? (entry.data as BrainstormMarker) : null;
}

function getPlanMarker(entry: SessionEntry): PlanMarker | null {
  if (entry.type !== "custom" || entry.customType !== PLAN_ENTRY_TYPE) {
    return null;
  }

  return isPlainObject(entry.data) ? (entry.data as PlanMarker) : null;
}

export function deriveStateFromBranch(branch: SessionEntry[]): BrainstormState {
  let state = inactiveState();
  let planActive = false;

  for (const entry of branch) {
    const planMarker = getPlanMarker(entry);
    if (planMarker?.action === "start") {
      planActive = true;
      state = {
        ...inactiveState(),
        supersededByPlan: true,
      };
      continue;
    }

    if (planMarker?.action === "finish" || planMarker?.action === "cancel") {
      planActive = false;
      if (state.supersededByPlan) {
        state = inactiveState();
      }
      continue;
    }

    const marker = getBrainstormMarker(entry);
    if (!marker) continue;

    if (marker.action === "start") {
      if (planActive) {
        state = {
          ...inactiveState(),
          supersededByPlan: true,
        };
        continue;
      }

      state = {
        active: true,
        startedAt:
          typeof marker.startedAt === "number" ? marker.startedAt : Date.now(),
        previousTools: isStringArray(marker.previousTools)
          ? marker.previousTools
          : DEFAULT_TOOLS,
        supersededByPlan: false,
      };
    }

    if (marker.action === "done" || marker.action === "cancel") {
      state = inactiveState();
    }
  }

  return state;
}

export function getBrainstormTools(
  pi: Pick<ExtensionAPI, "getAllTools">,
): string[] {
  const available = new Set(pi.getAllTools().map((tool) => tool.name));
  return BRAINSTORM_TOOLS.filter((tool) => available.has(tool));
}

function notify(
  ctx: ExtensionContext,
  message: string,
  level: "info" | "warning" | "error" = "info",
) {
  if (ctx.hasUI) {
    ctx.ui.notify(message, level);
  } else if (level !== "info") {
    console.error(`[brainstorm] ${level}: ${message}`);
  }
}

function setBrainstormUi(ctx: ExtensionContext, state: BrainstormState) {
  if (!ctx.hasUI) {
    return;
  }

  if (!state.active) {
    ctx.ui.setStatus("brainstorm", undefined);
    ctx.ui.setWidget("brainstorm", undefined);
    return;
  }

  ctx.ui.setStatus("brainstorm", "brainstorm");
  ctx.ui.setWidget("brainstorm", [
    "Brainstorm mode active",
    "freeform exploration; no implementation tools",
    "when ready, run /plan",
  ]);
}

function applyState(
  pi: Pick<ExtensionAPI, "setActiveTools" | "getAllTools">,
  ctx: ExtensionContext,
  currentState: BrainstormState,
  nextState: BrainstormState,
) {
  if (nextState.active && !currentState.active) {
    pi.setActiveTools(getBrainstormTools(pi));
  } else if (
    currentState.active &&
    !nextState.active &&
    !nextState.supersededByPlan
  ) {
    pi.setActiveTools(
      currentState.previousTools.length > 0
        ? currentState.previousTools
        : DEFAULT_TOOLS,
    );
  }

  setBrainstormUi(ctx, nextState);
}

function isAllowedToolCall(
  event: ToolCallEvent,
  allowedTools: string[],
): boolean {
  if (!allowedTools.includes(event.toolName)) {
    return false;
  }

  if (event.toolName !== "subagent") {
    return true;
  }

  const subagentType = (event.input as Record<string, unknown>).subagent_type;
  return (
    typeof subagentType === "string" && subagentType.toLowerCase() === "explore"
  );
}

export default function brainstormExtension(pi: ExtensionAPI): void {
  let state = inactiveState();

  const syncStateFromBranch = (ctx: ExtensionContext) => {
    const nextState = deriveStateFromBranch(ctx.sessionManager.getBranch());
    applyState(pi, ctx, state, nextState);
    state = nextState;
  };

  const planIsActive = (ctx: ExtensionContext) =>
    deriveStateFromBranch(ctx.sessionManager.getBranch()).supersededByPlan;

  const startBrainstorm = (ctx: ExtensionContext) => {
    syncStateFromBranch(ctx);

    if (planIsActive(ctx)) {
      notify(
        ctx,
        "Plan mode is active. Finish or cancel /plan before starting /brainstorm.",
        "warning",
      );
      return;
    }

    if (state.active) {
      notify(ctx, "Brainstorm mode is already active.", "warning");
      return;
    }

    const nextState: BrainstormState = {
      active: true,
      startedAt: Date.now(),
      previousTools: pi.getActiveTools(),
      supersededByPlan: false,
    };

    pi.appendEntry<BrainstormMarker>(BRAINSTORM_ENTRY_TYPE, {
      action: "start",
      startedAt: nextState.startedAt,
      previousTools: nextState.previousTools,
    });

    pi.setActiveTools(getBrainstormTools(pi));
    state = nextState;
    setBrainstormUi(ctx, state);
    notify(ctx, "Brainstorm mode started. Explore freely; run /plan when the idea is ready.");
  };

  const stopBrainstorm = (
    action: Exclude<BrainstormAction, "start">,
    ctx: ExtensionContext,
  ) => {
    if (!state.active) {
      notify(ctx, "Brainstorm mode is not active.", "warning");
      return;
    }

    const previousTools =
      state.previousTools.length > 0 ? state.previousTools : DEFAULT_TOOLS;

    pi.appendEntry<BrainstormMarker>(BRAINSTORM_ENTRY_TYPE, {
      action,
      startedAt: state.startedAt,
      previousTools,
    });

    pi.setActiveTools(previousTools);
    state = inactiveState();
    setBrainstormUi(ctx, state);
  };

  const openBrainstormMenu = async (ctx: ExtensionCommandContext) => {
    if (!ctx.hasUI) {
      stopBrainstorm("done", ctx);
      notify(
        ctx,
        "Brainstorming done. When you want an implementation plan, run /plan.",
        "info",
      );
      return;
    }

    const choice = await ctx.ui.select("Brainstorm mode", [
      CONTINUE_OPTION,
      DONE_OPTION,
      CANCEL_OPTION,
    ]);

    if (!choice || choice === CONTINUE_OPTION) {
      return;
    }

    if (choice === DONE_OPTION) {
      stopBrainstorm("done", ctx);
      notify(ctx, "Brainstorming done. When you want an implementation plan, run /plan.", "info");
      return;
    }

    stopBrainstorm("cancel", ctx);
    notify(ctx, "Brainstorm mode cancelled.", "info");
  };

  pi.registerCommand("brainstorm", {
    description: "Start or manage freeform brainstorm mode",
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      syncStateFromBranch(ctx);

      if (!state.active) {
        startBrainstorm(ctx);
        return;
      }

      await openBrainstormMenu(ctx);
    },
  });

  pi.on("before_agent_start", async (event: BeforeAgentStartEvent, ctx) => {
    syncStateFromBranch(ctx);

    if (!state.active) {
      return;
    }

    return {
      systemPrompt: [event.systemPrompt ?? "", BRAINSTORM_SYSTEM_PROMPT]
        .filter(Boolean)
        .join("\n\n"),
    };
  });

  pi.on("tool_call", async (event, ctx) => {
    syncStateFromBranch(ctx);

    if (!state.active) {
      return;
    }

    const allowedTools = getBrainstormTools(pi);
    if (isAllowedToolCall(event, allowedTools)) {
      return;
    }

    if (event.toolName === "subagent") {
      return {
        block: true,
        reason:
          '/brainstorm only allows subagent with subagent_type: "Explore". Continue brainstorming or run /brainstorm to exit before using implementation agents.',
      };
    }

    const allowed = allowedTools.map((tool) => `\`${tool}\``).join(", ");
    return {
      block: true,
      reason: `/brainstorm mode is for read-only exploration and only permits ${allowed}. Run /brainstorm to exit before using implementation tools.`,
    };
  });

  pi.on("session_start", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_tree", async (_event, ctx) => {
    syncStateFromBranch(ctx);
  });

  pi.on("session_shutdown", async (_event, ctx) => {
    setBrainstormUi(ctx, inactiveState());
  });
}
