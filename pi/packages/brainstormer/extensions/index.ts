import {
  type BeforeAgentStartEvent,
  type ExtensionAPI,
  type ExtensionCommandContext,
  type ExtensionContext,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

type BrainstormAction = "start" | "done" | "cancel";

type BrainstormMarker = {
  action: BrainstormAction;
  startedAt?: number;
};

type PlanAction = "start" | "finish" | "cancel";

type PlanMarker = {
  action: PlanAction;
};

type BrainstormState = {
  active: boolean;
  startedAt?: number;
  supersededByPlan: boolean;
};

const BRAINSTORM_ENTRY_TYPE = "ergon-brainstorm-state";
const PLAN_ENTRY_TYPE = "ergon-plan-state";

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
- Pi's active tool selection remains unchanged. Use available tools only for read-only exploration, and do not delegate implementation work.
- If the idea converges into something concrete enough to plan, suggest that the user run /plan. Do not switch modes yourself.`;

export const BRAINSTORM_SYSTEM_PROMPT = `${BRAINSTORM_PROMPT}\n\n${PI_BRAINSTORM_BOUNDARY}`;

export function inactiveState(): BrainstormState {
  return {
    active: false,
    supersededByPlan: false,
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
        supersededByPlan: false,
      };
    }

    if (marker.action === "done" || marker.action === "cancel") {
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
    "freeform exploration; no implementation work",
    "when ready, run /plan",
  ]);
}

export default function brainstormExtension(pi: ExtensionAPI): void {
  let state = inactiveState();

  const syncStateFromBranch = (ctx: ExtensionContext) => {
    const nextState = deriveStateFromBranch(ctx.sessionManager.getBranch());
    state = nextState;
    setBrainstormUi(ctx, state);
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
      supersededByPlan: false,
    };

    pi.appendEntry<BrainstormMarker>(BRAINSTORM_ENTRY_TYPE, {
      action: "start",
      startedAt: nextState.startedAt,
    });

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

    pi.appendEntry<BrainstormMarker>(BRAINSTORM_ENTRY_TYPE, {
      action,
      startedAt: state.startedAt,
    });

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
