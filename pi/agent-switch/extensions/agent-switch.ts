import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const AGENT_STATE_ENTRY = "agent-switch";
const UI_WIDGET_KEY = "agent-switch";
const UI_STATUS_KEY = "agent-switch";

type SessionCustomEntry = {
  type?: string;
  customType?: string;
  data?: { name?: string | null };
};

type AgentInfo = {
  name: string;
  path: string;
};

type AgentSelectorResult = {
  name: string | null;
  cancelled: boolean;
};

function normalizeAgentName(value: string): string {
  return value.trim().toLowerCase();
}

function discoverAgentFiles(cwd: string): AgentInfo[] {
  const candidates = [
    path.join(cwd, "agents"),
    path.join(cwd, ".pi", "agents"),
    path.join(process.env.HOME ?? "/", ".pi", "agent", "agents"),
  ].filter((candidate): candidate is string => {
    return existsSync(candidate);
  });

  const files = new Map<string, AgentInfo>();

  for (const dir of candidates) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (!entry.isFile() || !entry.name.endsWith(".md")) {
          continue;
        }
        const name = entry.name.replace(/\.md$/i, "");
        const normalized = normalizeAgentName(name);
        const resolvedPath = path.join(dir, entry.name);

        files.set(normalized, {
          name,
          path: resolvedPath,
        });
      }
    } catch {
      // Ignore unreadable directories; discovery is best-effort.
    }
  }

  return [...files.values()];
}

function stripFrontmatter(markdown: string): string {
  const trimmed = markdown.trim();
  if (!trimmed.startsWith("---")) {
    return trimmed;
  }

  const endIndex = trimmed.indexOf("---", 3);
  if (endIndex === -1) {
    return trimmed;
  }

  return trimmed.slice(endIndex + 3).trim();
}

function loadAgentPrompt(filePath: string): string | null {
  try {
    const raw = readFileSync(filePath, "utf8");
    return stripFrontmatter(raw);
  } catch {
    return null;
  }
}

function findPersistedAgent(entries: unknown[]): string | null {
  if (!Array.isArray(entries)) {
    return null;
  }

  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const entry = entries[i] as SessionCustomEntry;
    if (!entry || entry.type !== "custom" || entry.customType !== AGENT_STATE_ENTRY) {
      continue;
    }

    const name = entry.data?.name;
    if (typeof name === "string" && name.trim()) {
      return name.trim();
    }
  }

  return null;
}

function buildCompletions(prefix: string, values: string[]): Array<{ value: string; label: string }> {
  const term = normalizeAgentName(prefix);
  return values
    .filter((value) => normalizeAgentName(value).startsWith(term))
    .map((value) => ({ value, label: value }));
}

function getAvailableAgentPrompt(agents: AgentInfo[], selected: string | null): AgentInfo | null {
  if (!selected) {
    return null;
  }

  const match = normalizeAgentName(selected);
  return agents.find((agent) => normalizeAgentName(agent.name) === match) ?? null;
}

function setWidgetAndStatus(ctx: any, activeAgent: string | null): void {
  const hasWidget = !!ctx.ui?.setWidget;
  const hasStatus = !!ctx.ui?.setStatus;

  if (!hasWidget && !hasStatus) {
    return;
  }

  const label = activeAgent ? `Active Agent: ${activeAgent}` : undefined;

  if (hasWidget) {
    if (label) {
      ctx.ui.setWidget(UI_WIDGET_KEY, [label], { placement: "belowEditor" });
    } else {
      ctx.ui.setWidget(UI_WIDGET_KEY, undefined);
    }
  }

  if (hasStatus) {
    if (label) {
      ctx.ui.setStatus(UI_STATUS_KEY, `agent: ${activeAgent}`);
    } else {
      ctx.ui.setStatus(UI_STATUS_KEY, undefined);
    }
  }
}

function notifyIfPossible(ctx: any, message: string, type?: "info" | "warning" | "error") {
  if (ctx.ui?.notify) {
    ctx.ui.notify(message, type);
  }
}

async function chooseAgentViaPrompt(ctx: any, agents: AgentInfo[], activeAgent: string | null): Promise<AgentSelectorResult> {
  if (!ctx.ui?.select) {
    return {
      name: null,
      cancelled: true,
    };
  }

  if (!agents.length) {
    notifyIfPossible(ctx, "No agent files found. Expected agents in ./agents, ~/.pi/agent/agents, or ./.pi/agents.");
    return {
      name: null,
      cancelled: true,
    };
  }

  const names = agents.map((agent) => agent.name);
  const activeNormalized = activeAgent ? normalizeAgentName(activeAgent) : "";
  const marker = " [active]";
  const options = names.map((agentName) => {
    const isActive = normalizeAgentName(agentName) === activeNormalized;
    return isActive ? `${agentName}${marker}` : agentName;
  });

  const selection = await ctx.ui.select("Select agent:", options);

  if (!selection) {
    return {
      name: null,
      cancelled: true,
    };
  }

  const cleanedSelection = selection.endsWith(marker)
    ? selection.slice(0, -marker.length).trim()
    : selection;

  const selectedIndex = names.indexOf(cleanedSelection);
  if (selectedIndex === -1) {
    return {
      name: null,
      cancelled: true,
    };
  }

  return {
    name: agents[selectedIndex].name,
    cancelled: false,
  };
}

export default function (pi: ExtensionAPI) {
  let activeAgent: string | null = null;
  let knownAgents: AgentInfo[] = [];

  const syncAgents = (cwd: string) => {
    knownAgents = discoverAgentFiles(cwd);
    knownAgents.sort((a, b) => a.name.localeCompare(b.name));
    return knownAgents;
  };

  const saveAgentChoice = (agentName: string | null) => {
    activeAgent = agentName;
    pi.appendEntry(AGENT_STATE_ENTRY, {
      name: agentName,
    });
  };

  const renderCurrentAgent = (ctx: any) => {
    setWidgetAndStatus(ctx, activeAgent);
  };

  pi.on("session_start", async (_event, ctx) => {
    syncAgents(ctx.cwd || process.cwd());

    const persisted = findPersistedAgent(ctx.sessionManager.getEntries());
    if (persisted) {
      const nextAgent = persisted.trim();
      if (persisted !== activeAgent) {
        activeAgent = nextAgent;
      }
    }

    const available = getAvailableAgentPrompt(knownAgents, activeAgent);
    if (activeAgent && !available) {
      // Keep the selected name if it is intentionally persisted but no prompt file
      // exists in the current working tree. We still show it in status so user can
      // switch/restore later.
      notifyIfPossible(ctx, `Saved agent "${activeAgent}" is not available in this context`, "warning");
      activeAgent = null;
      saveAgentChoice(null);
    }

    renderCurrentAgent(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!activeAgent) {
      return undefined;
    }

    syncAgents(ctx.cwd || process.cwd());

    const available = getAvailableAgentPrompt(knownAgents, activeAgent);
    if (!available) {
      return {
        systemPrompt: `${event.systemPrompt ?? ""}\n\nNo agent profile matched active selection: ${activeAgent}.`,
      } as any;
    }

    const prompt = loadAgentPrompt(available.path);
    if (!prompt) {
      return {
        systemPrompt: `${event.systemPrompt ?? ""}\n\nNo readable prompt was found for agent ${activeAgent}.`,
      } as any;
    }

    return {
      systemPrompt:
        `${event.systemPrompt ?? ""}\n\n` +
        `# Active agent: ${activeAgent}\n\n${prompt}`,
    } as any;
  });

  pi.registerCommand("agent", {
    description: "Switch active Pi agent persona (/agent, /agent <name|number>, /agent off)",
    getArgumentCompletions: (prefix) => {
      const normalizedPrefix = prefix.trim().replace(/^\//, "");
      const args = normalizedPrefix.split(/\s+/).filter(Boolean);
      const typed = args.at(-1) ?? "";

      const staticOptions = ["off", "clear", "help", "current"];

      if (args.length <= 1) {
        return buildCompletions(
          typed,
          [...staticOptions, ...knownAgents.map((item) => item.name), "list"],
        );
      }

      // Arguments after the command are likely agent names.
      const command = args[0]?.toLowerCase() ?? "";
      const query = command ? args.slice(1).join(" ") : typed;
      const selectedOnly = knownAgents.map((agent) => agent.name);
      return buildCompletions(query || typed, selectedOnly);
    },
    handler: async (args, ctx) => {
      syncAgents(ctx.cwd || process.cwd());

      const trimmed = (args || "").trim();
      const tokens = trimmed ? trimmed.split(/\s+/) : [];
      const [firstRaw, ...rest] = tokens;
      const normalizedFirst = firstRaw ? normalizeAgentName(firstRaw) : "";

      if (normalizedFirst === "") {
        const { name: selectedName } = await chooseAgentViaPrompt(ctx, knownAgents, activeAgent);
        if (!selectedName) {
          if (activeAgent) {
            notifyIfPossible(ctx, "Agent selection cancelled.");
          } else {
            notifyIfPossible(ctx, "Agent selection cancelled.");
          }
          return;
        }

        activeAgent = selectedName;
        saveAgentChoice(selectedName);
        renderCurrentAgent(ctx);
        notifyIfPossible(ctx, `Switched to agent: ${selectedName}`);
        return;
      }

      if (normalizedFirst === "list") {
        const { name: selectedName } = await chooseAgentViaPrompt(ctx, knownAgents, activeAgent);
        if (selectedName) {
          activeAgent = selectedName;
          saveAgentChoice(selectedName);
          renderCurrentAgent(ctx);
          notifyIfPossible(ctx, `Switched to agent: ${selectedName}`);
        } else {
          notifyIfPossible(ctx, "Agent selection cancelled.");
        }
        return;
      }

      if (/^\d+$/.test(normalizedFirst)) {
        const index = Number(normalizedFirst) - 1;
        if (!Number.isInteger(index) || index < 0 || index >= knownAgents.length) {
          notifyIfPossible(ctx, `Unknown agent index "${firstRaw}".`);
          return;
        }

        const selected = knownAgents[index];
        activeAgent = selected.name;
        saveAgentChoice(selected.name);
        renderCurrentAgent(ctx);
        notifyIfPossible(ctx, `Switched to agent: ${selected.name}`);
        return;
      }

      if (["off", "clear", "none", "disable"].includes(normalizedFirst)) {
        activeAgent = null;
        saveAgentChoice(null);
        setWidgetAndStatus(ctx, null);
        notifyIfPossible(ctx, "Agent routing cleared. Pi will use default behavior.");
        return;
      }

      if (normalizedFirst === "current") {
        if (activeAgent) {
          notifyIfPossible(ctx, `Current agent: ${activeAgent}`);
        } else {
          notifyIfPossible(ctx, "No agent is currently selected.");
        }
        return;
      }

      if (normalizedFirst === "help") {
        const names = knownAgents.map((agent) => agent.name);
        notifyIfPossible(
          ctx,
          [
            "Usage: /agent <name>",
            "Commands:",
            "  /agent list      list agents",
            "  /agent off       clear active agent",
            "  /agent current   show active agent",
            names.length ? `\nAvailable: ${names.join(", ")}` : "No local agents discovered.",
          ].join("\n"),
        );
        return;
      }

      const candidate = rest.length ? [firstRaw, ...rest].join(" ").trim() : firstRaw;
      const candidateNormalized = normalizeAgentName(candidate);
      const selected = knownAgents.find((agent) => normalizeAgentName(agent.name) === candidateNormalized);

      if (!selected) {
        const suggestions = knownAgents
          .map((agent) => agent.name)
          .filter((name) => normalizeAgentName(name).startsWith(candidateNormalized))
          .slice(0, 5);

        if (suggestions.length) {
          notifyIfPossible(ctx, `Unknown agent "${candidate}". Did you mean: ${suggestions.join(", ")} ?`);
          return;
        }

        notifyIfPossible(
          ctx,
          `Unknown agent "${candidate}". Use \`/agent list\` to see available agents, then /agent <name>.`,
        );
        return;
      }

      activeAgent = selected.name;
      saveAgentChoice(selected.name);
      renderCurrentAgent(ctx);
      notifyIfPossible(ctx, `Switched to agent: ${selected.name}`);
    },
  });
}
