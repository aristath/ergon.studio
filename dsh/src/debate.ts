// Debate + run_parallel execution, on the DSH subagent seam.
//
// Ported from the OpenCode plugin's `debate` / `run_parallel` tools. The
// OpenCode version created named sub-sessions per agent; the DSH version
// spawns one-shot children (`ctx.subagents.start("spawn", ...)`) whose
// persona and tool filter come from the roster. Spawn children are stateless
// (zero parent context), so each debate turn's prompt carries the task, the
// debater's own previous response, and the peer's latest response — the
// full picture a continuing session would have had.

import type { RosterEntry } from "./roster.js";

// === verdict handling (verbatim port) ===

export type DebateVerdict = "AGREE" | "CONTINUE" | "BLOCKED";

export function parseDebateVerdict(text: string): DebateVerdict {
  const lastLine = text.trim().split(/\r?\n/).at(-1)?.trim() ?? "";
  const match = lastLine.match(/^Verdict:\s*(AGREE|CONTINUE|BLOCKED)$/i);
  if (!match) return "CONTINUE";
  return match[1].toUpperCase() as DebateVerdict;
}

export function debatePrompt(
  task: string,
  selfAgent: string,
  peerAgent: string,
  ownPrevious?: string,
  peerOutput?: string,
): string {
  const verdict =
    "End with exactly one line: Verdict: AGREE, Verdict: CONTINUE, or Verdict: BLOCKED.";

  if (!peerOutput) {
    return [
      `You are ${selfAgent} in a two-agent coding debate with ${peerAgent}.`,
      "",
      "Task:",
      task,
      "",
      "Do the first pass. If code changes are needed, make the changes you believe are right.",
      "Keep the response focused: what you did or propose, why, and what needs review.",
      "Use Verdict: CONTINUE unless you cannot proceed without the user.",
      verdict,
    ].join("\n");
  }

  return [
    `You are ${selfAgent} in a two-agent coding debate with ${peerAgent}.`,
    "",
    "Task:",
    task,
    "",
    "Your previous response:",
    ownPrevious ?? "(none)",
    "",
    `${peerAgent}'s latest response:`,
    peerOutput,
    "",
    "Review it directly. If you agree it is optimal, say why and use Verdict: AGREE.",
    "If it needs work and you can improve it, make the changes or propose the correction, then use Verdict: CONTINUE.",
    "If the user needs to decide something before progress is possible, use Verdict: BLOCKED and name the decision.",
    verdict,
  ].join("\n");
}

export interface DebateEntry {
  turn: number;
  agent: string;
  text: string;
  verdict: DebateVerdict;
}

export function renderDebateTranscript(input: {
  status: DebateVerdict | "FAILED" | "MAX_TURNS" | "ABORTED";
  agentA: string;
  agentB: string;
  entries: DebateEntry[];
  error?: string;
}): string {
  const lines = [
    "# Debate result",
    "",
    `Status: ${input.status}`,
    `Participants: ${input.agentA}, ${input.agentB}`,
    `Turns: ${input.entries.length}`,
  ];

  if (input.error) {
    lines.push("", "## Error", "", input.error);
  }

  const latest = input.entries.at(-1);
  if (latest) {
    lines.push("", "## Latest response", "", `### ${latest.agent}`, "", latest.text);
  }

  lines.push("", "## Transcript");
  for (const entry of input.entries) {
    lines.push("", `### Turn ${entry.turn} - ${entry.agent}`, "", entry.text);
  }

  return lines.join("\n");
}

// === subagent seam types (structural, matching dsh-subagent contracts) ===

export interface SubagentRun {
  result: Promise<{
    output: unknown;
    structured?: unknown;
    diagnostic?: string;
    stopReason: string;
  }>;
  dispose(): Promise<void> | void;
}

export interface SubagentStartOptions {
  label: string;
  prompt: string;
  parent: unknown;
  signal: AbortSignal;
  persona: string;
  toolFilter: { deny: string[] };
}

export type SubagentsService = {
  start(name: string, request: Record<string, unknown>): Promise<SubagentRun>;
};

/**
 * Extract assistant text from a settled one-shot child's `output`
 * (AssistantOutputFold: the child's last non-empty assistant message, or its
 * accumulated assistant text). Tolerates content-block arrays and plain
 * strings.
 */
export function outputText(output: unknown): string {
  if (typeof output === "string") return output.trim();
  if (Array.isArray(output)) {
    return output
      .map((part) => {
        if (part && typeof part === "object" && (part as any).type === "text" && typeof (part as any).text === "string") {
          return (part as any).text as string;
        }
        return "";
      })
      .filter((text) => text !== "")
      .join("\n")
      .trim();
  }
  if (output && typeof output === "object" && typeof (output as any).text === "string") {
    return (output as any).text as string;
  }
  return "";
}

// === debate loop ===

export interface DebateParams {
  agentA: RosterEntry;
  agentB: RosterEntry;
  task: string;
  maxTurns: number;
  parent: unknown;
  signal: AbortSignal;
  subagents: SubagentsService;
}

export interface DebateResult {
  transcript: string;
}

export async function runDebate(params: DebateParams): Promise<DebateResult> {
  const { agentA, agentB, task, maxTurns, parent, signal, subagents } = params;
  const entries: DebateEntry[] = [];
  let latestA = "";
  let latestB = "";
  let status: DebateVerdict | "MAX_TURNS" | "ABORTED" | "FAILED" = "MAX_TURNS";
  let error: string | undefined;

  try {
    for (let turn = 1; turn <= maxTurns; turn++) {
      if (signal.aborted) {
        status = "ABORTED";
        break;
      }
      const current = turn % 2 === 1 ? agentA : agentB;
      const peer = turn % 2 === 1 ? agentB : agentA;
      const ownPrevious = turn % 2 === 1 ? latestA : latestB;
      const peerOutput = turn === 1 ? undefined : turn % 2 === 1 ? latestB : latestA;

      const prompt = debatePrompt(task, current.id, peer.id, ownPrevious, peerOutput);

      const run = await subagents.start("spawn", {
        label: `${current.id} (debate t${turn})`,
        prompt: [{ type: "text", text: prompt }],
        parent,
        signal,
        persona: current.persona,
        toolFilter: { deny: current.deny },
      });

      const result = await run.result;
      void run.dispose();

      if (signal.aborted) {
        status = "ABORTED";
        break;
      }
      const text = outputText(result.output) || `(${result.stopReason})`;
      const verdict = parseDebateVerdict(text);
      entries.push({ turn, agent: current.id, text, verdict });
      if (turn % 2 === 1) latestA = text;
      else latestB = text;

      if (turn > 1 && (verdict === "AGREE" || verdict === "BLOCKED")) {
        status = verdict;
        break;
      }
    }
  } catch (err) {
    status = "FAILED";
    error = err instanceof Error ? err.message : String(err);
  }

  const transcript = renderDebateTranscript({
    status,
    agentA: agentA.id,
    agentB: agentB.id,
    entries,
    error,
  });
  return { transcript };
}

// === parallel fan-out ===

export interface ParallelTask {
  agent: RosterEntry;
  brief: string;
}

export interface ParallelParams {
  tasks: ParallelTask[];
  parent: unknown;
  signal: AbortSignal;
  subagents: SubagentsService;
}

export async function runParallel(params: ParallelParams): Promise<string> {
  const { tasks, parent, signal, subagents } = params;
  const results = await Promise.all(
    tasks.map(async (task) => {
      try {
        const run = await subagents.start("spawn", {
          label: `${task.agent.id} (parallel)`,
          prompt: [{ type: "text", text: task.brief }],
          parent,
          signal,
          persona: task.agent.persona,
          toolFilter: { deny: task.agent.deny },
        });
        const result = await run.result;
        void run.dispose();
        const text = outputText(result.output) || `(${result.stopReason})`;
        return `## ${task.agent.id}\n\n${text}`;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return `## ${task.agent.id}\n\n⚠️ Task failed: ${message}`;
      }
    }),
  );
  return results.join("\n\n---\n\n");
}
