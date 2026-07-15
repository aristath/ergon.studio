import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import type { ChildProcess, SpawnOptions } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { Type } from "typebox";

export type DebateVerdict = "AGREE" | "CONTINUE" | "BLOCKED";
export type DebateStatus = DebateVerdict | "FAILED" | "MAX_TURNS";

export type DebateEntry = {
  turn: number;
  role: string;
  text: string;
  verdict: DebateVerdict;
};

type Participant = {
  role: string;
  sessionID: string;
};

type TurnResult = {
  exitCode: number;
  output: string;
  stderr: string;
  error?: string;
};

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

export function parseDebateVerdict(text: string): DebateVerdict {
  const lastLine = text.trim().split(/\r?\n/).at(-1)?.trim() ?? "";
  const match = lastLine.match(/^Verdict:\s*(AGREE|CONTINUE|BLOCKED)$/i);
  if (!match) return "CONTINUE";
  return match[1].toUpperCase() as DebateVerdict;
}

export function debatePrompt(
  task: string,
  selfRole: string,
  peerRole: string,
  peerOutput?: string,
): string {
  const verdict =
    "End with exactly one line: Verdict: AGREE, Verdict: CONTINUE, or Verdict: BLOCKED.";

  if (!peerOutput) {
    return [
      `You are the ${selfRole} in a two-participant coding debate with the ${peerRole}.`,
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
    `You are the ${selfRole} in a two-participant coding debate with the ${peerRole}.`,
    "",
    "Task:",
    task,
    "",
    `The ${peerRole}'s latest response:`,
    peerOutput,
    "",
    "Review it directly. If you agree it is optimal, say why and use Verdict: AGREE.",
    "If it needs work and you can improve it, make the changes or propose the correction, then use Verdict: CONTINUE.",
    "If the user needs to decide something before progress is possible, use Verdict: BLOCKED and name the decision.",
    verdict,
  ].join("\n");
}

function renderDebateTranscript(input: {
  status: DebateStatus;
  roleA: string;
  roleB: string;
  entries: DebateEntry[];
  error?: string;
}): string {
  const lines = [
    `Status: ${input.status}`,
    `Roles: ${input.roleA} <-> ${input.roleB}`,
    `Turns: ${input.entries.length}`,
  ];

  if (input.error) lines.push(`Error: ${input.error}`);

  const latest = input.entries.at(-1);
  if (latest) {
    lines.push(
      "",
      "## Latest response",
      "",
      `### ${latest.role}`,
      "",
      latest.text,
    );
  }

  lines.push("", "## Transcript");
  for (const entry of input.entries) {
    lines.push("", `### Turn ${entry.turn} - ${entry.role}`, "", entry.text);
  }

  return lines.join("\n");
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

function extractAssistantText(event: unknown): string {
  if (typeof event !== "object" || event === null) return "";
  const value = event as Record<string, unknown>;
  if (value.type !== "message_end") return "";
  if (typeof value.message !== "object" || value.message === null) return "";

  const message = value.message as Record<string, unknown>;
  if (message.role !== "assistant") return "";
  if (typeof message.content === "string") return message.content.trim();
  if (!Array.isArray(message.content)) return "";

  return message.content
    .map((part) => {
      if (typeof part !== "object" || part === null) return "";
      const content = part as Record<string, unknown>;
      return content.type === "text" && typeof content.text === "string"
        ? content.text
        : "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

async function runPiTurn(input: {
  cwd: string;
  sessionDir: string;
  sessionID: string;
  prompt: string;
  modelRef?: string;
  thinkingLevel: string;
  activeTools: string[];
  signal?: AbortSignal;
}): Promise<TurnResult> {
  const args = [
    "--mode",
    "json",
    "-p",
    "--session-dir",
    input.sessionDir,
    "--session-id",
    input.sessionID,
  ];
  if (input.modelRef) args.push("--model", input.modelRef);
  args.push("--thinking", input.thinkingLevel);
  if (input.activeTools.length > 0) {
    args.push("--tools", input.activeTools.join(","));
  } else {
    args.push("--no-tools");
  }
  args.push(input.prompt);

  const invocation = getPiInvocation(args);
  let output = "";
  let stderr = "";
  let buffer = "";
  let processError: string | undefined;
  let wasAborted = false;

  const exitCode = await new Promise<number>((resolve) => {
    const child = spawnProcess(invocation.command, invocation.args, {
      cwd: input.cwd,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let settled = false;
    let killTimer: ReturnType<typeof setTimeout> | undefined;

    const processLine = (line: string) => {
      if (!line.trim()) return;
      try {
        const text = extractAssistantText(JSON.parse(line));
        if (text) output = text;
      } catch {
        // Pi JSON mode may emit non-JSON diagnostics; stderr remains available.
      }
    };

    const finish = (code: number) => {
      if (settled) return;
      settled = true;
      if (killTimer) clearTimeout(killTimer);
      input.signal?.removeEventListener("abort", abort);
      if (buffer.trim()) processLine(buffer);
      resolve(code);
    };

    const abort = () => {
      if (settled) return;
      wasAborted = true;
      child.kill("SIGTERM");
      killTimer = setTimeout(() => {
        if (!settled) child.kill("SIGKILL");
      }, childKillGraceMs);
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

    child.on("close", (code) => finish(code ?? 1));
    child.on("error", (error) => {
      processError = error instanceof Error ? error.message : String(error);
      finish(1);
    });

    if (input.signal?.aborted) abort();
    else input.signal?.addEventListener("abort", abort, { once: true });
  });

  const error = wasAborted
    ? "Debate aborted."
    : (processError ??
      (exitCode !== 0
        ? stderr.trim() || `Pi exited with code ${exitCode}.`
        : output
          ? undefined
          : "Pi returned no assistant text."));

  return { exitCode, output, stderr, error };
}

const DebateParams = Type.Object({
  role_a: Type.String({
    minLength: 1,
    description: "Role for the participant that takes the first turn",
  }),
  role_b: Type.String({
    minLength: 1,
    description: "Role for the participant that reviews the first turn",
  }),
  task: Type.String({
    minLength: 1,
    description: "Short, specific coding task or question for the debate",
  }),
  max_turns: Type.Optional(
    Type.Integer({
      minimum: 2,
      maximum: 12,
      description: "Maximum total participant turns. Defaults to 6.",
    }),
  ),
});

export default function debateExtension(pi: ExtensionAPI): void {
  pi.registerTool({
    name: "debate",
    label: "Debate",
    description:
      "Run two independent Pi sessions in an alternating coding debate until one agrees, blocks, or max_turns is reached.",
    parameters: DebateParams,
    execute: async (_toolCallId, params, signal, onUpdate, ctx) => {
      const maxTurns = params.max_turns ?? 6;
      const sessionDir = mkdtempSync(join(tmpdir(), "ergon-pi-debate-"));
      const participants: Participant[] = [
        { role: params.role_a, sessionID: randomUUID() },
        { role: params.role_b, sessionID: randomUUID() },
      ];
      const entries: DebateEntry[] = [];
      const modelRef = ctx.model
        ? `${ctx.model.provider}/${ctx.model.id}`
        : undefined;
      const thinkingLevel = pi.getThinkingLevel();
      const activeTools = pi
        .getActiveTools()
        .filter((toolName) => toolName !== "debate");
      let latestText = "";
      let status: DebateStatus = "MAX_TURNS";
      let error: string | undefined;

      try {
        try {
          for (let turn = 1; turn <= maxTurns; turn++) {
            const current = participants[(turn - 1) % 2];
            const peer = participants[turn % 2];
            const prompt = debatePrompt(
              params.task,
              current.role,
              peer.role,
              turn === 1 ? undefined : latestText,
            );
            const result = await runPiTurn({
              cwd: ctx.cwd,
              sessionDir,
              sessionID: current.sessionID,
              prompt,
              modelRef,
              thinkingLevel,
              activeTools,
              signal,
            });

            if (result.error) {
              status = "FAILED";
              error = `Turn ${turn} (${current.role}): ${result.error}`;
              break;
            }

            const verdict = parseDebateVerdict(result.output);
            entries.push({
              turn,
              role: current.role,
              text: result.output,
              verdict,
            });
            latestText = result.output;

            onUpdate?.({
              content: [
                {
                  type: "text",
                  text: `Debate turn ${turn}/${maxTurns}: ${current.role} -> ${verdict}`,
                },
              ],
              details: {
                status: "RUNNING",
                roleA: params.role_a,
                roleB: params.role_b,
                entries: [...entries],
              },
            });

            if (turn > 1 && (verdict === "AGREE" || verdict === "BLOCKED")) {
              status = verdict;
              break;
            }
          }
        } catch (caught) {
          status = "FAILED";
          error = caught instanceof Error ? caught.message : String(caught);
        }

        const details = {
          status,
          roleA: params.role_a,
          roleB: params.role_b,
          entries,
          error,
        };
        return {
          content: [
            {
              type: "text",
              text: renderDebateTranscript(details),
            },
          ],
          details,
          isError: status === "FAILED",
        };
      } finally {
        rmSync(sessionDir, { recursive: true, force: true });
      }
    },
  });
}
