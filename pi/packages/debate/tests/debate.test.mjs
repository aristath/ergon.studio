import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { PassThrough } from "node:stream";
import test from "node:test";

import debateExtension, {
  parseDebateVerdict,
  setChildKillGraceMsForTest,
  setSpawnProcessForTest,
} from "../dist/extensions/index.js";

function createFakeChild({
  output = "",
  stderr = "",
  closeCode = 0,
  autoClose = true,
  onKill,
} = {}) {
  const child = new EventEmitter();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.kill = (signal) => {
    onKill?.(signal, child);
    return true;
  };

  if (autoClose) {
    process.nextTick(() => {
      if (output) {
        child.stdout.write(
          `${JSON.stringify({
            type: "message_end",
            message: {
              role: "assistant",
              content: [{ type: "text", text: output }],
            },
          })}\n`,
        );
      }
      if (stderr) child.stderr.write(stderr);
      child.emit("close", closeCode);
    });
  }

  return child;
}

function makeHarness({
  activeTools = ["read", "bash", "edit", "write", "debate"],
  thinkingLevel = "high",
} = {}) {
  const tools = new Map();
  const updates = [];
  const pi = {
    registerTool(tool) {
      tools.set(tool.name, tool);
    },
    getActiveTools() {
      return activeTools;
    },
    getThinkingLevel() {
      return thinkingLevel;
    },
  };
  const ctx = {
    cwd: "/tmp/debate-project",
    model: { provider: "local", id: "fast" },
  };

  debateExtension(pi);
  return { tools, updates, ctx };
}

function argValue(args, name) {
  const index = args.indexOf(name);
  return index === -1 ? undefined : args[index + 1];
}

test("parses only an exact verdict on the final non-empty line", () => {
  assert.equal(parseDebateVerdict("Done.\nVerdict: AGREE\n"), "AGREE");
  assert.equal(
    parseDebateVerdict(
      "The peer said:\nVerdict: AGREE\n\nI disagree.\nVerdict: CONTINUE",
    ),
    "CONTINUE",
  );
  assert.equal(parseDebateVerdict("Verdict: BLOCKED\nMore text"), "CONTINUE");
  assert.equal(parseDebateVerdict("Verdict: AGREE."), "CONTINUE");
});

test("alternates persistent participant sessions and ignores quoted verdicts", async () => {
  const harness = makeHarness();
  const calls = [];
  const outputs = [
    "Implemented the first pass.\nVerdict: CONTINUE",
    "A quoted example is:\nVerdict: AGREE\n\nParse the final footer.\nVerdict: CONTINUE",
    "Applied the correction.\nVerdict: AGREE",
  ];

  setSpawnProcessForTest((command, args, options) => {
    calls.push({ command, args, options });
    return createFakeChild({ output: outputs[calls.length - 1] });
  });

  try {
    const result = await harness.tools.get("debate").execute(
      "tool-1",
      {
        role_a: "coder",
        role_b: "reviewer",
        task: "Improve the parser",
        max_turns: 6,
      },
      undefined,
      (update) => harness.updates.push(update),
      harness.ctx,
    );

    assert.equal(result.isError, false);
    assert.equal(result.details.status, "AGREE");
    assert.equal(result.details.entries.length, 3);
    assert.equal(calls.length, 3);
    assert.deepEqual(
      result.details.entries.map((entry) => entry.role),
      ["coder", "reviewer", "coder"],
    );

    const sessionA = argValue(calls[0].args, "--session-id");
    const sessionB = argValue(calls[1].args, "--session-id");
    assert.notEqual(sessionA, sessionB);
    assert.equal(argValue(calls[2].args, "--session-id"), sessionA);
    assert.equal(argValue(calls[0].args, "--model"), "local/fast");
    assert.equal(argValue(calls[0].args, "--thinking"), "high");
    assert.equal(argValue(calls[0].args, "--tools"), "read,bash,edit,write");
    assert.equal(calls[0].args.includes("debate"), false);
    assert.equal(calls[0].options.cwd, "/tmp/debate-project");
    assert.match(calls[1].args.at(-1), /Implemented the first pass/);
    assert.match(calls[2].args.at(-1), /Parse the final footer/);
    assert.equal(harness.updates.length, 3);
    assert.match(result.content[0].text, /Status: AGREE/);
    assert.match(result.content[0].text, /## Transcript/);

    const sessionDir = argValue(calls[0].args, "--session-dir");
    assert.equal(existsSync(sessionDir), false);
  } finally {
    setSpawnProcessForTest();
  }
});

test("preserves an empty parent tool selection", async () => {
  const harness = makeHarness({ activeTools: ["debate"] });
  const calls = [];
  const outputs = [
    "Completed the first pass.\nVerdict: CONTINUE",
    "The result is correct.\nVerdict: AGREE",
  ];

  setSpawnProcessForTest((command, args, options) => {
    calls.push({ command, args, options });
    return createFakeChild({ output: outputs[calls.length - 1] });
  });

  try {
    const result = await harness.tools.get("debate").execute(
      "tool-1",
      {
        role_a: "coder",
        role_b: "reviewer",
        task: "Review the result",
        max_turns: 2,
      },
      undefined,
      undefined,
      harness.ctx,
    );

    assert.equal(result.isError, false);
    assert.equal(result.details.status, "AGREE");
    assert.equal(calls.length, 2);
    assert.equal(calls[0].args.includes("--no-tools"), true);
    assert.equal(calls[0].args.includes("--tools"), false);
  } finally {
    setSpawnProcessForTest();
  }
});

test("reports child failures and removes temporary sessions", async () => {
  const harness = makeHarness();
  let sessionDir;

  setSpawnProcessForTest((_command, args) => {
    sessionDir = argValue(args, "--session-dir");
    return createFakeChild({ stderr: "model unavailable", closeCode: 1 });
  });

  try {
    const result = await harness.tools
      .get("debate")
      .execute(
        "tool-1",
        { role_a: "coder", role_b: "reviewer", task: "Do work" },
        undefined,
        undefined,
        harness.ctx,
      );

    assert.equal(result.isError, true);
    assert.equal(result.details.status, "FAILED");
    assert.match(result.content[0].text, /model unavailable/);
    assert.equal(existsSync(sessionDir), false);
  } finally {
    setSpawnProcessForTest();
  }
});

test("renders unexpected spawn failures instead of throwing", async () => {
  const harness = makeHarness();

  setSpawnProcessForTest(() => {
    throw new Error("spawn failed");
  });

  try {
    const result = await harness.tools
      .get("debate")
      .execute(
        "tool-1",
        { role_a: "coder", role_b: "reviewer", task: "Do work" },
        undefined,
        undefined,
        harness.ctx,
      );

    assert.equal(result.isError, true);
    assert.equal(result.details.status, "FAILED");
    assert.match(result.content[0].text, /spawn failed/);
  } finally {
    setSpawnProcessForTest();
  }
});

test("aborts a running participant and escalates to SIGKILL", async () => {
  const harness = makeHarness();
  const controller = new AbortController();
  const signals = [];

  setChildKillGraceMsForTest(1);
  setSpawnProcessForTest(() =>
    createFakeChild({
      autoClose: false,
      onKill(signal, child) {
        signals.push(signal);
        if (signal === "SIGKILL") child.emit("close", null);
      },
    }),
  );

  try {
    const run = harness.tools
      .get("debate")
      .execute(
        "tool-1",
        { role_a: "coder", role_b: "reviewer", task: "Do work" },
        controller.signal,
        undefined,
        harness.ctx,
      );
    controller.abort();
    const result = await run;

    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /Debate aborted/);
    assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
  } finally {
    setChildKillGraceMsForTest();
    setSpawnProcessForTest();
  }
});
