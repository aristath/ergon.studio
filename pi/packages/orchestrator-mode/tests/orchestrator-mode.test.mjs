import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import orchestratorExtension, {
  deriveStateFromBranch,
  loadAgentDefinition,
  ORCHESTRATOR_SYSTEM_PROMPT,
  setChildKillGraceMsForTest,
  setSpawnProcessForTest,
} from "../dist/extensions/index.js";

function createFakeChild({
  stdoutLines = [],
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
      for (const line of stdoutLines) {
        child.stdout.write(`${JSON.stringify(line)}\n`);
      }
      if (stderr) {
        child.stderr.write(stderr);
      }
      child.emit("close", closeCode);
    });
  }

  return child;
}

function makeHarness() {
  const commands = new Map();
  const handlers = new Map();
  const tools = new Map();
  const branch = [];
  let activeTools = ["read", "find", "grep", "bash", "edit", "write"];
  const toolApiCalls = [];
  const notifications = [];
  const widgets = new Map();
  const statuses = new Map();
  const selects = [];
  let selectChoice = "Continue orchestrating";

  const pi = {
    registerCommand(name, options) {
      commands.set(name, options);
    },
    registerTool(tool) {
      tools.set(tool.name, tool);
    },
    on(event, handler) {
      handlers.set(event, handler);
    },
    appendEntry(customType, data) {
      branch.push({
        type: "custom",
        customType,
        data,
        id: String(branch.length),
        parentId: branch.at(-1)?.id ?? null,
        timestamp: new Date().toISOString(),
      });
    },
    getActiveTools() {
      toolApiCalls.push("getActiveTools");
      return [...activeTools];
    },
    setActiveTools(toolNames) {
      toolApiCalls.push("setActiveTools");
      activeTools = [...toolNames];
    },
    getAllTools() {
      toolApiCalls.push("getAllTools");
      return [
        "read",
        "find",
        "grep",
        "ls",
        "bash",
        "edit",
        "write",
        "ask_user_question",
        "task",
        "run_parallel",
        "subagent",
        "get_subagent_result",
      ].map((name) => ({ name }));
    },
  };

  const ctx = {
    hasUI: true,
    cwd: "/tmp/orchestrator",
    sessionManager: {
      getBranch() {
        return branch;
      },
    },
    ui: {
      notify(message, level = "info") {
        notifications.push({ message, level });
      },
      setStatus(key, value) {
        statuses.set(key, value);
      },
      setWidget(key, value) {
        widgets.set(key, value);
      },
      async select(title, options) {
        selects.push({ title, options });
        return selectChoice;
      },
    },
  };

  orchestratorExtension(pi);

  return {
    commands,
    handlers,
    tools,
    branch,
    notifications,
    selects,
    statuses,
    toolApiCalls,
    widgets,
    ctx,
    get activeTools() {
      return activeTools;
    },
    setActiveToolsForTest(toolNames) {
      activeTools = [...toolNames];
    },
    setSelectChoice(value) {
      selectChoice = value;
    },
  };
}

test("loads aligned orchestrator and quality agent prompts", () => {
  const orchestrator = loadAgentDefinition("orchestrator");
  const quality = loadAgentDefinition("quality_controller");
  const reviewer = loadAgentDefinition("reviewer");
  const coder = loadAgentDefinition("coder");

  assert.match(orchestrator.prompt, /changes executable behavior/);
  assert.match(orchestrator.prompt, /`quality_controller` agent/);
  assert.match(orchestrator.prompt, /Track the rejection count in this parent session/);
  assert.match(quality.prompt, /final quality gate/);
  assert.match(quality.prompt, /Invoke the \*\*reviewer\*\* agent/);
  assert.match(quality.prompt, /Phase 3: Verification Evidence/);
  assert.match(quality.prompt, /Verdict: APPROVED/);
  assert.doesNotMatch(quality.prompt, /COMPLETION\.md/);
  assert.match(ORCHESTRATOR_SYSTEM_PROMPT, /Pi \/orchestrator Mode Boundary/);
  assert.match(ORCHESTRATOR_SYSTEM_PROMPT, /quality gate remains agent-owned/);
  assert.deepEqual(quality.tools, ["read", "find", "grep", "ls", "task"]);
  assert.deepEqual(reviewer.tools, ["read", "find", "grep", "ls", "bash"]);
  assert.ok(coder.tools.includes("edit"));
  assert.ok(coder.tools.includes("write"));
});

test("derives mode state around plan and brainstorm markers", () => {
  const active = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-orchestrator-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read", "bash"] },
    },
  ]);

  assert.equal(active.active, true);
  assert.equal("previousTools" in active, false);

  const supersededByPlan = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-orchestrator-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "2",
      parentId: "1",
      timestamp: new Date().toISOString(),
      data: { action: "start" },
    },
  ]);

  assert.equal(supersededByPlan.active, false);
  assert.equal(supersededByPlan.supersededByMode, true);

  const ignoredWhileBrainstormActive = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start" },
    },
    {
      type: "custom",
      customType: "ergon-orchestrator-state",
      id: "2",
      parentId: "1",
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
  ]);

  assert.equal(ignoredWhileBrainstormActive.active, false);
  assert.equal(ignoredWhileBrainstormActive.supersededByMode, true);
});

test("registers command and delegation tools", () => {
  const harness = makeHarness();

  assert.deepEqual([...harness.commands.keys()], ["orchestrator"]);
  assert.equal(typeof harness.tools.get("task").execute, "function");
  assert.equal(typeof harness.tools.get("run_parallel").execute, "function");
  assert.equal(harness.handlers.has("tool_call"), false);
});

test("starts orchestrator mode without changing tools and injects prompt", async () => {
  const harness = makeHarness();

  await harness.commands.get("orchestrator").handler("", harness.ctx);

  assert.equal(harness.branch[0].customType, "ergon-orchestrator-state");
  assert.equal(harness.branch[0].data.action, "start");
  assert.equal("previousTools" in harness.branch[0].data, false);
  assert.deepEqual(harness.activeTools, [
    "read",
    "find",
    "grep",
    "bash",
    "edit",
    "write",
  ]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.statuses.get("orchestrator"), "orchestrator");

  const result = await harness.handlers.get("before_agent_start")(
    { systemPrompt: "Base system prompt" },
    harness.ctx,
  );

  assert.match(result.systemPrompt, /Base system prompt/);
  assert.match(result.systemPrompt, /You are the lead dev/);
  assert.match(result.systemPrompt, /## Quality Gate/);
  assert.match(result.systemPrompt, /changes executable behavior/);
});

test("active /orchestrator opens menu and finish leaves tools unchanged", async () => {
  const harness = makeHarness();

  await harness.commands.get("orchestrator").handler("", harness.ctx);
  harness.setSelectChoice("Finish orchestrating");
  await harness.commands.get("orchestrator").handler("", harness.ctx);

  assert.equal(harness.selects.length, 1);
  assert.equal(harness.branch.at(-1).data.action, "finish");
  assert.deepEqual(harness.activeTools, [
    "read",
    "find",
    "grep",
    "bash",
    "edit",
    "write",
  ]);
  assert.deepEqual(harness.toolApiCalls, []);
});

test("/orchestrator refuses to start while /plan is active", async () => {
  const harness = makeHarness();

  harness.branch.push({
    type: "custom",
    customType: "ergon-plan-state",
    id: "plan",
    parentId: null,
    timestamp: new Date().toISOString(),
    data: { action: "start" },
  });
  harness.setActiveToolsForTest(["read", "edit", "write"]);

  await harness.commands.get("orchestrator").handler("", harness.ctx);

  assert.equal(harness.branch.length, 1);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.notifications.at(-1).level, "warning");
});

test("a later mode marker supersedes orchestrator without changing tools", async () => {
  const harness = makeHarness();

  await harness.commands.get("orchestrator").handler("", harness.ctx);
  harness.branch.push({
    type: "custom",
    customType: "ergon-brainstorm-state",
    id: "brainstorm",
    parentId: harness.branch.at(-1)?.id ?? null,
    timestamp: new Date().toISOString(),
    data: { action: "start" },
  });
  harness.setActiveToolsForTest(["read", "edit", "write"]);

  const result = await harness.handlers.get("before_agent_start")(
    { systemPrompt: "Base system prompt" },
    harness.ctx,
  );

  assert.equal(result, undefined);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.statuses.get("orchestrator"), undefined);
});

test("task tool reports unknown bundled agents without spawning", async () => {
  const harness = makeHarness();

  const result = await harness.tools
    .get("task")
    .execute(
      "tool-1",
      { agent: "missing", brief: "Do work" },
      undefined,
      undefined,
      harness.ctx,
    );

  assert.equal(result.isError, true);
  assert.match(result.content[0].text, /Unknown agent/);
  assert.match(result.content[0].text, /quality_controller/);
});

test("does not mode-scope delegation tools with a tool-call blocker", () => {
  const harness = makeHarness();

  assert.equal(harness.handlers.has("tool_call"), false);
  assert.deepEqual(harness.toolApiCalls, []);
});

test("task tool runs bundled agents through child pi with specialist tools", async () => {
  const harness = makeHarness();
  const calls = [];

  setSpawnProcessForTest((command, args, options) => {
    calls.push({ command, args, options });
    return createFakeChild({
      stdoutLines: [
        {
          type: "message_end",
          message: {
            role: "assistant",
            content: [{ type: "text", text: "Review complete." }],
          },
        },
      ],
    });
  });

  try {
    const result = await harness.tools
      .get("task")
      .execute(
        "tool-1",
        { agent: "reviewer", brief: "Review the implementation" },
        undefined,
        undefined,
        harness.ctx,
      );

    assert.equal(result.isError, false);
    assert.match(result.content[0].text, /Review complete/);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].options.env, undefined);
    assert.equal(calls[0].options.cwd, "/tmp/orchestrator");
    assert.ok(calls[0].args.includes("--mode"));
    assert.ok(calls[0].args.includes("json"));
    assert.ok(calls[0].args.includes("--no-session"));
    assert.ok(calls[0].args.includes("--append-system-prompt"));
    assert.ok(calls[0].args.includes("--tools"));
    assert.ok(calls[0].args.includes("read,find,grep,ls,bash"));
  } finally {
    setSpawnProcessForTest();
  }
});

test("aborted child tasks escalate to SIGKILL if the process does not close", async () => {
  const harness = makeHarness();
  const controller = new AbortController();
  const signals = [];

  setChildKillGraceMsForTest(1);
  setSpawnProcessForTest(() =>
    createFakeChild({
      autoClose: false,
      onKill(signal, child) {
        signals.push(signal);
        if (signal === "SIGKILL") {
          child.emit("close", null);
        }
      },
    }),
  );

  try {
    const run = harness.tools
      .get("task")
      .execute(
        "tool-1",
        { agent: "reviewer", brief: "Review the implementation" },
        controller.signal,
        undefined,
        harness.ctx,
      );

    controller.abort();
    const result = await run;

    assert.equal(result.isError, true);
    assert.match(result.content[0].text, /aborted/);
    assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
  } finally {
    setChildKillGraceMsForTest();
    setSpawnProcessForTest();
  }
});
