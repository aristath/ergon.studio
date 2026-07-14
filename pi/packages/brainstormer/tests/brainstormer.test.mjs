import assert from "node:assert/strict";
import test from "node:test";

import brainstormExtension, {
  BRAINSTORM_SYSTEM_PROMPT,
  deriveStateFromBranch,
} from "../dist/extensions/index.js";

function makeHarness(options = {}) {
  const commands = new Map();
  const handlers = new Map();
  const branch = [];
  let activeTools = ["read", "find", "grep", "bash", "edit", "write"];
  const toolApiCalls = [];
  const notifications = [];
  const widgets = new Map();
  const statuses = new Map();
  const selects = [];
  let selectChoice = "Continue brainstorming";

  const pi = {
    registerCommand(name, options) {
      commands.set(name, options);
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
        "subagent",
        "get_subagent_result",
      ].map((name) => ({ name }));
    },
  };

  const ctx = {
    hasUI: options.hasUI ?? true,
    cwd: "/tmp/brainstorm",
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

  brainstormExtension(pi);

  return {
    commands,
    handlers,
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

test("derives active and inactive state from session markers", () => {
  const active = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: {
        action: "start",
        startedAt: 1,
        previousTools: ["read", "bash"],
      },
    },
  ]);

  assert.equal(active.active, true);
  assert.equal("previousTools" in active, false);

  const inactive = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "2",
      parentId: "1",
      timestamp: new Date().toISOString(),
      data: { action: "done" },
    },
  ]);

  assert.equal(inactive.active, false);

  const supersededByPlan = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
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
  assert.equal(supersededByPlan.supersededByPlan, true);

  const ignoredWhilePlanActive = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start" },
    },
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "2",
      parentId: "1",
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
  ]);

  assert.equal(ignoredWhilePlanActive.active, false);
  assert.equal(ignoredWhilePlanActive.supersededByPlan, true);

  const brainstormAfterPlanEnds = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start" },
    },
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "2",
      parentId: "1",
      timestamp: new Date().toISOString(),
      data: { action: "cancel" },
    },
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "3",
      parentId: "2",
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
  ]);

  assert.equal(brainstormAfterPlanEnds.active, true);
  assert.equal(brainstormAfterPlanEnds.supersededByPlan, false);
});

test("registers only /brainstorm and starts without changing tools", async () => {
  const harness = makeHarness();

  assert.deepEqual([...harness.commands.keys()], ["brainstorm"]);

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.branch[0].customType, "ergon-brainstorm-state");
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
  assert.equal(harness.handlers.has("tool_call"), false);
  assert.equal(harness.statuses.get("brainstorm"), "brainstorm");
  assert.match(harness.notifications[0].message, /Brainstorm mode started/);
});

test("injects brainstorm prompt while active", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);
  const result = await harness.handlers.get("before_agent_start")(
    { systemPrompt: "Base system prompt" },
    harness.ctx,
  );

  assert.match(result.systemPrompt, /Base system prompt/);
  assert.match(result.systemPrompt, /thinking partner/);
  assert.match(result.systemPrompt, /Want to shift into planning/);
  assert.match(result.systemPrompt, /Pi \/brainstorm Mode Boundary/);
  assert.match(BRAINSTORM_SYSTEM_PROMPT, /There is no required topic/);
  assert.match(BRAINSTORM_SYSTEM_PROMPT, /active tool selection remains unchanged/);
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

test("opens a menu on repeated /brainstorm and leaves tools unchanged", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);
  harness.setSelectChoice("Done brainstorming");
  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.selects.length, 1);
  assert.deepEqual(harness.selects[0].options, [
    "Continue brainstorming",
    "Done brainstorming",
    "Cancel brainstorming",
  ]);
  assert.equal(harness.branch.at(-1).data.action, "done");
  assert.deepEqual(harness.activeTools, [
    "read",
    "find",
    "grep",
    "bash",
    "edit",
    "write",
  ]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.match(harness.notifications.at(-1).message, /run \/plan/);
});

test("/brainstorm manages restored active brainstorm state", async () => {
  const harness = makeHarness();

  harness.branch.push({
    type: "custom",
    customType: "ergon-brainstorm-state",
    id: "brainstorm",
    parentId: null,
    timestamp: new Date().toISOString(),
    data: {
      action: "start",
      startedAt: 1,
      previousTools: ["read", "bash"],
    },
  });
  harness.setSelectChoice("Cancel brainstorming");

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.selects.length, 1);
  assert.equal(harness.branch.length, 2);
  assert.equal(harness.branch.at(-1).data.action, "cancel");
});

test("repeated /brainstorm exits as done without UI", async () => {
  const harness = makeHarness({ hasUI: false });

  await harness.commands.get("brainstorm").handler("", harness.ctx);
  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.branch.at(-1).customType, "ergon-brainstorm-state");
  assert.equal(harness.branch.at(-1).data.action, "done");
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

test("/brainstorm does not start while /plan is active", async () => {
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

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.branch.length, 1);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.statuses.get("brainstorm"), undefined);
  assert.equal(harness.notifications.at(-1).level, "warning");
  assert.match(harness.notifications.at(-1).message, /Plan mode is active/);
});

test("restored branches cannot keep brainstorm active over an active plan", async () => {
  const harness = makeHarness();

  harness.branch.push(
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "plan",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: { action: "start" },
    },
    {
      type: "custom",
      customType: "ergon-brainstorm-state",
      id: "brainstorm",
      parentId: "plan",
      timestamp: new Date().toISOString(),
      data: { action: "start", previousTools: ["read"] },
    },
  );
  harness.setActiveToolsForTest(["read", "edit", "write"]);

  await harness.handlers.get("session_start")({}, harness.ctx);
  const promptResult = await harness.handlers.get("before_agent_start")(
    { systemPrompt: "Base system prompt" },
    harness.ctx,
  );

  assert.equal(promptResult, undefined);
  assert.equal(harness.handlers.has("tool_call"), false);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.statuses.get("brainstorm"), undefined);
});

test("cancel exits without planning nudge", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);
  harness.setSelectChoice("Cancel brainstorming");
  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.branch.at(-1).data.action, "cancel");
  assert.equal(harness.notifications.at(-1).message, "Brainstorm mode cancelled.");
});

test("does not install a mode-level tool-call blocker", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.handlers.has("tool_call"), false);
  assert.deepEqual(harness.toolApiCalls, []);
});

test("later /plan marker supersedes brainstorm without changing tools", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);
  harness.branch.push({
    type: "custom",
    customType: "ergon-plan-state",
    id: "plan",
    parentId: harness.branch.at(-1)?.id ?? null,
    timestamp: new Date().toISOString(),
    data: { action: "start" },
  });
  harness.setActiveToolsForTest(["read", "edit", "write"]);

  const promptResult = await harness.handlers.get("before_agent_start")(
    { systemPrompt: "Base system prompt" },
    harness.ctx,
  );

  assert.equal(promptResult, undefined);
  assert.equal(harness.handlers.has("tool_call"), false);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.deepEqual(harness.toolApiCalls, []);
  assert.equal(harness.statuses.get("brainstorm"), undefined);
});
