import assert from "node:assert/strict";
import test from "node:test";

import brainstormExtension, {
  BRAINSTORM_SYSTEM_PROMPT,
  deriveStateFromBranch,
  getBrainstormTools,
} from "../dist/extensions/index.js";

function makeHarness(options = {}) {
  const commands = new Map();
  const handlers = new Map();
  const branch = [];
  let activeTools = ["read", "find", "grep", "bash", "edit", "write"];
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
      return [...activeTools];
    },
    setActiveTools(toolNames) {
      activeTools = [...toolNames];
    },
    getAllTools() {
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
  assert.deepEqual(active.previousTools, ["read", "bash"]);

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

test("detects available brainstorm tools", () => {
  const tools = getBrainstormTools({
    getAllTools() {
      return ["read", "grep", "write", "subagent"].map((name) => ({ name }));
    },
  });

  assert.deepEqual(tools, ["read", "grep", "subagent"]);
});

test("registers only /brainstorm and starts freeform mode", async () => {
  const harness = makeHarness();

  assert.deepEqual([...harness.commands.keys()], ["brainstorm"]);

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  assert.equal(harness.branch[0].customType, "ergon-brainstorm-state");
  assert.equal(harness.branch[0].data.action, "start");
  assert.deepEqual(harness.activeTools, [
    "read",
    "find",
    "grep",
    "ls",
    "ask_user_question",
    "subagent",
    "get_subagent_result",
  ]);
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
  assert.deepEqual(harness.activeTools, [
    "read",
    "find",
    "grep",
    "ls",
    "ask_user_question",
    "subagent",
    "get_subagent_result",
  ]);
});

test("opens a menu on repeated /brainstorm and done restores tools", async () => {
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
  const toolResult = await harness.handlers.get("tool_call")(
    {
      type: "tool_call",
      toolName: "edit",
      input: { path: ".ergon.studio/scratchpad.md", edits: [] },
      toolCallId: "5",
    },
    harness.ctx,
  );

  assert.equal(promptResult, undefined);
  assert.equal(toolResult, undefined);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
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

test("blocks implementation tools and only permits Explore subagents", async () => {
  const harness = makeHarness();

  await harness.commands.get("brainstorm").handler("", harness.ctx);

  const editResult = await harness.handlers.get("tool_call")(
    {
      type: "tool_call",
      toolName: "edit",
      input: { path: "src/index.ts", edits: [] },
      toolCallId: "1",
    },
    harness.ctx,
  );
  assert.equal(editResult.block, true);

  const coderSubagent = await harness.handlers.get("tool_call")(
    {
      type: "tool_call",
      toolName: "subagent",
      input: { subagent_type: "Coder" },
      toolCallId: "2",
    },
    harness.ctx,
  );
  assert.equal(coderSubagent.block, true);

  const exploreSubagent = await harness.handlers.get("tool_call")(
    {
      type: "tool_call",
      toolName: "subagent",
      input: { subagent_type: "Explore" },
      toolCallId: "3",
    },
    harness.ctx,
  );
  assert.equal(exploreSubagent, undefined);
});

test("later /plan marker supersedes brainstorm without restoring tools", async () => {
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
  const toolResult = await harness.handlers.get("tool_call")(
    {
      type: "tool_call",
      toolName: "edit",
      input: { path: ".ergon.studio/scratchpad.md", edits: [] },
      toolCallId: "4",
    },
    harness.ctx,
  );

  assert.equal(promptResult, undefined);
  assert.equal(toolResult, undefined);
  assert.deepEqual(harness.activeTools, ["read", "edit", "write"]);
  assert.equal(harness.statuses.get("brainstorm"), undefined);
});
