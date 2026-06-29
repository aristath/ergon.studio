import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import planExtension, {
  deriveStateFromBranch,
  getPlanTools,
  SCOUT_PLAN_PROMPT,
  stripAgentFrontmatter,
} from "../dist/extensions/index.js";

const legacyScoutPrompt = readFileSync(
  new URL("../../../../agents/scout.md", import.meta.url),
  "utf8",
);

function tempProject() {
  const cwd = mkdtempSync(join(tmpdir(), "pi-plan-"));
  return {
    cwd,
    cleanup() {
      rmSync(cwd, { recursive: true, force: true });
    },
  };
}

function makeHarness() {
  const commands = new Map();
  const handlers = new Map();
  const branch = [];
  let activeTools = ["read", "find", "grep", "bash", "edit", "write"];
  const notifications = [];
  const widgets = new Map();
  const statuses = new Map();
  const confirms = [];
  const editors = [];
  let hasEditorOverride = false;
  let editorResult;

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
    hasUI: true,
    cwd: tempProject().cwd,
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
      async select(_title, options) {
        return options[0];
      },
      async editor(_title, prefill) {
        editors.push({ title: _title, prefill });
        if (hasEditorOverride) {
          return editorResult;
        }
        return `${prefill}\nFinal plan text.\n`;
      },
      async confirm(title, message) {
        confirms.push({ title, message });
        return true;
      },
    },
  };

  planExtension(pi);

  return {
    pi,
    ctx,
    commands,
    handlers,
    branch,
    notifications,
    confirms,
    editors,
    statuses,
    widgets,
    get activeTools() {
      return activeTools;
    },
    setEditorResult(value) {
      hasEditorOverride = true;
      editorResult = value;
    },
    cleanup() {
      rmSync(ctx.cwd, { recursive: true, force: true });
    },
  };
}

test("derives active and inactive state from session markers", () => {
  const state = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-plan-state",
      id: "1",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: {
        action: "start",
        topic: "agent packages",
        startedAt: 1,
        previousTools: ["read", "bash"],
      },
    },
  ]);

  assert.equal(state.active, true);
  assert.equal(state.topic, "agent packages");
  assert.deepEqual(state.previousTools, ["read", "bash"]);

  const inactive = deriveStateFromBranch([
    {
      type: "custom",
      customType: "ergon-plan-state",
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
      data: { action: "cancel" },
    },
  ]);

  assert.equal(inactive.active, false);
});

test("detects available planning tools", () => {
  const tools = getPlanTools({
    getAllTools() {
      return ["read", "grep", "write", "subagent"].map((name) => ({ name }));
    },
  });

  assert.deepEqual(tools, ["read", "grep", "subagent", "write"]);
});

test("registers /plan and starts read-only plan mode", async () => {
  const harness = makeHarness();
  try {
    assert.equal(typeof harness.commands.get("plan").handler, "function");

    await harness.commands
      .get("plan")
      .handler("Pi package architecture", harness.ctx);

    assert.deepEqual(harness.activeTools, [
      "read",
      "find",
      "grep",
      "ls",
      "ask_user_question",
      "subagent",
      "get_subagent_result",
      "edit",
      "write",
    ]);
    assert.equal(harness.branch[0].customType, "ergon-plan-state");
    assert.equal(harness.branch[0].data.action, "start");
    assert.equal(harness.branch[0].data.topic, "Pi package architecture");
    assert.match(harness.statuses.get("plan"), /plan/);
  } finally {
    harness.cleanup();
  }
});

test("injects the Scout planning workflow into the system prompt", async () => {
  const harness = makeHarness();
  try {
    await harness.commands.get("plan").handler("memory steward", harness.ctx);

    const result = await harness.handlers.get("before_agent_start")(
      {
        systemPrompt: "Base system prompt",
      },
      harness.ctx,
    );

    assert.match(result.systemPrompt, /Base system prompt/);
    assert.match(result.systemPrompt, /Optimal Solution/);
    assert.match(result.systemPrompt, /Assume You're Wrong/);
    assert.match(result.systemPrompt, /Current planning topic: memory steward/);
  } finally {
    harness.cleanup();
  }
});

test("preserves the legacy Scout wording exactly", () => {
  const legacyBody = stripAgentFrontmatter(legacyScoutPrompt);

  assert.ok(SCOUT_PLAN_PROMPT.startsWith(legacyBody));
  assert.match(SCOUT_PLAN_PROMPT, /## Pi \/plan Mode Boundary/);
  assert.match(SCOUT_PLAN_PROMPT, /Because the user invoked \/plan/);
});

test("blocks implementation tools and only permits Explore subagents", async () => {
  const harness = makeHarness();
  try {
    await harness.commands.get("plan").handler("", harness.ctx);

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
  } finally {
    harness.cleanup();
  }
});

test("allows write and edit only for the Scout scratchpad", async () => {
  const harness = makeHarness();
  try {
    await harness.commands.get("plan").handler("", harness.ctx);

    const scratchpadWrite = await harness.handlers.get("tool_call")(
      {
        type: "tool_call",
        toolName: "write",
        input: {
          path: ".ergon.studio/scratchpad.md",
          content: "## Decisions\n\n- Keep the plan lean.",
        },
        toolCallId: "1",
      },
      harness.ctx,
    );
    assert.equal(scratchpadWrite, undefined);

    const outsideWrite = await harness.handlers.get("tool_call")(
      {
        type: "tool_call",
        toolName: "write",
        input: { path: "HANDOFF.md", content: "nope" },
        toolCallId: "2",
      },
      harness.ctx,
    );
    assert.equal(outsideWrite.block, true);
  } finally {
    harness.cleanup();
  }
});

test("finishes by writing HANDOFF and restoring previous tools", async () => {
  const harness = makeHarness();
  try {
    await harness.commands.get("plan").handler("handoff test", harness.ctx);
    harness.branch.push({
      type: "message",
      id: "assistant-plan",
      parentId: harness.branch.at(-1)?.id ?? null,
      timestamp: new Date().toISOString(),
      message: {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "# Handoff: generated plan\n\n## Plan\n\n1. Build the right thing.",
          },
        ],
        timestamp: Date.now(),
      },
    });
    await harness.commands.get("plan").handler("finish", harness.ctx);

    const handoffPath = join(harness.ctx.cwd, ".ergon.studio", "HANDOFF.md");
    assert.equal(existsSync(handoffPath), true);
    assert.match(readFileSync(handoffPath, "utf8"), /Handoff: generated plan/);
    assert.match(readFileSync(handoffPath, "utf8"), /Final plan text/);
    assert.deepEqual(harness.activeTools, [
      "read",
      "find",
      "grep",
      "bash",
      "edit",
      "write",
    ]);
    assert.equal(harness.branch.at(-1).data.action, "finish");
    assert.equal(
      harness.branch.at(-1).data.handoffPath,
      ".ergon.studio/HANDOFF.md",
    );
  } finally {
    harness.cleanup();
  }
});

test("injects existing HANDOFF into context and confirms after review before overwrite", async () => {
  const harness = makeHarness();
  try {
    mkdirSync(join(harness.ctx.cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(
      join(harness.ctx.cwd, ".ergon.studio", "HANDOFF.md"),
      "# Existing handoff\n",
    );

    await harness.commands.get("plan").handler("existing handoff", harness.ctx);
    const result = await harness.handlers.get("before_agent_start")(
      { systemPrompt: "Base system prompt" },
      harness.ctx,
    );
    assert.match(result.systemPrompt, /Existing \.ergon\.studio\/HANDOFF\.md/);
    assert.match(result.systemPrompt, /# Existing handoff/);

    await harness.commands.get("plan").handler("finish", harness.ctx);
    assert.equal(harness.editors.length, 1);
    assert.equal(harness.confirms.length, 1);
    assert.match(harness.confirms[0].message, /already exists/);
  } finally {
    harness.cleanup();
  }
});

test("does not confirm overwrite when handoff review is cancelled", async () => {
  const harness = makeHarness();
  try {
    mkdirSync(join(harness.ctx.cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(
      join(harness.ctx.cwd, ".ergon.studio", "HANDOFF.md"),
      "# Existing handoff\n",
    );
    harness.setEditorResult(undefined);

    await harness.commands.get("plan").handler("existing handoff", harness.ctx);
    await harness.commands.get("plan").handler("finish", harness.ctx);

    assert.equal(harness.editors.length, 1);
    assert.equal(harness.confirms.length, 0);
    assert.equal(
      readFileSync(
        join(harness.ctx.cwd, ".ergon.studio", "HANDOFF.md"),
        "utf8",
      ),
      "# Existing handoff\n",
    );
  } finally {
    harness.cleanup();
  }
});
