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
  const toolApiCalls = [];
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
    toolApiCalls,
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
  assert.equal("previousTools" in state, false);

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

test("registers /plan and starts without changing tools", async () => {
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
      "bash",
      "edit",
      "write",
    ]);
    assert.deepEqual(harness.toolApiCalls, []);
    assert.equal(harness.handlers.has("tool_call"), false);
    assert.equal(harness.branch[0].customType, "ergon-plan-state");
    assert.equal(harness.branch[0].data.action, "start");
    assert.equal(harness.branch[0].data.topic, "Pi package architecture");
    assert.equal("previousTools" in harness.branch[0].data, false);
    assert.match(harness.statuses.get("plan"), /plan/);
  } finally {
    harness.cleanup();
  }
});

test("restores plan state without changing tools", async () => {
  const harness = makeHarness();
  try {
    harness.branch.push({
      type: "custom",
      customType: "ergon-plan-state",
      id: "plan",
      parentId: null,
      timestamp: new Date().toISOString(),
      data: {
        action: "start",
        topic: "restored plan",
        previousTools: ["read"],
      },
    });

    await harness.handlers.get("session_start")({}, harness.ctx);

    assert.match(harness.statuses.get("plan"), /restored plan/);
    assert.deepEqual(harness.activeTools, [
      "read",
      "find",
      "grep",
      "bash",
      "edit",
      "write",
    ]);
    assert.deepEqual(harness.toolApiCalls, []);
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
    assert.match(result.systemPrompt, /active tool selection remains unchanged/);
    assert.deepEqual(harness.toolApiCalls, []);
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

test("does not install a mode-level tool-call blocker", async () => {
  const harness = makeHarness();
  try {
    await harness.commands.get("plan").handler("", harness.ctx);

    assert.equal(harness.handlers.has("tool_call"), false);
    assert.deepEqual(harness.toolApiCalls, []);
  } finally {
    harness.cleanup();
  }
});

test("finishes by writing HANDOFF without changing tools", async () => {
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
    assert.deepEqual(harness.toolApiCalls, []);
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
