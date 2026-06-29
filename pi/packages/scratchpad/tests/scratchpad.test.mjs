import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import scratchpadExtension from "../dist/extensions/index.js";

function registerExtension() {
  const handlers = new Map();
  scratchpadExtension({
    on(event, handler) {
      handlers.set(event, handler);
    },
  });
  return handlers;
}

function tempProject() {
  const cwd = mkdtempSync(join(tmpdir(), "pi-scratchpad-"));
  return {
    cwd,
    writeScratchpad(content) {
      mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
      writeFileSync(join(cwd, ".ergon.studio", "scratchpad.md"), content);
    },
    cleanup() {
      rmSync(cwd, { recursive: true, force: true });
    },
  };
}

test("registers a before_agent_start handler", () => {
  const handlers = registerExtension();

  assert.equal(typeof handlers.get("before_agent_start"), "function");
});

test("injects project scratchpad when the file exists", async () => {
  const project = tempProject();
  try {
    project.writeScratchpad("## Decisions\n\n- Keep Pi ports on 18090+.");
    const handler = registerExtension().get("before_agent_start");

    const result = await handler({
      systemPrompt: "Base prompt",
      systemPromptOptions: { cwd: project.cwd },
    });

    assert.deepEqual(result, {
      systemPrompt:
        "Base prompt\n\n## Project Scratchpad\n\n## Decisions\n\n- Keep Pi ports on 18090+.",
    });
  } finally {
    project.cleanup();
  }
});

test("falls back to extension context cwd", async () => {
  const project = tempProject();
  try {
    project.writeScratchpad("## Notes\n\n- Context cwd works.");
    const handler = registerExtension().get("before_agent_start");

    const result = await handler(
      {
        systemPrompt: "Base prompt",
      },
      { cwd: project.cwd },
    );

    assert.equal(
      result.systemPrompt,
      "Base prompt\n\n## Project Scratchpad\n\n## Notes\n\n- Context cwd works.",
    );
  } finally {
    project.cleanup();
  }
});

test("stays quiet when the scratchpad file is missing", async () => {
  const project = tempProject();
  try {
    const handler = registerExtension().get("before_agent_start");

    const result = await handler({
      systemPrompt: "Base prompt",
      systemPromptOptions: { cwd: project.cwd },
    });

    assert.equal(result, undefined);
  } finally {
    project.cleanup();
  }
});

test("packages the Pi skill and manifest together", () => {
  const skill = readFileSync(
    new URL("../skills/scratchpad/SKILL.md", import.meta.url),
    "utf8",
  );
  const pkg = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );

  assert.match(skill, /^name: scratchpad$/m);
  assert.match(skill, /\.ergon\.studio\/scratchpad\.md/);
  assert.match(
    skill,
    /Create the file only when you have something worth writing/,
  );
  assert.deepEqual(pkg.pi.skills, ["./skills"]);
  assert.ok(pkg.files.includes("skills/"));
});
