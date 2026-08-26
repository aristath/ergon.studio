// Scratchpad + handoff context rendering.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  scratchpadBlock,
  readScratchpad,
  readHandoff,
  NO_SCRATCHPAD,
  SCRATCHPAD_REL,
  HANDOFF_REL,
} from "../dist/scratchpad.js";

function tmpCwd() {
  return mkdtempSync(join(tmpdir(), "ergon-test-"));
}

test("scratchpadBlock: no scratchpad → NO_SCRATCHPAD template", () => {
  const cwd = tmpCwd();
  try {
    const block = scratchpadBlock(cwd);
    assert.ok(block.startsWith("## Project Scratchpad\n\n"));
    assert.ok(block.includes("No scratchpad yet for this project."));
    assert.ok(block.includes(NO_SCRATCHPAD));
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("scratchpadBlock: scratchpad content included, trimmed", () => {
  const cwd = tmpCwd();
  try {
    mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(join(cwd, SCRATCHPAD_REL), "\n## Conventions\n- prefer pnpm\n\n\n");
    const block = scratchpadBlock(cwd);
    assert.ok(block.includes("## Conventions"));
    assert.ok(block.includes("- prefer pnpm"));
    // trimmed: no leading blank line after the header
    assert.ok(block.startsWith("## Project Scratchpad\n\n## Conventions"));
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("scratchpadBlock: handoff appended with 'read this first' marker", () => {
  const cwd = tmpCwd();
  try {
    mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(join(cwd, SCRATCHPAD_REL), "## Notes\n- x\n");
    writeFileSync(join(cwd, HANDOFF_REL), "  Mid-task: fix the auth bug. \n");
    const block = scratchpadBlock(cwd);
    assert.ok(block.includes("## Handoff (read this first)"));
    assert.ok(block.includes("Mid-task: fix the auth bug."));
    assert.ok(!block.includes("  Mid-task"), "handoff should be trimmed");
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("scratchpadBlock: empty scratchpad treated as absent (handoff alone ignored)", () => {
  const cwd = tmpCwd();
  try {
    mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(join(cwd, SCRATCHPAD_REL), "   \n");
    writeFileSync(join(cwd, HANDOFF_REL), "handoff only\n");
    const block = scratchpadBlock(cwd);
    assert.ok(block.includes("No scratchpad yet"));
    assert.ok(!block.includes("handoff only"));
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("readScratchpad / readHandoff: null when missing, content when present", () => {
  const cwd = tmpCwd();
  try {
    assert.equal(readScratchpad(cwd), null);
    assert.equal(readHandoff(cwd), null);
    mkdirSync(join(cwd, ".ergon.studio"), { recursive: true });
    writeFileSync(join(cwd, SCRATCHPAD_REL), "pad");
    writeFileSync(join(cwd, HANDOFF_REL), "hand");
    assert.equal(readScratchpad(cwd), "pad");
    assert.equal(readHandoff(cwd), "hand");
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("rel path constants point at .ergon.studio", () => {
  assert.match(SCRATCHPAD_REL, /scratchpad\.md$/);
  assert.match(HANDOFF_REL, /HANDOFF\.md$/);
});
