#!/usr/bin/env node
// ergon-dsh — installer CLI for @ergon.studio/dsh.
//
//   npx @ergon.studio/dsh init [--profile <name>] [--force]
//
// Installs the ergon preset and skills into the DSH home and checks that the
// plugin package is present in the target profile's node_modules.

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, cpSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { generatePreset, writePreset, validatePresetFile } from "./preset-gen.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PACKAGE_ROOT = join(__dirname, "..");

function dshHome(): string {
  return process.env.DSH_HOME ?? join(homedir(), ".dsh");
}

function presetDir(home: string): string {
  return join(home, ".agent-presets", "ergon");
}

function skillsDir(home: string): string {
  return join(home, "skills");
}

function profileNodeModules(home: string, profile: string): string {
  return join(home, "profiles", profile, "node_modules");
}

function sameContent(a: string, b: string): boolean {
  return readFileSync(a, "utf8") === readFileSync(b, "utf8");
}

function ensureSkills(force: boolean): { written: string[]; upToDate: string[]; kept: string[] } {
  const src = join(PACKAGE_ROOT, "skills");
  const dest = skillsDir(dshHome());
  const written: string[] = [];
  const upToDate: string[] = [];
  const kept: string[] = [];
  if (!existsSync(src)) {
    throw new Error(`skills directory not found at ${src} — incomplete package install`);
  }
  for (const skill of readdirSync(src, { withFileTypes: true }).filter((d) => d.isDirectory())) {
    const from = join(src, skill.name);
    const to = join(dest, skill.name);
    const fromSkill = join(from, "SKILL.md");
    const toSkill = join(to, "SKILL.md");
    if (existsSync(toSkill) && sameContent(fromSkill, toSkill)) {
      upToDate.push(skill.name);
      continue;
    }
    // Drifted installed copy: keep the user's version (same policy as the
    // plugin's preset self-install) and report it — never hard-fail init.
    if (existsSync(to) && !force) {
      kept.push(skill.name);
      continue;
    }
    if (existsSync(to)) rmSync(to, { recursive: true, force: true });
    mkdirSync(to, { recursive: true });
    cpSync(from, to, { recursive: true });
    written.push(skill.name);
  }
  return { written, upToDate, kept };
}

function checkProfilePlugin(profile: string): boolean {
  const nm = profileNodeModules(dshHome(), profile);
  const pkg = join(nm, "@ergon.studio", "dsh");
  return existsSync(join(pkg, "package.json"));
}

function init(profile: string, force: boolean): number {
  const home = dshHome();
  if (!existsSync(home)) {
    console.error(`DSH home not found at ${home}. Is DeepSeek Harness installed?`);
    return 1;
  }

  // 1. preset
  const dir = presetDir(home);
  const preset = generatePreset();
  const existing = existsSync(join(dir, "agent.cordis.yml"));
  if (existing && !force) {
    const problems = validatePresetFile(join(dir, "agent.cordis.yml"));
    if (problems.length === 0) {
      console.log(`• preset already installed at ${dir} (validated OK) — use --force to rewrite`);
    } else {
      console.log(`• existing preset at ${dir} is invalid (${problems.join("; ")}) — use --force to rewrite`);
      return 1;
    }
  } else {
    writePreset(dir, preset);
    console.log(`• preset written to ${dir}`);
  }
  const problems = validatePresetFile(join(dir, "agent.cordis.yml"));
  if (problems.length > 0) {
    for (const p of problems) console.error(`  ✗ ${p}`);
    return 1;
  }

  // 2. skills
  const { written, upToDate, kept } = ensureSkills(force);
  for (const s of written) console.log(`• skill installed: ${s}`);
  for (const s of upToDate) console.log(`• skill up to date: ${s}`);
  for (const s of kept) {
    console.log(
      `• skill differs from the package copy — keeping your version: ${s} (refresh with --force)`,
    );
  }

  // 3. profile plugin check
  if (checkProfilePlugin(profile)) {
    console.log(`• plugin found in profile "${profile}" node_modules`);
  } else {
    console.log(
      `• plugin NOT installed in profile "${profile}" — run:\n` +
        `    dsh plugin --profile ${profile} add @ergon.studio/dsh`,
    );
  }

  console.log("");
  console.log("Done. Start a new session and pick the Ergon preset from the session picker");
  console.log("(or set it as the default in Settings). Live processes pick up preset");
  console.log("edits for NEW sessions automatically.");
  return 0;
}

function status(): number {
  const home = dshHome();
  console.log(`DSH home: ${home}`);
  const dir = presetDir(home);
  if (existsSync(join(dir, "agent.cordis.yml"))) {
    const problems = validatePresetFile(join(dir, "agent.cordis.yml"));
    console.log(`preset:    installed at ${dir}${problems.length ? ` (INVALID: ${problems.join("; ")})` : ""}`);
  } else {
    console.log("preset:    not installed");
  }
  const sd = skillsDir(home);
  const installed = existsSync(sd) ? readdirSync(sd) : [];
  console.log(`skills:    ${installed.length ? installed.join(", ") : "none"}`);
  for (const profile of existsSync(join(home, "profiles")) ? readdirSync(join(home, "profiles")) : []) {
    // `node_modules` is a pnpm virtual-store byproduct of `dsh plugin add`,
    // not a profile — don't report it as "plugin missing".
    if (profile === "node_modules") continue;
    if (!statSync(join(home, "profiles", profile)).isDirectory()) continue;
    console.log(`profile ${profile}: plugin ${checkProfilePlugin(profile) ? "installed" : "missing"}`);
  }
  return 0;
}

function main(argv: string[]): number {
  const [cmd, ...rest] = argv;
  const flag = (name: string) => rest.includes(name);
  const flagValue = (name: string, fallback: string): string => {
    const i = rest.indexOf(name);
    return i >= 0 && i + 1 < rest.length ? rest[i + 1] : fallback;
  };

  switch (cmd) {
    case "init":
      return init(flagValue("--profile", "web"), flag("--force"));
    case "status":
      return status();
    case "generate-preset": {
      // Internal command used by `npm run build` to regenerate the checked-in
      // presets/ergon/ files from agents/*.md.
      const out = flagValue("--out", join(PACKAGE_ROOT, "presets", "ergon"));
      writePreset(out);
      console.log(`preset generated at ${out}`);
      return 0;
    }
    default:
      console.log("ergon-dsh — installer for the ergon DeepSeek Harness preset\n");
      console.log("Usage:");
      console.log("  ergon-dsh init [--profile <name>] [--force]  Install preset + skills");
      console.log("  ergon-dsh status                             Show install state");
      console.log("  ergon-dsh generate-preset --out <dir>        Regenerate preset files");
      return cmd ? 1 : 0;
  }
}

process.exit(main(process.argv.slice(2)));
