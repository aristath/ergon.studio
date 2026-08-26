// Ergon user-asset self-install (preset + skills) into the DSH home.
//
// The ergon bundle patch (cordis.patch.yml) makes the ergon agent preset the
// DEFAULT for new sessions, so the preset must exist in the user preset root
// by the time the first session resolves that default. apply() calls
// ensureErgonAssets() synchronously at mount, which copies the package's
// bundled assets into the DSH home when they are not present yet.
//
// Policy:
//   - install ONLY when missing — an existing preset or skill is user-owned
//     (a person or a later `ergon-dsh init --force` may have authored or
//     refreshed it) and is never overwritten here;
//   - fail-open — any error (incomplete package, read-only home) degrades to
//     a single warning; the profile keeps working with the other presets.

import { existsSync, mkdirSync, readFileSync, cpSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
/** The package root (the directory containing dist/, presets/, skills/). */
const PACKAGE_ROOT = join(__dirname, "..");

export type Warn = (message: string) => void;

/** The DSH home for self-install: $DSH_HOME, else ~/.dsh (matches the CLI). */
export function dshHome(): string {
  return process.env.DSH_HOME ?? join(homedir(), ".dsh");
}

function sameFile(a: string, b: string): boolean {
  try {
    return readFileSync(a, "utf8") === readFileSync(b, "utf8");
  } catch {
    return false;
  }
}

/**
 * Copy the bundled ergon preset and skills into the DSH home when missing.
 * Never throws.
 */
export function ensureErgonAssets(home: string, warn: Warn): void {
  // ── preset ──────────────────────────────────────────────────────────────
  const presetSrc = join(PACKAGE_ROOT, "presets", "ergon");
  const presetDest = join(home, ".agent-presets", "ergon");
  const presetSrcFile = join(presetSrc, "agent.cordis.yml");
  const presetDestFile = join(presetDest, "agent.cordis.yml");
  const presetSrcMeta = join(presetSrc, "preset.yml");
  const presetDestMeta = join(presetDest, "preset.yml");
  try {
    if (!existsSync(presetSrcFile)) {
      warn(`ergon: preset not found in package at ${presetSrcFile}; skipping self-install`);
      return;
    }
    if (!existsSync(presetDestFile)) {
      mkdirSync(presetDest, { recursive: true });
      cpSync(presetSrc, presetDest, { recursive: true });
      warn(`ergon: installed agent preset at ${presetDest}`);
    } else if (!existsSync(presetDestMeta) && existsSync(presetSrcMeta)) {
      // Half-written install: the composition survived but the preset
      // metadata (name/description/order) is missing — repair just that file.
      // Still install-only-when-missing: a user-edited preset.yml is kept.
      cpSync(presetSrcMeta, presetDestMeta);
      warn(`ergon: repaired missing preset metadata at ${presetDestMeta}`);
    } else if (sameFile(presetSrcFile, presetDestFile)) {
      // up to date — nothing to do
    } else {
      warn(
        `ergon: installed preset at ${presetDestFile} differs from the package version; ` +
          `keeping the installed copy (run 'npx @ergon.studio/dsh init --force' to refresh)`,
      );
    }
  } catch (err) {
    warn(`ergon: preset self-install failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  // ── skills ──────────────────────────────────────────────────────────────
  const skillsSrc = join(PACKAGE_ROOT, "skills");
  const skillsDest = join(home, "skills");
  try {
    if (!existsSync(skillsSrc)) return;
    const entries = readdirSync(skillsSrc, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
    for (const entry of entries) {
      const from = join(skillsSrc, entry);
      const to = join(skillsDest, entry);
      const fromSkill = join(from, "SKILL.md");
      const toSkill = join(to, "SKILL.md");
      if (!existsSync(fromSkill)) continue;
      if (existsSync(toSkill) && sameFile(fromSkill, toSkill)) continue; // up to date
      if (existsSync(toSkill)) continue; // user-owned copy — never overwrite
      mkdirSync(skillsDest, { recursive: true });
      cpSync(from, to, { recursive: true });
      warn(`ergon: installed skill "${entry}" at ${to}`);
    }
  } catch (err) {
    warn(`ergon: skill self-install failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

