// Regenerates presets/ergon/ from agents/*.md after a TypeScript build.
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const result = spawnSync(
  process.execPath,
  [join(root, "dist", "cli.js"), "generate-preset", "--out", join(root, "presets", "ergon")],
  { stdio: "inherit" },
);
process.exit(result.status ?? 1);
