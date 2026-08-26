// @ergon.studio/dsh — Ergon for DeepSeek Harness.
//
// Cordis plugin entry (static package plugin): named exports `name`,
// `inject`, `Config`, `apply` — the shape the DSH preset loader imports for
// row `name: '@ergon.studio/dsh'`.

export { name, inject, Config, apply } from "./plugin.js";

// Re-exported for tests and advanced use.
export { createStewardClient, parseStewardMd } from "./steward.js";
export { createMemoryClient } from "./memory.js";
export { loadRoster, getRosterEntry, denyListFor } from "./roster.js";
export {
  parseDebateVerdict,
  debatePrompt,
  renderDebateTranscript,
  outputText,
} from "./debate.js";
export { generatePreset, writePreset, validatePresetFile } from "./preset-gen.js";
