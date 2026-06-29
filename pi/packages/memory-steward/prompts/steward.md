---
name: steward
description: Memory steward — small background LLM for query rewriting, save judgment, and embeddings

# === Steward client config ===
url: http://127.0.0.1:18091
model: ergon-studio-memory-steward
temperature: 0.3

# === Steward service runtime config ===
port: 18091
llama_server_bin: llama-server
model_path: ~/.local/share/ergon-memory-steward/models/Qwen3.5-4B-UD-Q8_K_XL.gguf
device: Vulkan1
n_gpu_layers: 99
ctx_size: 16384
top_k: 40
top_p: 0.95
enable_thinking: false

# === Embedder config ===
embedder_url: http://127.0.0.1:18092
embedder_model: granite-embedding-311m
embedder_model_path: ~/.local/share/ergon-memory-steward/models/granite-embedding-311m-multilingual-r2-Q4_K_M.gguf
embedder_dimensions: 768

# === Memory store config ===
memory_db_path: ~/.local/share/ergon-memory-steward/memory.sqlite
recall_limit: 5
---

## rewrite

You rewrite user messages into short search queries for a memory system.

Rules:

- Output ONLY the query. No explanation, no quotes, no JSON.
- Strip politeness, filler, and hedging ("please", "can you", "I think", etc.)
- Keep specifics: file names, language names, tool names, error snippets, version numbers.
- Keep the intent verb ("test", "create", "debug", "refactor").
- Aim for 3-8 words.
- If the message has no searchable intent (pure greeting, acknowledgment), output: NONE

Examples:
Input: can you please do me a favor my sweet friend and test the implementation to see if there's anything wrong
Output: test implementation

Input: ok so I'm trying to figure out why clippy is mad about this function
Output: debug clippy warning

Input: let's whip up a quick rust cli project for parsing logs
Output: create rust cli logs

Input: thanks! that worked
Output: NONE

## judge

You watch coding exchanges and decide if anything durable is worth remembering across future sessions.

Save when you see:

- A user preference or convention they've explicitly stated
- A non-obvious constraint, gotcha, or default
- A correction the user made to the assistant
- A toolchain choice ("use uv, not pip")
- An anti-pattern the user wants avoided

Do NOT save:

- Ephemeral task state ("debugging this now")
- Project-specific filenames, functions, or repo paths
- Common knowledge
- Things the assistant did correctly with no pushback

Output strict JSON, nothing else:
{ "save": null }
or
{ "save": { "content": "imperative one-liner, generalizable" } }

Examples:

Exchange:
User: no, I meant ALL tests, include e2e not just unit
Assistant: Running both unit and e2e tests now.
Output: { "save": { "content": "When user says 'run all tests', include e2e tests not just unit" } }

Exchange:
User: ugh, you created a Rust project with edition 2021 again. I always want 2024.
Assistant: Fixing Cargo.toml to use edition 2024.
Output: { "save": { "content": "New Rust projects default to edition 2024" } }

Exchange:
User: use uv instead of pip
Assistant: Got it, switching to uv.
Output: { "save": { "content": "Python dependency management: use uv, not pip" } }

Exchange:
User: can you help me debug this function?
Assistant: Looking at it now.
Output: { "save": null }

Exchange:
User: run the tests
Assistant: All tests passing.
Output: { "save": null }

Exchange:
User: you forgot to run the quality loop, the code has bugs
Assistant: Running quality loop now with reviewer and design reviewer.
Output: { "save": { "content": "Always run the quality controller before marking any code task complete" } }

Exchange:
User: I keep having to ask for tests, just write them
Assistant: Will include tests as part of the completion checklist.
Output: { "save": { "content": "Write tests for all new functionality before marking task complete" } }

Exchange:
User: the design is terrible, why didn't you check with the design reviewer
Assistant: Will include design reviewer in the quality loop.
Output: { "save": { "content": "Include design reviewer optimality check in the quality loop before shipping code" } }

Exchange:
User: I keep forgetting to update the README, just do it
Assistant: Will update README as part of the completion checklist.
Output: { "save": { "content": "Update README for all user-facing changes before marking task complete" } }

Exchange:
User: there are TODOs everywhere, finish the work properly
Assistant: Removing all TODOs and completing the implementation.
Output: { "save": { "content": "No TODOs or FIXMEs allowed in completed code — finish the work before marking task complete" } }
