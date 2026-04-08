#!/bin/bash
#
# Memory steward runner.
#
# Parses the YAML frontmatter of prompts/steward.md and execs llama-server
# with the right arguments. This keeps the service runtime config in the
# same place as the steward client config — edit prompts/steward.md and
# restart llama-steward.service to pick up changes.
#
# Meant to be run by systemd (see prompts/steward.service) but can also be
# run directly for manual testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEWARD_MD="$SCRIPT_DIR/../prompts/steward.md"

if [ ! -f "$STEWARD_MD" ]; then
  echo "run-steward.sh: cannot find $STEWARD_MD" >&2
  exit 1
fi

# Extract the first YAML frontmatter block (between the first two `---` lines).
FM=$(awk '/^---[[:space:]]*$/{c++; if(c==2) exit; next} c==1' "$STEWARD_MD")

# Simple `key: value` getter. Strips leading/trailing whitespace. Ignores
# commented (#) lines and blank lines.
fm_get() {
  local key="$1"
  echo "$FM" | awk -F': *' -v k="$key" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    $1==k { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }
  '
}

PORT="$(fm_get port)"
MODEL="$(fm_get model)"
MODEL_PATH="$(fm_get model_path)"
LLAMA_SERVER_BIN="$(fm_get llama_server_bin)"
DEVICE="$(fm_get device)"
N_GPU_LAYERS="$(fm_get n_gpu_layers)"
CTX_SIZE="$(fm_get ctx_size)"
TEMPERATURE="$(fm_get temperature)"
TOP_K="$(fm_get top_k)"
TOP_P="$(fm_get top_p)"
ENABLE_THINKING="$(fm_get enable_thinking)"

# Validate the essentials.
for var in LLAMA_SERVER_BIN MODEL_PATH DEVICE PORT; do
  if [ -z "${!var:-}" ]; then
    echo "run-steward.sh: missing required frontmatter key: ${var,,}" >&2
    exit 1
  fi
done

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "run-steward.sh: llama_server_bin not executable: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "run-steward.sh: model_path not found: $MODEL_PATH" >&2
  exit 1
fi

# Defaults for optional values.
: "${N_GPU_LAYERS:=99}"
: "${CTX_SIZE:=16384}"
: "${TEMPERATURE:=0.3}"
: "${TOP_K:=40}"
: "${TOP_P:=0.95}"
: "${MODEL:=ergon-studio-memory-steward}"
: "${ENABLE_THINKING:=false}"

# Some small instruct-tuned models (Qwen 3.5 in particular) default to
# chain-of-thought output. The steward's job is classification, not
# reasoning — we want it to go straight to the answer. `--reasoning-format
# none` only controls formatting of think blocks, not whether the model
# produces them; `--chat-template-kwargs '{"enable_thinking":false}'`
# actually disables the thinking phase in the chat template.
CHAT_TEMPLATE_KWARGS="{\"enable_thinking\":${ENABLE_THINKING}}"

exec "$LLAMA_SERVER_BIN" \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL_PATH" \
  --alias "$MODEL" \
  --device "$DEVICE" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --ctx-size "$CTX_SIZE" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn true --jinja \
  --chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
  --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" \
  --reasoning-format none
