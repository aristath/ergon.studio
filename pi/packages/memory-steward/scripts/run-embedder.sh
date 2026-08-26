#!/usr/bin/env bash
# Launches the Granite embedding server for the memory steward.
#
# Reads embedder config from prompts/steward.md frontmatter,
# then runs llama-server in embedding mode under a health supervisor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPTS_FILE="$SCRIPT_DIR/../prompts/steward.md"

# --- Minimal YAML frontmatter parser ---
# Extracts key: value pairs between --- markers.

parse_value() {
	local key="$1"
	sed -n '/^---$/,/^---$/p' "$PROMPTS_FILE" |
		grep "^${key}:" |
		head -1 |
		sed "s/^${key}:[[:space:]]*//" |
		sed 's/[[:space:]]*#.*$//' |
		sed 's/^"\(.*\)"$/\1/' |
		sed "s/^'\(.*\)'$/\1/"
}

parse_url_port() {
	local url="$1"
	if [[ "$url" =~ ^https?://[^:/]+:([0-9]+) ]]; then
		printf '%s\n' "${BASH_REMATCH[1]}"
	fi
}

expand_home() {
	local path="$1"
	if [ "$path" = "~" ]; then
		printf '%s\n' "$HOME"
	elif [[ "$path" == "~/"* ]]; then
		printf '%s/%s\n' "$HOME" "${path#~/}"
	else
		printf '%s\n' "$path"
	fi
}

resolve_executable() {
	local candidate="$1"
	candidate="$(expand_home "$candidate")"
	if [ -x "$candidate" ]; then
		printf '%s\n' "$candidate"
		return 0
	fi
	command -v "$candidate"
}

# --- Read config ---

LLAMA_SERVER_BIN="$(resolve_executable "${ERGON_LLAMA_SERVER_BIN:-$(parse_value 'llama_server_bin')}")"
EMBEDDER_MODEL="$(expand_home "${ERGON_EMBEDDER_MODEL_PATH:-$(parse_value 'embedder_model_path')}")"
EMBEDDER_PORT="${ERGON_EMBEDDER_PORT:-$(parse_value 'embedder_port' || true)}"
if [ -z "$EMBEDDER_PORT" ]; then
	EMBEDDER_URL="${ERGON_EMBEDDER_URL:-$(parse_value 'embedder_url' || true)}"
	EMBEDDER_PORT="$(parse_url_port "$EMBEDDER_URL")"
fi
EMBEDDER_PORT="${EMBEDDER_PORT:-18092}"
EMBEDDER_MODEL_NAME="${ERGON_EMBEDDER_MODEL:-$(parse_value 'embedder_model' || echo 'granite-embedding-311m')}"
DEVICE="${ERGON_EMBEDDER_DEVICE:-$(parse_value 'embedder_device' || true)}"
N_GPU_LAYERS="${ERGON_EMBEDDER_N_GPU_LAYERS:-$(parse_value 'n_gpu_layers' || echo '99')}"
CTX_SIZE="${ERGON_EMBEDDER_CTX_SIZE:-8192}"
DIMENSIONS="${ERGON_EMBEDDER_DIMENSIONS:-$(parse_value 'embedder_dimensions' || echo '768')}"

# --- Validate ---

if [ -z "$EMBEDDER_MODEL" ]; then
	echo "ERROR: embedder_model_path not set in $PROMPTS_FILE" >&2
	exit 1
fi

if [ ! -f "$EMBEDDER_MODEL" ]; then
	echo "ERROR: embedder model not found at $EMBEDDER_MODEL" >&2
	exit 1
fi

# --- Launch ---

echo "[embedder] Starting Granite 311M embedding server on port $EMBEDDER_PORT"
echo "[embedder] Model: $EMBEDDER_MODEL"

DEVICE_ARGS=()
if [ -n "${DEVICE:-}" ]; then
	DEVICE_ARGS=(--device "$DEVICE")
fi

exec "$SCRIPT_DIR/llama-server-supervisor.sh" \
	--name embedder \
	--health embedder \
	--url "http://127.0.0.1:$EMBEDDER_PORT" \
	--model "$EMBEDDER_MODEL_NAME" \
	--dimensions "$DIMENSIONS" \
	-- \
	"$LLAMA_SERVER_BIN" \
	--host 127.0.0.1 \
	--port "$EMBEDDER_PORT" \
	--model "$EMBEDDER_MODEL" \
	--alias "$EMBEDDER_MODEL_NAME" \
	--ctx-size "$CTX_SIZE" \
	"${DEVICE_ARGS[@]}" \
	--n-gpu-layers "$N_GPU_LAYERS" \
	--embedding
