#!/usr/bin/env bash
# Launches the dedicated memory steward llama-server.
#
# Reads steward runtime config from prompts/steward.md frontmatter,
# then runs llama-server on the configured reserved port under a health supervisor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEWARD_MD="$SCRIPT_DIR/../prompts/steward.md"

if [ ! -f "$STEWARD_MD" ]; then
	echo "run-steward.sh: cannot find $STEWARD_MD" >&2
	exit 1
fi

FM="$(awk '/^---[[:space:]]*$/{c++; if(c==2) exit; next} c==1' "$STEWARD_MD")"

fm_get() {
	local key="$1"
	echo "$FM" | awk -F': *' -v k="$key" '
		/^[[:space:]]*#/ { next }
		/^[[:space:]]*$/ { next }
		$1==k { sub(/^[^:]*:[[:space:]]*/, ""); print; exit }
	'
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

PORT="${ERGON_STEWARD_PORT:-$(fm_get port)}"
MODEL="${ERGON_STEWARD_MODEL:-$(fm_get model)}"
MODEL_PATH="$(expand_home "${ERGON_STEWARD_MODEL_PATH:-$(fm_get model_path)}")"
LLAMA_SERVER_BIN="$(resolve_executable "${ERGON_LLAMA_SERVER_BIN:-$(fm_get llama_server_bin)}")"
DEVICE="${ERGON_STEWARD_DEVICE:-$(fm_get device)}"
N_GPU_LAYERS="${ERGON_STEWARD_N_GPU_LAYERS:-$(fm_get n_gpu_layers)}"
CTX_SIZE="${ERGON_STEWARD_CTX_SIZE:-$(fm_get ctx_size)}"
TEMPERATURE="${ERGON_STEWARD_TEMPERATURE:-$(fm_get temperature)}"
TOP_K="${ERGON_STEWARD_TOP_K:-$(fm_get top_k)}"
TOP_P="${ERGON_STEWARD_TOP_P:-$(fm_get top_p)}"
ENABLE_THINKING="${ERGON_STEWARD_ENABLE_THINKING:-$(fm_get enable_thinking)}"

for var in LLAMA_SERVER_BIN MODEL_PATH DEVICE PORT; do
	if [ -z "${!var:-}" ]; then
		echo "run-steward.sh: missing required frontmatter key: ${var,,}" >&2
		exit 1
	fi
done

if [ ! -f "$MODEL_PATH" ]; then
	echo "run-steward.sh: model_path not found: $MODEL_PATH" >&2
	exit 1
fi

: "${N_GPU_LAYERS:=99}"
: "${CTX_SIZE:=16384}"
: "${TEMPERATURE:=0.3}"
: "${TOP_K:=40}"
: "${TOP_P:=0.95}"
: "${MODEL:=ergon-studio-memory-steward}"
: "${ENABLE_THINKING:=false}"

case "${ENABLE_THINKING,,}" in
	true | on) REASONING_MODE=on ;;
	false | off) REASONING_MODE=off ;;
	auto) REASONING_MODE=auto ;;
	*)
		echo "run-steward.sh: enable_thinking must be true, false, on, off, or auto" >&2
		exit 1
		;;
esac

exec "$SCRIPT_DIR/llama-server-supervisor.sh" \
	--name steward \
	--health models \
	--url "http://127.0.0.1:$PORT" \
	-- \
	"$LLAMA_SERVER_BIN" \
	--host 127.0.0.1 --port "$PORT" \
	--model "$MODEL_PATH" \
	--alias "$MODEL" \
	--device "$DEVICE" \
	--n-gpu-layers "$N_GPU_LAYERS" \
	--ctx-size "$CTX_SIZE" \
	--cache-type-k q8_0 --cache-type-v q8_0 \
	--flash-attn true --jinja \
	--reasoning "$REASONING_MODE" \
	--temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" \
	--reasoning-format none
