#!/usr/bin/env bash
# Supervises a llama-server child by probing its HTTP contract.

set -euo pipefail

NAME="llama-server"
HEALTH="models"
URL=""
MODEL=""
DIMENSIONS=""

usage() {
	cat <<'EOF'
Usage: llama-server-supervisor.sh --name NAME --health KIND --url URL [options] -- COMMAND [ARGS...]

Health kinds:
  models     GET /v1/models must return HTTP 2xx
  embedder   POST /v1/embeddings must return the configured vector dimensions

Options:
  --model NAME
  --dimensions N

Environment:
  ERGON_LLAMA_HEALTH_INTERVAL   seconds between steady-state probes, default 30
  ERGON_LLAMA_START_PERIOD      seconds allowed for first healthy probe, default 120
  ERGON_LLAMA_HEALTH_FAILURES   consecutive failures before restart, default 2
  ERGON_LLAMA_STOP_TIMEOUT      seconds to wait before SIGKILL on restart, default 10
EOF
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--name) shift; NAME="${1:-}" ;;
		--health) shift; HEALTH="${1:-}" ;;
		--url) shift; URL="${1:-}" ;;
		--model) shift; MODEL="${1:-}" ;;
		--dimensions) shift; DIMENSIONS="${1:-}" ;;
		--help | -h) usage; exit 0 ;;
		--) shift; break ;;
		*) echo "llama-server-supervisor.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
	shift
done

[ "$#" -gt 0 ] || { echo "llama-server-supervisor.sh: missing command" >&2; exit 2; }
[ -n "$URL" ] || { echo "llama-server-supervisor.sh: missing --url" >&2; exit 2; }

INTERVAL="${ERGON_LLAMA_HEALTH_INTERVAL:-30}"
START_PERIOD="${ERGON_LLAMA_START_PERIOD:-120}"
FAILURE_LIMIT="${ERGON_LLAMA_HEALTH_FAILURES:-2}"
STOP_TIMEOUT="${ERGON_LLAMA_STOP_TIMEOUT:-10}"

check_health() {
	case "$HEALTH" in
		models)
			curl -fsS --max-time 5 "$URL/v1/models" >/dev/null
			;;
		embedder)
			[ -n "$MODEL" ] || return 1
			[ -n "$DIMENSIONS" ] || return 1
			local json
			json="$(
				curl -fsS --max-time 10 "$URL/v1/embeddings" \
					-H "Content-Type: application/json" \
					-d "{\"model\":\"$MODEL\",\"input\":\"doctor\"}"
			)" || return 1
			node -e '
				const data = JSON.parse(process.argv[1]);
				const expected = Number(process.argv[2]);
				const actual = data.data?.[0]?.embedding?.length ?? 0;
				process.exit(actual === expected ? 0 : 1);
			' "$json" "$DIMENSIONS"
			;;
		*)
			echo "llama-server-supervisor.sh: unknown health kind: $HEALTH" >&2
			return 1
			;;
	esac
}

"$@" &
child=$!

terminate() {
	if kill -0 "$child" 2>/dev/null; then
		kill "$child" 2>/dev/null || true
		local deadline=$((SECONDS + STOP_TIMEOUT))
		while kill -0 "$child" 2>/dev/null; do
			if [ "$SECONDS" -ge "$deadline" ]; then
				echo "[$NAME] child did not stop after ${STOP_TIMEOUT}s; killing" >&2
				kill -KILL "$child" 2>/dev/null || true
				break
			fi
			sleep 0.2
		done
		wait "$child" 2>/dev/null || true
	fi
}

trap 'terminate; exit 143' INT TERM

deadline=$((SECONDS + START_PERIOD))
while kill -0 "$child" 2>/dev/null; do
	if check_health; then
		echo "[$NAME] healthcheck passed"
		break
	fi
	if [ "$SECONDS" -ge "$deadline" ]; then
		echo "[$NAME] healthcheck did not pass within ${START_PERIOD}s; restarting" >&2
		terminate
		exit 1
	fi
	sleep 2
done

failures=0
while kill -0 "$child" 2>/dev/null; do
	sleep "$INTERVAL" || true
	if ! kill -0 "$child" 2>/dev/null; then
		break
	fi
	if check_health; then
		failures=0
	else
		failures=$((failures + 1))
		echo "[$NAME] healthcheck failed ($failures/$FAILURE_LIMIT)" >&2
		if [ "$failures" -ge "$FAILURE_LIMIT" ]; then
			echo "[$NAME] unhealthy; restarting" >&2
			terminate
			exit 1
		fi
	fi
done

wait "$child"
