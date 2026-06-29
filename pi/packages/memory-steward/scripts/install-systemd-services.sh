#!/usr/bin/env bash
# Compatibility wrapper for older docs/scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/ergon-memory-steward" setup "$@"
