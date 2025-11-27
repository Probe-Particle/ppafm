#!/usr/bin/env bash

# Usage: ./run_web_server.sh [PORT] [DOCROOT]
# Defaults: PORT=8000, DOCROOT=.
# - Kills any existing process listening on PORT
# - Then starts `python3 -m http.server PORT` in DOCROOT

set -euo pipefail

PORT="${1:-8000}"
DOCROOT="${2:-.}"

# Find any process listening on PORT (requires `lsof`)
PIDS="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"

if [[ -n "${PIDS}" ]]; then
  echo "Killing existing server(s) on port ${PORT}: ${PIDS}" >&2
  kill ${PIDS} 2>/dev/null || true
  sleep 0.5
fi

echo "Starting python3 -m http.server ${PORT} in ${DOCROOT}" >&2
cd "${DOCROOT}"
exec python3 -m http.server "${PORT}"
