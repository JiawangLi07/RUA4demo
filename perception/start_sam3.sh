#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYRUA_SAM3_PYTHON:-"$root/../dexagent-eval/.venv-sam3/bin/python"}
checkpoint=${SAM3_CHECKPOINT_PATH:-"$root/../dexagent-eval/models/sam3/sam3.pt"}
test -x "$python" && test -r "$checkpoint" || { echo "Missing SAM3 runtime/checkpoint" >&2; exit 1; }
exec "$python" "$root/upstream/src/pyrualean/hosts/sam3_http.py" \
    --host 127.0.0.1 --port "${PYRUA_SAM3_PORT:-8790}" --checkpoint "$checkpoint"
