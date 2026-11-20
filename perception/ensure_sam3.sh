#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
port=${PYRUA_SAM3_PORT:-8790}
url="http://127.0.0.1:$port/health"
state_dir="$root/.local/sam3"
mkdir -p "$state_dir"
exec 9>"$state_dir/$port.lock"
flock 9
healthy() {
    curl --noproxy '*' --max-time 2 -fsS "$url" 2>/dev/null |
        python3 -c 'import json,sys; sys.exit(json.load(sys.stdin).get("ok") is not True)' 2>/dev/null
}
if healthy; then
    echo "SAM3 ready at $url"
    exit 0
fi
log="$state_dir/$port.log"
nohup env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$root/perception/start_sam3.sh" </dev/null >>"$log" 2>&1 9>&- &
pid=$!
for ((i=0; i<60; i++)); do
    if healthy; then
        echo "SAM3 ready at $url"
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
echo "SAM3 startup failed or still loading. See $log" >&2
exit 1
