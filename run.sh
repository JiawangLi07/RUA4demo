#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
deployment=$(cd -- "${DEXHUMANID_RUA_DEPLOYMENT:-"$root/../real_experiment/deployment"}" && pwd)
export DEXHUMANID_RUA_DEPLOYMENT="$deployment"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$deployment/src:$deployment/vendor/bridge:$deployment/deps/infra:$deployment/deps/observer:$deployment/vendor/universal_robot_infra/src:$deployment/assets/robot:$deployment/vendor/mujoco_marvin:$deployment/sdk/sharpa/python:$root/upstream/src:$root/.venv/lib/python3.10/site-packages"
export MARVIN_SDK_DIR="$deployment/sdk/tianji"
export SHARPA_SDK_PY="$deployment/sdk/sharpa/python"
if [[ "${1:-}" == reset-shuttle ]]; then
    shift
    export PYTHONPATH="$root/upstream-rsijudge/src:$root:$PYTHONPATH"
    exec "$deployment/envs/robot/bin/python" -m exploration.reset_shuttle "$@"
fi
if [[ "${1:-}" == task || "${1:-}" == task-server ]]; then
    subcommand=$1
    shift
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    export PYTHONPATH="$root/upstream-rsijudge/src:$root:$PYTHONPATH"
    if [[ "$subcommand" == task ]]; then
        exec "$deployment/envs/robot/bin/python" "$root/task.py" --local "$@"
    fi
    exec "$deployment/envs/robot/bin/python" "$root/task_server.py" "$@"
fi
if [[ "${1:-}" == calibrate-head ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/calibrate_head.py" "$@"
fi
if [[ "${1:-}" == explore ]]; then
    shift
    if [[ ! -f "$root/upstream-rsijudge/src/pyrualean/learn/campaign.py" ]]; then
        printf '%s\n' "Missing RSIJudge checkout at $root/upstream-rsijudge" >&2
        exit 1
    fi
    export PYTHONPATH="$root/upstream-rsijudge/src:$root:$PYTHONPATH"
    exec "$deployment/envs/robot/bin/python" -m exploration.server "$@"
fi
if [[ "${1:-}" == cells-serve || "${1:-}" == cells-script ]]; then
    subcommand=${1#cells-}
    shift
    exec "$deployment/envs/robot/bin/python" "$root/cells_server.py" "$subcommand" "$@"
fi
if [[ "${1:-}" == cells-agent ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/cells_agent.py" "$@"
fi
if [[ "${1:-}" == station-capture ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/station_capture.py" "$@"
fi
if [[ "${1:-}" == force-baseline ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/force_baseline.py" "$@"
fi
if [[ "${1:-}" == arm-hold ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/arm_hold.py" "$@"
fi
if [[ "${1:-}" == tactile-check ]]; then
    shift
    exec "$deployment/envs/robot/bin/python" "$root/tactile_check.py" "$@"
fi
exec "$deployment/envs/robot/bin/python" "$root/cli.py" "$@"
