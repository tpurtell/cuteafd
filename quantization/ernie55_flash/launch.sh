#!/usr/bin/env bash
# Start one detached phase on this boot. Arguments after the phase go to Python.
set -euo pipefail
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
run="$HOME/Developer/quantization-runs/ernie55-flash-k4-mc"
build="$HOME/.cache/cuteafd/builds/ernie55-flash-k4-mc"
phase="${1:?usage: launch.sh build|preflight|pilot PYTHON-ARGUMENT...}"
shift
case "$phase" in
    build|preflight) deadline=2h ;;
    pilot) deadline=30m ;;
    *) echo "Unknown phase $phase" >&2; exit 2 ;;
esac
(( $# )) || { echo "Missing Python arguments" >&2; exit 2; }
"$repo/scripts/build/assert-build-filesystem.py" "$build"
mkdir -p "$build" "$run/logs"
systemd-run --user --expand-environment=no --collect --setenv=PATH --setenv=HOME \
    --unit="cuteafd-ernie55-$phase" --working-directory="$repo" \
    --setenv="PYTHONPATH=$repo/quantization" \
    --property=Restart=no --property="RuntimeMaxSec=$deadline" \
    --property="StandardOutput=append:$run/logs/$phase.log" --property=StandardError=inherit \
    "$run/environment/bin/python" -m ernie55_flash.container "$phase" "$@"
