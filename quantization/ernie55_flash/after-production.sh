#!/usr/bin/env bash
# Arm only on TJ's admitted boot. All units are transient or manually started.
set -euo pipefail
run="$HOME/Developer/quantization-runs/ernie55-flash-k4-mc"
case "${1:-}" in
    --wait)
        while systemctl --user is-active --quiet cuteafd-ernie55-production.service; do sleep 60; done
        test -s "$run/production/numerical-complete.json"
        running="$(docker ps --format '{{.Names}}')"
        if grep -Fxq cuteafd-ernie55-flash-production <<< "$running"; then
            echo "The ERNIE GPU container is still running" >&2
            exit 1
        fi
        ;;
    --arm)
        repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
        test "$(hostname -s)" = maxq
        systemctl --user is-active --quiet cuteafd-ernie55-production.service || test -s "$run/production/numerical-complete.json"
        test -s "$HOME/.cache/huggingface/token"
        mkdir -p "$HOME/.config/systemd/user" "$run/progress-updates"
        install -m 644 "$repo"/quantization/ernie55_flash/systemd/cuteafd-ernie55-{publish,handoff,cleanup}.service "$HOME/.config/systemd/user"
        cp /proc/sys/kernel/random/boot_id "$run/admitted-boot-id"
        "$run/venv/bin/python" - "$run" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
request = dict(intermediates=[str(run / "production/stages")],
               build_root=str(Path.home() / ".cache/cuteafd/builds/ernie55-flash-k4-mc"))
path = run / "cleanup-request.json"
temp = path.with_suffix(".tmp")
temp.write_text(json.dumps(request) + "\n")
temp.replace(path)
PY
        systemctl --user daemon-reload
        systemd-run --user --expand-environment=no --setenv=PATH --setenv=HOME \
            --unit=cuteafd-ernie55-postproduction --property=Restart=no \
            --property=OnFailure=cuteafd-ernie55-progress.service \
            --property=OnSuccess=cuteafd-ernie55-publish.service \
            "$repo/quantization/ernie55_flash/after-production.sh" --wait
        ;;
    *) echo "usage: after-production.sh --arm|--wait" >&2; exit 2 ;;
esac
