#!/usr/bin/env bash
# Run one compile command under any free build slot. raptor has 64 threads,
# so a few niced -j16 compiles run side by side instead of queueing on one
# lock. Slot 0 is the historical ~/.cache/cuteafd/build.lock, so wip.sh,
# build.sh and older flock callers still serialize against it.
#   scripts/build/build-slot.sh cargo test --workspace --no-run --locked
# CUTEAFD_BUILD_SLOTS (default 3) and CUTEAFD_BUILD_SLOT_WAIT seconds
# (default 7200) override; exit 75 when no slot frees up in time.
set -uo pipefail

slots="${CUTEAFD_BUILD_SLOTS:-3}"
wait_s="${CUTEAFD_BUILD_SLOT_WAIT:-7200}"
dir="$HOME/.cache/cuteafd"
(($# > 0)) || { echo "usage: $0 COMMAND [ARGS...]" >&2; exit 2; }
deadline=$((SECONDS + wait_s))
while :; do
  for ((i = 0; i < slots; i++)); do
    lock="$dir/build.lock"
    ((i == 0)) || lock="$dir/build.lock.$i"
    exec {fd}>>"$lock"
    if flock -n "$fd"; then
      echo "build slot $i: $*" >&2
      CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}" nice -n 19 "$@"
      exit $?
    fi
    exec {fd}>&-
  done
  ((SECONDS < deadline)) || { echo "no build slot free after ${wait_s}s" >&2; exit 75; }
  sleep 15
done
