#!/usr/bin/env bash
# Coordinator native/AOT export on a remote SM120 export host (oliver), with no
# GPU turn on raptor. Run from raptor by wip.sh's coordinator native phase.
#
#   export-remote.sh HOST STAGING_DIR KEY ENV_FILE DEV_IMAGE DEV_IMAGE_LAYERS
#
# STAGING_DIR is wip.sh's frozen source tree; KEY the native key raptor's Rust
# phase recorded; ENV_FILE the coordinator build scopes (KEY=VALUE lines).
# Bulk data moves only through sparknest (the remote host's link is a slow
# per-connection WSL2 vSwitch): the source goes up as one sealed archive
# replicated to HOST, and the export comes back through the existing native
# tier (build-cache/wip-native/x86_64/coordinator-120/KEY.tar.zst), which
# wip.sh imports and restores with CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1.
#
# On HOST: admission reads the HOST's nvidia-smi (on WSL2 a container's
# cudaMemGetInfo cannot see the host's other processes, e.g. TJ's own jobs);
# the export runs in a throwaway container of the pinned dev image under a
# memory watchdog. Exports carry no exporting-card SM count (#130) and pin
# their host target (#127), so any SM120 host's export is the same bytes.
# Exit 75 means "not here": wip.sh falls back.
set -euo pipefail
trap 'echo "export-remote: line $LINENO: \"$BASH_COMMAND\" exited $?" >&2' ERR
host="$1" staging="$2" key="$3" env_file="$4" image="$5" layers="$6"
# DEV_IMAGE_LAYERS is the sha256 of the image's RootFS layer list: its content
# identity. A local image ID is not portable (docker save | load can give the
# same layers a different ID on another host); the native key still carries
# raptor's ID through ENV_FILE (CUTEAFD_WIP_DEV_IMAGE_ID), so the export lands
# under the key raptor's own build expects.
[[ "$key" =~ ^[0-9a-f]{64}$ ]] || { echo "export-remote: bad native key" >&2; exit 2; }
NO_TURN=75
fqdn="${CUTEAFD_EXPORT_HOST_ADDR:-$host}"
ssh_host=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o "HostKeyAlias=$host" "$fqdn")
area="/mnt/sparknest/build-cache/export/$key"
archive="/mnt/sparknest/build-cache/wip-native/x86_64/coordinator-120/$key.tar.zst"
started=$SECONDS

mountpoint -q /mnt/sparknest && command -v nest >/dev/null || { echo "export-remote: no sparknest here" >&2; exit "$NO_TURN"; }
timeout 60 "${ssh_host[@]}" "mountpoint -q /mnt/sparknest && docker image inspect -f '{{json .RootFS.Layers}}' '$image' | sha256sum" 2>/dev/null |
  awk '{print $1}' | grep -qx "$layers" ||
  { echo "export-remote: $host lacks dev image $image (layers $layers) or sparknest" >&2; exit "$NO_TURN"; }

settle() {
  local i
  for ((i = 0; i < 600; i++)); do
    [[ "$(nest where --json "$1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["writing"])')" == 0 ]] && return 0
    sleep 0.5
  done
  return 1
}

if [[ ! -f "$archive" ]]; then
  mkdir -p "$area"
  tar -C "$staging" -cf - . | zstd -T8 -3 -q >"$area/.source.partial"
  mv -T "$area/.source.partial" "$area/source.tar.zst"
  grep -E '^[A-Z0-9_]+=' "$env_file" >"$area/scope.env"
  settle "$area/source.tar.zst" && settle "$area/scope.env"
  nest seal -r "$area" >/dev/null  # one path per call; -r seals the area
  timeout 900 nest replicate --hosts "$host" --wait "$area" >/dev/null
  echo "export-remote: staged source on $host in $((SECONDS - started))s"
  set +e
  timeout --kill-after=30 7200 "${ssh_host[@]}" bash -s -- "$area" "$image" "$key" "$archive" <<'REMOTE'
set -euo pipefail
area="$1" image="$2" key="$3" archive="$4"
NO_TURN=75
smi=/usr/lib/wsl/lib/nvidia-smi
[[ -x "$smi" ]] || smi=nvidia-smi
root="$HOME/.cache/cuteafd/builds/export-$key"
rm -rf "$root"; mkdir -p "$root/src" "$root/build" "$root/output" "$root/home"
zstd -d -q <"$area/source.tar.zst" | tar -C "$root/src" -xf -
placement="$root/src/scripts/build/export-placement.py"
python3 "$root/src/scripts/build/assert-build-filesystem.py" "$root"
read -r cap jobs baseline < <(python3 "$placement" admit --smi "$smi" --gpu 0) || exit "$NO_TURN"
echo "export-remote: admitted cap ${cap} MiB, ${jobs} jobs, device ${baseline} MiB in use"
status=0
python3 "$placement" watch --smi "$smi" --gpu 0 --cap-mib "$cap" --baseline-mib "$baseline" -- \
  docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e "CMAKE_BUILD_PARALLEL_LEVEL=$jobs" \
    --env-file "$area/scope.env" -e CUTEAFD_WIP_NATIVE_STORE=/wip/build/native-store \
    -v "$root/src:/wip/source" -v "$root/build:/wip/build" -v "$root/output:/wip/output" -v "$root/home:/wip/home" \
    "$image" /wip/source/scripts/build/build-wip-artifacts.sh \
    /wip/source coordinator 120 /wip/build/coordinator /wip/output/coordinator export || status=$?
if ((status == 0)); then
  read -r _ got <"$root/output/coordinator/.native-cache-key"
  entry="$root/build/native-store/coordinator-120/$got"
  if [[ "$got" != "$key" ]]; then
    echo "export-remote: exported key $got, raptor expects $key" >&2
    status="$NO_TURN"
  elif [[ -f "$entry/COMPLETE" ]]; then
    mkdir -p "$(dirname "$archive")"
    tar -C "$entry" -cf - . | zstd -T0 -3 -q >"$(dirname "$archive")/.$key.partial-$$"
    mv -T "$(dirname "$archive")/.$key.partial-$$" "$archive"
  else
    status="$NO_TURN"
  fi
fi
rm -rf "$root"
exit "$status"
REMOTE
  status=$?
  set -e
  nest rm -r "$area" >/dev/null 2>&1 || true
  ((status == 0)) || { echo "export-remote: $host export exited $status" >&2; exit "$NO_TURN"; }
  settle "$archive" && nest seal "$archive" >/dev/null
fi
echo "export-remote: $host export ready in $((SECONDS - started))s ($archive)"
