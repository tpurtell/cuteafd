#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/lib/cdi-gpus.sh"
source "$repo_root/scripts/build/compiler-cache.sh"
source "$repo_root/scripts/build/build-caches.sh"
cuteafd_build_cache_defaults
wip_toolchain_hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"
audio_aot="${CUTEAFD_WIP_AUDIO_AOT:-OFF}"
case "$audio_aot" in ON|OFF) ;; *) release_die "CUTEAFD_WIP_AUDIO_AOT must be ON or OFF, got: $audio_aot" ;; esac
bf16_families="${CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES:-}"
bf16_family_pattern='^(mimo|mimop|mimof|glm|glmf|qwen4|step5)(;(mimo|mimop|mimof|glm|glmf|qwen4|step5))*$'
[[ -z "$bf16_families" || "$bf16_families" =~ $bf16_family_pattern ]] ||
  release_die "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES must be a semicolon list of mimo, mimop, mimof, glm, glmf, qwen4 or step5"

usage() {
  cat <<'EOF'
Usage: ./wip.sh [--slot NAME] [--role coordinator|expert|both]
                [--from-slot NAME] [--config FILE] [--seed HOST] [--recreate] [--dry-run]

Synchronizes the current checkout into persistent development containers and
incrementally builds a named WIP slot. The coordinator container builds and
runs coordinator slots. A seed host builds Spark slots, which are copied to
every configured Spark's persistent WIP container. --seed HOST (or
WIP_SEED_HOST) names the seed; it need not be a configured Spark (a build-only
seed such as kakapo is never one), and the config's SPARK_*_HOST list stays the
serving Sparks that receive the slot. Unset, WIP_BUILD_SEED defaults to kakapo.
WIP_SEED_ADDR optionally gives a seed without RDMA its bulk-transfer address
(default: its LANE_A when it is a configured Spark, else its directly attached
LAN address).
The default builds the shared release Spark role set alongside historical TP4,
independent of the serving topology. CUTEAFD_WIP_SPARK_TP_ROLES selects an
explicit subset (empty keeps only TP4).
Set CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=mimo to add BF16-input Spark siblings
for selected FAMILY:fp8 packages. This does not change serving defaults.
Set CUTEAFD_WIP_AUDIO_AOT=ON to build the optional audio tower on both SM120
and SM121. Audio serving remains separately opt-in.
--dry-run prints the resolved hosts, role plan and build invocations without
touching Docker, SSH or any container.

--from-slot NAME first clones an existing slot, then rebuilds the selected
role. This is useful for coordinator-only or expert-only A/B candidates.
For a coordinator-only clone, wip.sh stops only the coordinator process and
keeps resident Spark experts available for fingerprint-checked reuse.
WIP_INSTANCE (environment or config key) isolates containers and caches; unset
keeps the legacy names. WIP_ROOT (environment or config) optionally puts /wip
and source-staging under ~/.cache/cuteafd/builds/ on each host.
--recreate discards only this instance before
creating them again from the configured development images.
EOF
}

config="$repo_root/cuteafd.config"
slot=current
role=both
from_slot=
seed_option="${WIP_SEED_HOST:-}"
seed_explicit=0
recreate=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slot)
      slot="${2:?--slot requires a name}"
      shift 2
      ;;
    --role)
      role="${2:?--role requires coordinator, expert, or both}"
      shift 2
      ;;
    --from-slot)
      from_slot="${2:?--from-slot requires a name}"
      shift 2
      ;;
    --config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --seed)
      seed_option="${2:?--seed requires a host}"
      seed_explicit=1
      shift 2
      ;;
    --recreate)
      recreate=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown WIP argument: $1"
      ;;
  esac
done

[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid WIP slot name: $slot"
[[ -z "$from_slot" || "$from_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid source WIP slot name: $from_slot"
[[ "$slot" != "$from_slot" ]] || release_die "--from-slot must differ from --slot"
[[ -z "$seed_option" || "$seed_option" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
  release_die "invalid WIP seed host: $seed_option"
case "$role" in coordinator|expert|both) ;; *) release_die "--role must be coordinator, expert, or both" ;; esac
# A fresh coordinator-only slot (0-Spark cards) touches no Spark: no Spark dev
# image, container, process check, build or distribution.
wip_spark_leg=1
[[ "$role" != coordinator || -n "$from_slot" ]] || wip_spark_leg=0

wip_export_host_env="${CUTEAFD_EXPORT_HOST-}"
wip_export_host_set="${CUTEAFD_EXPORT_HOST+x}"
release_load_config "$config"
[[ -z "$wip_export_host_set" ]] || CUTEAFD_EXPORT_HOST="$wip_export_host_env"
export CUTEAFD_EXPORT_HOST
# Resolve once for direct builds too; explicit card/environment scopes win,
# including an empty expert list or Spark role override.
wip_scopes="$(python3 "$repo_root/scripts/lib/wip_aot_scopes.py" --config "$config" \
  --set "SPARK_COUNT=$SPARK_COUNT" --set "SPARK_TP=$SPARK_TP" --set "RTX_GPUS=$RTX_GPUS")" ||
  release_die "could not derive WIP AOT scopes from $config"
wip_scope_summary=()
while IFS='=' read -r key value; do
  if [[ "$key" == CUTEAFD_WIP_PROGRAM_FAMILY ]]; then
    wip_program_family="$value"
    continue
  fi
  [[ -v "$key" ]] || printf -v "$key" '%s' "$value"
  export "$key"
  wip_scope_summary+=("$key=${!key}")
done <<<"$wip_scopes"
audio_aot="${CUTEAFD_WIP_AUDIO_AOT:-OFF}"
case "$audio_aot" in ON|OFF) ;; *) release_die "CUTEAFD_WIP_AUDIO_AOT must be ON or OFF, got: $audio_aot" ;; esac
# Re-read after scope resolution, which derives it for families whose cards need
# BF16-input FP8 siblings (#516); an explicit value set before resolution wins.
bf16_families="${CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES:-}"
[[ -z "$bf16_families" || "$bf16_families" =~ $bf16_family_pattern ]] ||
  release_die "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES must be a semicolon list of mimo, mimop, mimof, glm, glmf, qwen4 or step5"
# EXLR8 programs (derived from EXLR8_MANIFEST, or an explicit ON/OFF): passed to both
# legs only when on, so an OFF build's environment and native key are unchanged.
case "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" in ON|OFF) ;; *) release_die "CUTEAFD_WIP_EXLR8_AOT must be ON or OFF, got: $CUTEAFD_WIP_EXLR8_AOT" ;; esac
wip_exlr8_env=()
[[ "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" == OFF ]] || wip_exlr8_env=(-e CUTEAFD_WIP_EXLR8_AOT=ON)
echo "WIP AOT scopes ($wip_program_family): ${wip_scope_summary[*]}"
case "${CUTEAFD_RELEASE_DEV_IMAGE_SOURCE:-registry}" in registry|build) ;; *) release_die "CUTEAFD_RELEASE_DEV_IMAGE_SOURCE must be registry or build" ;; esac
release_validate_wip_instance
release_need docker
release_need ssh
release_need rsync
release_need python3
release_need sha256sum
release_need install
release_need nvidia-smi

case "${CUTEAFD_WIP_EXPORT_LOCKS:-on}" in on|off) ;; *) release_die "CUTEAFD_WIP_EXPORT_LOCKS must be on or off" ;; esac
wip_export_locks=off
if [[ "${CUTEAFD_WIP_EXPORT_LOCKS:-on}" == on ]]; then
  wip_export_locks=on
  [[ "$COORDINATOR_GPU" =~ ^[01]$ ]] || release_die "export locking needs COORDINATOR_GPU=0 or 1"
  # Rust takes no hardware mutex. Native misses queue only after the seed
  # lease and Cargo build slot have been released.
fi
mapfile -t wip_hosts < <(release_spark_values HOST)
# A coordinator-only clone of a config with no active Sparks (0-Spark cards)
# also skips the Spark leg: nothing to clone, rebuild or reach.
((${#wip_hosts[@]})) || [[ "$role" != coordinator ]] || wip_spark_leg=0
((${#wip_hosts[@]} || !wip_spark_leg)) || release_die "configuration has no active Spark hosts"
[[ "${#wip_hosts[@]}" == "$SPARK_COUNT" ]] ||
  release_die "SPARK_COUNT=$SPARK_COUNT does not match ${#wip_hosts[@]} configured Spark hosts"
# The seed builds the Spark slot; the configured Sparks (wip_hosts) serve it.
# A seed outside them (kakapo) is a build host only: it gets the build
# container, never a serving one, and is never a distribution target. Every
# host that holds this instance's Spark WIP container is wip_container_hosts.
seed_host="${seed_option:-${WIP_BUILD_SEED:-${wip_hosts[0]:-}}}"
wip_target_hosts=()
wip_seed_serves=0
for host in "${wip_hosts[@]}"; do
  if [[ "$host" == "$seed_host" ]]; then wip_seed_serves=1; else wip_target_hosts+=("$host"); fi
done
wip_container_hosts=("${wip_hosts[@]}")
((wip_seed_serves)) || wip_container_hosts=("$seed_host" "${wip_hosts[@]}")

# Opt-in replicated-group Spark expert roles for the WIP slot. Shared slots
# derive coverage from the release role configuration; an explicit empty
# override keeps the legacy TP4-only escape hatch.
wip_spark_tp_roles="$(release_spark_tp_roles_canonical \
  "${CUTEAFD_WIP_SPARK_TP_ROLES-$(release_spark_tp_roles_default)}" \
  CUTEAFD_WIP_SPARK_TP_ROLES)"

if ((dry_run)); then
  echo "WIP dry-run passed; no container, image, SSH or build operation was performed."
  echo "  config: $RELEASE_CONFIG"
  echo "  slot: $slot (role $role)"
  echo "  containers: $(release_wip_container coordinator), $(release_wip_container spark-expert)"
  echo "  WIP root: ${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}"
  echo "  Spark leg: $( ((wip_spark_leg)) && echo on || echo off)"
  echo "  Spark hosts (${#wip_hosts[@]}): $(IFS=,; echo "${wip_hosts[*]}")"
  echo "  seed host: $seed_host ($( ((wip_seed_serves)) && echo serving || echo build-only), transfer ${CUTEAFD_WIP_SEED_TRANSFER:-auto}${WIP_SEED_ADDR:+ via $WIP_SEED_ADDR})"
  echo "  Spark targets (${#wip_target_hosts[@]}): $(IFS=,; echo "${wip_target_hosts[*]}")"
  echo "  topology: tp=$(release_spark_tp) ep=$(release_spark_ep) explicit=$(release_spark_topology_explicit && echo 1 || echo 0)"
  echo "  V41 Spark expert roles: ${wip_spark_tp_roles:-<legacy TP4 only>}"
  echo "  EXL3 AOT: ${CUTEAFD_WIP_EXL3_AOT:-ON}; NVFP4 AOT: ${CUTEAFD_WIP_NVFP4_AOT:-ON}"
  echo "  Audio AOT: $audio_aot"
  echo "  FP8 BF16-input siblings: ${bf16_families:-<none>}"
  echo "  EXLR8 AOT: ${CUTEAFD_WIP_EXLR8_AOT:-OFF}"
  cuteafd_build_cache_docker_args "${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}" /wip/home "$wip_toolchain_hash" dry >/dev/null
  echo "  Spark cache plan (host HOME template; states/admission and uid rechecked on each Spark, no SSH):"
  cuteafd_build_cache_docker_args "${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}" /wip/home "$wip_toolchain_hash" dry aarch64 >/dev/null
  exit 0
fi

# A coordinator-only retry repairs distribution of a Spark slot sealed by a
# previous both-role build, without building it again. Fresh 0-Spark slots stay local.
if [[ "$role" == coordinator && -z "$from_slot" ]] && ((${#wip_hosts[@]})) &&
   timeout 30 ssh -o BatchMode=yes "$seed_host" \
     "root=\$(docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/wip\"}}{{.Source}}{{end}}{{end}}' '$(release_wip_container spark-expert)'); test -s \"\$root/slots/$slot/spark-expert/FINGERPRINT\""; then
  wip_spark_leg=1
fi

# Probe only after dry-run exits; a dry-run performs no network operations.
if [[ "$seed_host" == kakapo && "$role" != coordinator ]] && ((seed_explicit == 0)) &&
   ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=5 kakapo true; then
  for host in rhea moa; do
    if timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" true; then
      seed_host="$host"
      unset WIP_SEED_ADDR
      break
    fi
  done
  wip_target_hosts=()
  wip_seed_serves=0
  for host in "${wip_hosts[@]}"; do
    if [[ "$host" == "$seed_host" ]]; then wip_seed_serves=1; else wip_target_hosts+=("$host"); fi
  done
  wip_container_hosts=("${wip_hosts[@]}")
  ((wip_seed_serves)) || wip_container_hosts=("$seed_host" "${wip_hosts[@]}")
fi

wip_build_resource_release() {
  [[ -n "${wip_build_resource_pid:-}" ]] || return 0
  if [[ "${wip_resource_complete:-0}" == 1 ]]; then
    printf x >"$wip_build_resource_dir/release"
  else
    kill "$wip_build_resource_pid" 2>/dev/null || true
  fi
  wait "$wip_build_resource_pid" 2>/dev/null || true
  rm -rf "$wip_build_resource_dir"
  unset wip_build_resource_pid
}
wip_build_resource_acquire() {
  [[ "$wip_export_locks" == on ]] || return 0
  # Multiple instances have disjoint containers, source trees and Cargo targets.
  # The unsuffixed legacy container can use slot zero but not multiple leases.
  [[ -n "${WIP_INSTANCE:-}" ]] || CUTEAFD_SEED_BUILD_SLOTS=1
  wip_build_resource_dir="$(mktemp -d "${TMPDIR:-/tmp}/wip-build-resource.XXXXXXXX")"
  mkfifo "$wip_build_resource_dir/release"
  resource_args=(--host "$seed_host" --lock-host "$(wip_seed_lock "$seed_host")" --owner "$BASHPID" --container "$spark_container" --lease-dir "$wip_build_resource_dir" --wait "${CUTEAFD_WIP_EXPORT_WAIT_S:-14400}")
  if [[ "$seed_host" != kakapo ]]; then resource_args+=(--slots 1)
  elif [[ -n "${CUTEAFD_SEED_BUILD_SLOTS:-}" ]]; then resource_args+=(--slots "$CUTEAFD_SEED_BUILD_SLOTS"); fi
  # Explicit --seed is a placement request; automatic/environment seeds may fall back.
  [[ "$seed_host" != kakapo ]] || ((seed_explicit)) || resource_args+=(--fallback "${CUTEAFD_WIP_SEED_FALLBACK:-rhea,moa}")
  python3 "$repo_root/scripts/build/build-resource.py" "${resource_args[@]}" &
  wip_build_resource_pid=$!
  trap wip_build_resource_release EXIT
  while [[ ! -s "$wip_build_resource_dir/granted" ]]; do
    kill -0 "$wip_build_resource_pid" 2>/dev/null || release_die "no Spark build resource available"
    sleep 1
  done
  local original_seed="$seed_host"
  seed_host="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["host"])' "$wip_build_resource_dir/granted")"
  [[ "$seed_host" == kakapo ]] || unset WIP_SEED_ADDR
  wip_seed_serves=0
  wip_target_hosts=()
  for host in "${wip_hosts[@]}"; do
    if [[ "$host" == "$seed_host" ]]; then wip_seed_serves=1; else wip_target_hosts+=("$host"); fi
  done
  wip_container_hosts=("${wip_hosts[@]}")
  ((wip_seed_serves)) || wip_container_hosts=("$seed_host" "${wip_hosts[@]}")
  echo "WIP Spark build resource: $seed_host"
  : >"$wip_build_resource_dir/active"
  if [[ "$seed_host" != "$original_seed" ]]; then
    wip_seed_rdma=1
    wip_seed_addr=""
    ensure_seed_image
    ensure_remote_container "$seed_host"
  fi
}

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
release_resolve_coordinator_gpu_identity

coordinator_container="$(release_wip_container coordinator)"
spark_container="$(release_wip_container spark-expert)"
state_dir="${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}"
wip_mount_root="$state_dir"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$state_dir"
if [[ -n "${WIP_ROOT:-}" ]]; then
  python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$WIP_ROOT"
fi
staging_dir="$state_dir/source-staging"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$state_dir" "$hf_home"

snapshot_args=(
  -a --delete --delete-excluded
  --exclude .git --exclude '.venv*/' --exclude .mypy_cache/
  --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude __pycache__/
  --exclude '*.pyc' --exclude '*.pyo' --exclude .cuteafd-cache/
  --exclude .cuteafd-release/ --exclude .cuteafd-release-image/
  --exclude '.cuteafd-wip*' --exclude dist/ --exclude rust/target/
  --exclude 'native/build*/'
  --filter 'P .cuteafd-source-revision'
)

echo "== freezing current checkout for WIP slot $slot =="
mkdir -p "$staging_dir"
rsync "${snapshot_args[@]}" "$repo_root/" "$staging_dir/"
# The selected complete configuration is part of the slot, even when the
# caller chose a file other than the repository's default cuteafd.config.
install -m 0644 "$RELEASE_CONFIG" "$staging_dir/cuteafd.config"
# The live console names the source revision; slots carry no .git, so the
# daemon's build script reads it from this stamp (rewritten only on change).
source_revision="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=no 2>/dev/null || true)" ]] ||
  source_revision+=" dirty"
[[ "$(cat "$staging_dir/.cuteafd-source-revision" 2>/dev/null || true)" == "$source_revision" ]] ||
  printf '%s\n' "$source_revision" >"$staging_dir/.cuteafd-source-revision"
# The frozen tree has no .git: its identity goes in BUILD_IDENTITY.json for report footers.
"$repo_root/scripts/build/write-build-identity.sh" "$repo_root" "$staging_dir/BUILD_IDENTITY.json"
python3 "$staging_dir/scripts/build/verify-sparkinfer-source.py" \
  --source "$staging_dir/third_party/sparkinfer" \
  --lock "$staging_dir/third_party/sparkinfer.lock.json"
python3 "$staging_dir/scripts/build/verify-xgrammar-source.py" \
  --source "$staging_dir/third_party/xgrammar" \
  --lock "$staging_dir/third_party/xgrammar.lock.json"
python3 "$staging_dir/scripts/build/verify-transformers-source.py" \
  --source "$staging_dir/third_party/transformers" \
  --lock "$staging_dir/third_party/transformers.lock.json"

sparkinfer_revision="$(python3 "$staging_dir/scripts/build/verify-sparkinfer-source.py" \
  --source "$staging_dir/third_party/sparkinfer" \
  --lock "$staging_dir/third_party/sparkinfer.lock.json" \
  --print-revision)"

# Explicit registry refs retain pull-if-missing behavior. Missing default local
# tags prefer the checkout-matched published toolchain, then build on that host.
ensure_local_image() {
  if [[ "$COORDINATOR_DOCKER_DEV" == */* ]] || docker image inspect "$COORDINATOR_DOCKER_DEV" >/dev/null 2>&1; then
    release_ensure_dev_image "$COORDINATOR_DOCKER_DEV"
  else
    local admitted
    admitted="$(python3 "$staging_dir/scripts/build/select-dev-image.py" select \
      --source "$staging_dir" --arch amd64 --local-tag "$COORDINATOR_DOCKER_DEV" \
      --engine-commit "$source_revision" --mode "${CUTEAFD_RELEASE_DEV_IMAGE_SOURCE:-registry}" \
      --output "$dev_image_logs/coordinator.json")"
    docker tag "$admitted" "$COORDINATOR_DOCKER_DEV"
  fi
}
ensure_seed_image() (
  if [[ "$SPARK_EXPERT_DOCKER_DEV" == */* ]] || release_ssh "$seed_host" docker image inspect "$SPARK_EXPERT_DOCKER_DEV" >/dev/null 2>&1; then
    release_ensure_dev_image "$SPARK_EXPERT_DOCKER_DEV" "$seed_host"
  else
    # Only these immutable toolchain inputs are needed for a fallback dev build.
    local remote_source
    remote_source="$(release_ssh "$seed_host" 'mkdir -p "$HOME/.cache/cuteafd/builds/wip-dev-image"; mktemp -d "$HOME/.cache/cuteafd/builds/wip-dev-image/source.XXXXXXXX"')"
    release_validate_path_setting remote_source "$remote_source"
    trap 'release_ssh "$seed_host" "rm -rf $remote_source" >/dev/null 2>&1 || true' EXIT
    tar -C "$staging_dir" -cf - docker/Dockerfile.dev docker/entrypoint.sh \
      scripts/build/install-dev-cache-tools.sh scripts/build/dev-toolchain.py \
      scripts/build/select-dev-image.py scripts/build/assert-build-filesystem.py .dockerignore |
      release_ssh "$seed_host" "tar -C $remote_source -xf -"
    release_ssh "$seed_host" setsid --wait bash -s -- "$remote_source" "$SPARK_EXPERT_DOCKER_DEV" \
      "$(printf '%q' "$source_revision")" "${CUTEAFD_RELEASE_DEV_IMAGE_SOURCE:-registry}" "$dev_image_run_id" <<'REMOTE'
set -euo pipefail
source_dir="$1"
process_dir="$HOME/.cache/cuteafd/builds/wip-dev-image/processes"
mkdir -p "$process_dir"
process_file="$process_dir/$5.pid"
cancel_file="$process_dir/$5.cancel"
[[ ! -e "$cancel_file" ]] || exit 143
printf '%s\n' "$$" >"$process_file"
cleanup_dev_phase() {
  trap '' HUP INT TERM
  kill -TERM -- "-$$" 2>/dev/null || true
  rm -f "$process_file"
  rm -rf "$source_dir"
}
trap cleanup_dev_phase EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
[[ ! -e "$cancel_file" ]] || exit 143
python3 "$source_dir/scripts/build/assert-build-filesystem.py" "$source_dir"
admitted="$(python3 "$source_dir/scripts/build/select-dev-image.py" select \
  --source "$source_dir" --arch arm64 --local-tag "$2" --engine-commit "$3" \
  --mode "$4" --output "$source_dir/DEV_IMAGE_REUSE.json")"
docker tag "$admitted" "$2"
REMOTE
  trap - EXIT
  fi
)

source "$repo_root/scripts/lib/build-supervision.sh"
dev_image_log_root="$HOME/.cache/cuteafd/builds/wip-dev-image"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$dev_image_log_root"
mkdir -p "$dev_image_log_root"
dev_image_logs="$(mktemp -d "$dev_image_log_root/legs.XXXXXXXX")"
dev_image_run_id="wip-dev-$(basename "$dev_image_logs")"
cancel_local_dev_image() { :; }
cancel_seed_dev_image() {
  timeout 30 ssh "${release_ssh_opts[@]}" -o ConnectTimeout=5 "$seed_host" bash -s -- "$dev_image_run_id" <<'CANCEL'
set -euo pipefail
process_dir="$HOME/.cache/cuteafd/builds/wip-dev-image/processes"
mkdir -p "$process_dir"
: >"$process_dir/$1.cancel"
pid=""
[[ ! -f "$process_dir/$1.pid" ]] || read -r pid <"$process_dir/$1.pid" || true
if [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] &&
   grep -zFq -- "$1" "/proc/$pid/cmdline"; then
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((i=0; i<5; i++)); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
fi
rm -f "$process_dir/$1.pid"
CANCEL
}
release_configure_ssh_transport
# The seed's transport. A seed on the RoCE fabric streams directly (rdmasync,
# rdmapipe). A seed without RDMA (a 10 GbE build seed such as kakapo) moves bulk
# data over WIP_SEED_ADDR (default: its LANE_A when it is a configured Spark,
# else its directly attached LAN address from scripts/build/seed-transfer.py;
# not a slower name route) with release_bulk_ssh, compresses what
# crosses that link, and hands sealed slots to the serving Sparks through
# sparknest (CUTEAFD_WIP_SEED_TRANSFER: auto, nest or stream; see
# wip_slot_transfer_mode).
wip_seed_rdma=1
wip_seed_addr=""
wip_seed_transfer="${CUTEAFD_WIP_SEED_TRANSFER:-auto}"
case "$wip_seed_transfer" in auto|nest|stream) ;; *) release_die "CUTEAFD_WIP_SEED_TRANSFER must be auto, nest or stream" ;; esac
if ((wip_spark_leg)) && ! release_host_has_rdma "$seed_host"; then
  wip_seed_rdma=0
  wip_seed_addr="${WIP_SEED_ADDR:-}"
  if [[ -z "$wip_seed_addr" ]]; then
    for ((wip_rank = 0; wip_rank < SPARK_COUNT; wip_rank++)); do
      wip_key="SPARK_${wip_rank}_HOST"
      [[ "${!wip_key}" != "$seed_host" ]] || { wip_key="SPARK_${wip_rank}_LANE_A"; wip_seed_addr="${!wip_key}"; }
    done
  fi
  [[ -n "$wip_seed_addr" ]] ||
    wip_seed_addr="$(python3 "$repo_root/scripts/build/seed-transfer.py" lan-addr "$seed_host")" ||
    release_die "WIP seed $seed_host has no RDMA and no reachable LAN address; set WIP_SEED_ADDR"
  echo "WIP seed $seed_host has no RDMA: bulk transfers over $wip_seed_addr"
fi
if ((wip_spark_leg)); then
  ( build_supervise 'WIP dev image' "$dev_image_logs" 0 \
    coordinator ensure_local_image cancel_local_dev_image coordinator.log \
    expert ensure_seed_image cancel_seed_dev_image expert.log )
else
  ( build_supervise 'WIP dev image' "$dev_image_logs" 0 \
    coordinator ensure_local_image cancel_local_dev_image coordinator.log )
fi
trap wip_build_resource_release EXIT

# stream_between_hosts SRC SRC_CMD DST DST_CMD: SRC_CMD's stdout into DST_CMD's stdin, over
# rdmapipe when both hosts have it, otherwise a plain ssh pipe relayed through this host (the
# fallback scripts/build/build-dev-images.sh uses for the development images).
stream_between_hosts() {
  local src="$1" src_cmd="$2" dst="$3" dst_cmd="$4"
  if [[ "$src" == "$seed_host" && "${wip_seed_rdma:-1}" == 0 ]]; then
    # The seed's leg crosses 10 GbE: compress it, fast cipher, LAN address.
    release_bulk_ssh "$src" "$wip_seed_addr" "$src_cmd | zstd -T8 -3 -q" |
      ssh -o BatchMode=yes "$dst" "zstd -d -q | $dst_cmd"
  elif ssh -o BatchMode=yes "$src" 'command -v rdmapipe >/dev/null' &&
    ssh -o BatchMode=yes "$dst" 'command -v rdmapipe >/dev/null'; then
    ssh -o BatchMode=yes "$src" "$src_cmd | rdmapipe --send" | ssh -o BatchMode=yes "$dst" "rdmapipe --recv | $dst_cmd"
  else
    ssh -o BatchMode=yes "$src" "$src_cmd" | ssh -o BatchMode=yes "$dst" "$dst_cmd"
  fi
}

distribute_spark_dev_image() {
  local seed_id
  seed_id="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  local -a targets=()
  local host remote_id
  for host in "${wip_target_hosts[@]}"; do
    remote_id="$(ssh -o BatchMode=yes "$host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV' 2>/dev/null || true")"
    [[ "$remote_id" == "$seed_id" ]] || targets+=("$host")
  done
  ((${#targets[@]})) || return 0
  local source="$seed_host"
  if [[ "$wip_seed_rdma" == 0 ]]; then
    # One compressed crossing of the seed's 10 GbE link, then fan out on RoCE.
    echo "== distributing Spark development image from $seed_host to ${targets[0]} =="
    (
      set -o pipefail
      stream_between_hosts "$seed_host" "docker image save '$SPARK_EXPERT_DOCKER_DEV'" "${targets[0]}" 'docker image load'
    ) || release_die "Spark development image distribution to ${targets[0]} failed"
    source="${targets[0]}"
    targets=("${targets[@]:1}")
    ((${#targets[@]})) || return 0
  fi
  echo "== concurrently distributing Spark development image from $source =="
  local -a pids=()
  for host in "${targets[@]}"; do
    (
      set -o pipefail
      stream_between_hosts "$source" "docker image save '$SPARK_EXPERT_DOCKER_DEV'" "$host" 'docker image load'
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark development image distribution failed"
}

# --recreate follows --role: coordinator touches only raptor's container and
# state, expert only the Spark legs, both everything. A leg the role does not
# name keeps its containers and sealed slot outputs.
recreate_local=1 recreate_remote=1
[[ "$role" != expert ]] || recreate_local=0
[[ "$role" != coordinator ]] || recreate_remote=0
remove_wip_containers() {
  ((recreate_local == 0)) || docker rm -f "$coordinator_container" >/dev/null 2>&1 || true
  local host
  ((recreate_remote == 0)) || for host in "${wip_container_hosts[@]}"; do
    ssh -o BatchMode=yes "$host" "docker rm -f '$spark_container' >/dev/null 2>&1 || true" &
  done
  wait
}

# Refuse root-owned trees before removing containers: normal builds never repair
# permissions, and legacy slots remain runnable via run.sh --wip unchanged.
bad_owner="$(find "$state_dir" -maxdepth 2 ! -user "$(id -u)" -print -quit)"
if [[ ! -w "$state_dir" || -n "$bad_owner" ]]; then
  release_die "WIP path ${bad_owner:-$state_dir} is not owned/writable by host uid $(id -u); choose a fresh WIP instance with --recreate, or have the user run agent-sudo chown -R $(id -u):$(id -g) -- '${bad_owner:-$state_dir}' (no automatic chown or deletion)"
fi
if ((recreate)); then
  # Refuse before removing any container or payload, including remote uid-1001
  # trees. Legacy root slots stay available until an operator chooses a new root.
  ((recreate_remote == 0)) || for host in "${wip_container_hosts[@]}"; do
    ssh -o BatchMode=yes "$host" bash -s -- "${WIP_ROOT:-__default__}" "${WIP_INSTANCE:-__none__}" <<'CHECK_ROOT'
set -euo pipefail
root="$1"
instance="$2"
[[ "$instance" != __none__ ]] || instance=
[[ "$root" != __default__ ]] || root="$HOME/.cache/cuteafd/builds/wip${instance:+-$instance}"
bad_owner=
[[ ! -e "$root" ]] || bad_owner="$(find "$root" -maxdepth 2 ! -user "$(id -u)" -print -quit)"
if [[ -e "$root" && ( ! -w "$root" || -n "$bad_owner" ) ]]; then
  echo "WIP path ${bad_owner:-$root} is not owned/writable by host uid $(id -u); choose a fresh WIP instance with --recreate, or have the user run agent-sudo chown -R $(id -u):$(id -g) -- '${bad_owner:-$root}' (no automatic chown or deletion)" >&2
  exit 2
fi
CHECK_ROOT
  done
  echo "== discarding persistent WIP containers and build caches =="
  remove_wip_containers
  # The frozen source and global stores survive. --recreate clears this
  # instance's build payloads even when its host-backed root was implicit.
  for part in build output slots incoming run cache; do
    ((recreate_local == 0)) || rm -rf "$state_dir/$part"
    ((recreate_remote == 0)) || for host in "${wip_container_hosts[@]}"; do
      ssh -o BatchMode=yes "$host" bash -s -- "${WIP_ROOT:-__default__}" "${WIP_INSTANCE:-__none__}" "$part" <<'CLEAR_ROOT'
set -euo pipefail
root="$1"
instance="$2"
[[ "$instance" != __none__ ]] || instance=
[[ "$root" != __default__ ]] || root="$HOME/.cache/cuteafd/builds/wip${instance:+-$instance}"
rm -rf "$root/$3"
CLEAR_ROOT
    done
  done
fi

preflight_existing_container_images() {
  local expected actual host coordinator_device
  expected="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  if docker container inspect "$coordinator_container" >/dev/null 2>&1; then
    actual="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
    [[ "$actual" == "$expected" ]] ||
      release_die "$coordinator_container uses an old development image; rerun ./wip.sh --recreate"
    coordinator_device="$(
      docker inspect -f \
        '{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{println .}}{{end}}{{end}}' \
        "$coordinator_container"
    )"
    # CDI records the request as nvidia.com/gpu=UUID; containers made with --gpus record the bare UUID.
    coordinator_device="${coordinator_device#nvidia.com/gpu=}"
    [[ "$coordinator_device" == "$RELEASE_COORDINATOR_GPU_UUID" ]] ||
      release_die "$coordinator_container is not bound to the configured physical GPU; rerun ./wip.sh --recreate"
  fi
  ((wip_spark_leg)) || return 0
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  for host in "${wip_container_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker inspect -f '{{.Image}}' '$spark_container' 2>/dev/null || true")"
    [[ -z "$actual" || "$actual" == "$expected" ]] ||
      release_die "$host $spark_container uses an old development image; rerun ./wip.sh --recreate"
  done
}

preflight_existing_container_images
((wip_spark_leg == 0)) || distribute_spark_dev_image

# wip_container_gpu_ok CONTAINER: its GPU still answers NVML.
wip_container_gpu_ok() {
  timeout 60 docker exec "$1" nvidia-smi -L >/dev/null 2>&1
}

ensure_local_container() {
  local image_id container_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  if docker container inspect "$coordinator_container" >/dev/null 2>&1; then
    [[ "$(docker inspect -f '{{.Config.User}}' "$coordinator_container")" == "$(id -u):$(id -g)" ]] ||
      release_die "$coordinator_container is a legacy root WIP container; run.sh --wip still works, but rebuilding requires --recreate with a fresh writable WIP_ROOT"
    [[ "$(docker inspect -f '{{index .Config.Labels "io.cuteafd.build-caches"}}' "$coordinator_container")" == "${CUTEAFD_BUILD_CACHES:-on}:$wip_toolchain_hash" ]] ||
      release_die "$coordinator_container cache plan changed; rerun ./wip.sh --recreate"
    container_id="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
    [[ "$container_id" == "$image_id" ]] ||
      release_die "$coordinator_container uses an old development image; rerun ./wip.sh --recreate"
    if [[ -n "$wip_mount_root" ]]; then
      [[ "$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/wip"}}{{.Source}}{{end}}{{end}}' "$coordinator_container")" == "$(realpath "$wip_mount_root")" ]] ||
        release_die "$coordinator_container has a different /wip build mount; rerun ./wip.sh --recreate"
    fi
    cuteafd_build_cache_docker_args "$wip_mount_root" /wip/home "$wip_toolchain_hash" dry >/dev/null
    [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]] ||
      docker start "$coordinator_container" >/dev/null
    docker exec "$coordinator_container" test -d /export-slots ||
      release_die "$coordinator_container lacks host worker admission; rerun ./wip.sh --recreate"
    # A container can outlive its GPU access (a host daemon-reload drops NVML in
    # running containers). Its state is all on the /wip bind mount, so recreate it
    # rather than fail later in the export.
    if wip_container_gpu_ok "$coordinator_container"; then
      docker exec "$coordinator_container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
      return
    fi
    echo "WIP: $coordinator_container lost GPU access (nvidia-smi -L failed); recreating it" >&2
    docker rm -f "$coordinator_container" >/dev/null
  fi
  cuteafd_cdi_gpu_args "$RELEASE_COORDINATOR_GPU_UUID" || exit 2
  local -a args=(
    run -d --name "$coordinator_container" --restart no
    --label "io.cuteafd.build-caches=${CUTEAFD_BUILD_CACHES:-on}:$wip_toolchain_hash"
    --user "$(id -u):$(id -g)"
    -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
    "${CUTEAFD_CDI_ARGS[@]}"
    --net=host --ipc=host --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK
    -v "$hf_home:$hf_home:ro" -v "$hf_home:/root/.cache/huggingface:ro"
    -e "WIP_INSTANCE=${WIP_INSTANCE:-}" -e "WIP_ROOT=${WIP_ROOT:-}"
    -e HF_HOME="$hf_home"
    -e CUDA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
    -e NVIDIA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
  )
  local -a cache_args=()
  local cache_plan
  cache_plan="$(cuteafd_build_cache_docker_args "$wip_mount_root" /wip/home "$wip_toolchain_hash")" || release_die "WIP cache plan failed"
  mapfile -t cache_args <<<"$cache_plan"
  args+=("${cache_args[@]}")
  mkdir -p "$wip_mount_root/home"
  args+=(-v "$wip_mount_root:/wip")
  [[ ! -e /dev/infiniband ]] || args+=(--device=/dev/infiniband)
  # sparknest keeps hub/ as a symlink into its mount; expose it at the same path.
  [[ ! -d /mnt/sparknest ]] || args+=(-v /mnt/sparknest:/mnt/sparknest:ro)
  docker "${args[@]}" "$COORDINATOR_DOCKER_DEV" sleep infinity >/dev/null
  docker exec "$coordinator_container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
}

ensure_remote_container() {
  local host="$1" cache_staging= cache_helper_dir
  {
    if cache_staging="$(ssh -o BatchMode=yes "$host" 'printf "%s/.cache/cuteafd/kache-helper" "$HOME"')${WIP_INSTANCE:+-$WIP_INSTANCE}"; then
      printf -v cache_helper_dir '%q' "$cache_staging/scripts/build"
      if ! ssh -o BatchMode=yes "$host" "mkdir -p $cache_helper_dir" ||
         ! scp -q "$repo_root/scripts/build/compiler-cache.sh" "$repo_root/scripts/build/assert-build-filesystem.py" "$repo_root/scripts/build/build-caches.sh" "$repo_root/scripts/build/build-cache-plan.py" "$host:$cache_staging/scripts/build/" ||
         ! ssh -o BatchMode=yes "$host" "mkdir -p ${cache_helper_dir%/build}/lib" ||
         ! scp -q "$repo_root/scripts/lib/cdi-gpus.sh" "$host:$cache_staging/scripts/lib/"; then
        cuteafd_compiler_cache_warn 'cannot stage Spark cache helper'
        cache_staging=
      fi
    else
      cuteafd_compiler_cache_warn 'cannot stage Spark cache helper'
      cache_staging=
    fi
  }
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$spark_container" "$SPARK_EXPERT_DOCKER_DEV" \
    "$(printf '%q' "${CUTEAFD_KACHE_SPARK:-__unset__}")" \
    "$(printf '%q' "${CUTEAFD_KACHE_REMOTE:-__unset__}")" \
    "$(printf '%q' "${CUTEAFD_KACHE_SPARK_CACHE_DIR:-__unset__}")" \
    "$(printf '%q' "${cache_staging:-__unset__}")" "${WIP_INSTANCE:-__none__}" "${WIP_ROOT:-__none__}" \
    "$(base64 -w0 "$repo_root/scripts/build/assert-build-filesystem.py")" "${CUTEAFD_SCCACHE_CUDA:-0}" "${CUTEAFD_BUILD_CACHES:-on}" "$wip_toolchain_hash" <<'REMOTE'
set -euo pipefail
container="$1"
image="$2"
instance="${7:-__none__}"
[[ "$instance" != __none__ ]] || instance=
root="${8:-__none__}"
[[ "$root" != __none__ ]] || root="$HOME/.cache/cuteafd/builds/wip${instance:+-$instance}"
if [[ -n "$root" ]]; then
  [[ "$root" == "$HOME/.cache/cuteafd/builds/"* ]] || { echo "WIP_ROOT outside host build root" >&2; exit 2; }
  printf '%s' "$9" | base64 -d | python3 - "$root"
  mkdir -p "$root"
fi
cache_args=()
export CUTEAFD_BUILD_CACHES="${11:-on}" CUTEAFD_SCCACHE_CUDA="${10:-1}"
[[ "${3:-__unset__}" == __unset__ ]] || export CUTEAFD_KACHE="$3"
[[ "${5:-__unset__}" == __unset__ ]] || export CUTEAFD_KACHE_CACHE_DIR="$5"
[[ -f "$6/scripts/build/build-caches.sh" ]] || { echo "Spark cache helper not staged" >&2; exit 2; }
source "$6/scripts/build/build-caches.sh"
[[ -f "$6/scripts/lib/cdi-gpus.sh" ]] || { echo "Spark CDI helper not staged" >&2; exit 2; }
source "$6/scripts/lib/cdi-gpus.sh"
cuteafd_build_cache_defaults
cache_plan="$(cuteafd_build_cache_docker_args "$root" /wip/home "${12:?}")" || exit 2
mapfile -t cache_args <<<"$cache_plan"
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
if docker container inspect "$container" >/dev/null 2>&1; then
  [[ "$(docker inspect -f '{{.Config.User}}' "$container")" == "$(id -u):$(id -g)" ]] ||
    { echo "$container is a legacy root WIP container; run.sh --wip still works, but rebuilding requires --recreate with a fresh writable WIP_ROOT" >&2; exit 2; }
  [[ "$(docker inspect -f '{{index .Config.Labels "io.cuteafd.build-caches"}}' "$container")" == "${CUTEAFD_BUILD_CACHES:-on}:${12:?}" ]] ||
    { echo "$container cache plan changed; rerun ./wip.sh --recreate" >&2; exit 2; }
  container_id="$(docker inspect -f '{{.Image}}' "$container")"
  if [ "$container_id" != "$image_id" ]; then
    echo "$container uses an old development image; rerun ./wip.sh --recreate" >&2
    exit 2
  fi
  if [[ -n "$root" ]]; then
    [[ "$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/wip"}}{{.Source}}{{end}}{{end}}' "$container")" == "$(realpath "$root")" ]] ||
      { echo "$container has a different /wip build mount; rerun ./wip.sh --recreate" >&2; exit 2; }
  fi
  [ "$(docker inspect -f '{{.State.Running}}' "$container")" = true ] || docker start "$container" >/dev/null
  docker exec "$container" test -d /export-slots || { echo "$container lacks host worker admission; rerun ./wip.sh --recreate" >&2; exit 2; }
  docker exec "$container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
  exit 0
fi
bad_owner="$(find "$root" -maxdepth 2 ! -user "$(id -u)" -print -quit)"
if [[ ! -w "$root" || -n "$bad_owner" ]]; then
  echo "WIP path ${bad_owner:-$root} is not owned/writable by host uid $(id -u); choose a fresh WIP instance with --recreate, or have the user run agent-sudo chown -R $(id -u):$(id -g) -- '${bad_owner:-$root}' (no automatic chown or deletion)" >&2
  exit 2
fi
mkdir -p "$root/home"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
cuteafd_cdi_gpu_args all || exit 2
args=(
  run -d --name "$container" --restart no
  --label "io.cuteafd.build-caches=${CUTEAFD_BUILD_CACHES:-on}:${12:?}"
  --user "$(id -u):$(id -g)"
  -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
  "${CUTEAFD_CDI_ARGS[@]}" --net=host --ipc=host --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK
  -v "$hf_home:$hf_home:ro" -v "$hf_home:/root/.cache/huggingface:ro"
  -e "WIP_INSTANCE=$instance" -e "WIP_ROOT=$root"
  -e HF_HOME="$hf_home"
)
args+=("${cache_args[@]}")
[[ -z "$root" ]] || args+=(-v "$root:/wip")
[ ! -e /dev/infiniband ] || args+=(--device=/dev/infiniband)
[ ! -d /mnt/sparknest ] || args+=(-v /mnt/sparknest:/mnt/sparknest:ro)
docker "${args[@]}" "$image" sleep infinity >/dev/null
docker exec "$container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
REMOTE
}

# Content-keyed slot reuse. The build key is the committed source tree (with
# each submodule's pinned commit, as gitlinks in the tree), the slot's config,
# every resolved build scope, the Spark role set and both dev images. A clean
# checkout whose key matches a sealed slot elsewhere on these hosts adopts that
# slot (hardlinks, seal verified, identity resealed) instead of rebuilding.
# Dirty trees, --from-slot clones and CUTEAFD_WIP_REUSE=off always build.
# Host CPU targets of the coordinator slot (scripts/build/host-isa.sh): this
# host's own ISA unless CUTEAFD_WIP_HOST_ISA names a list (base first). The
# Spark leg always builds its one GB10 target.
source "$repo_root/scripts/build/host-isa.sh"
wip_coordinator_host_isa="$(cuteafd_host_isa_list wip "${CUTEAFD_WIP_HOST_ISA:-}")" ||
  release_die "CUTEAFD_WIP_HOST_ISA: ${CUTEAFD_WIP_HOST_ISA:-}"
wip_coordinator_host_isa="${wip_coordinator_host_isa//$'\n'/ }"
# Dev-only Cargo features for the coordinator daemon: a
# space- or comma-separated list of dev-* names, keyed into the build and
# recorded in the slot's build phases. Production features are never set here.
wip_cargo_features="${CUTEAFD_WIP_CARGO_FEATURES:-}"
wip_cargo_features="${wip_cargo_features//,/ }"
for feature in $wip_cargo_features; do
  [[ "$feature" =~ ^dev-[a-z0-9-]+$ ]] || release_die "CUTEAFD_WIP_CARGO_FEATURES: only dev-* features may be set ($feature)"
done
wip_cargo_features="$(echo $wip_cargo_features)"
wip_build_key=""
wip_coordinator_key=""
wip_expert_key=""
wip_reuse_mode="${CUTEAFD_WIP_REUSE:-on}"
case "$wip_reuse_mode" in on|off) ;; *) release_die "CUTEAFD_WIP_REUSE must be on or off" ;; esac
if [[ "$source_revision" != *dirty* && "$source_revision" != unknown && -z "$from_slot" ]]; then
  wip_build_key="$(
    {
      # The commit, not only its tree: the slot's BUILD_IDENTITY.json and source
      # revision stamp name it (report footers, console). Gitlinks pin submodules.
      printf 'schema 1\ncommit %s\ntree %s\n' "$source_revision" "$(git -C "$repo_root" rev-parse 'HEAD^{tree}')"
      git -C "$repo_root" ls-tree -r HEAD | awk '$2 == "commit" {print $3, $4}'
      printf 'config %s\n' "$(sha256sum "$staging_dir/cuteafd.config" | awk '{print $1}')"
      printf 'toolchain %s caches %s\n' "$wip_toolchain_hash" "${CUTEAFD_BUILD_CACHES:-on}"
      # Host CPU targets: the coordinator slot builds the ISA this host runs, so a
      # slot built on an AVX-512 host is never adopted by an AVX2-only one.
      printf 'host-isa coordinator %s spark grace\n' "$wip_coordinator_host_isa"
      printf 'spark-tp-roles %s\nbf16 %s\naudio %s\n' "$wip_spark_tp_roles" "$bf16_families" "$audio_aot"
      for key in EXL3_AOT NVFP4_AOT DSV4_MAX_CONTEXT DSV4_AOT GLM_AOT CONTEXT_SPLIT_AOT MIMO_AOT MIMO_GEOMETRIES \
        GLMF_AOT GLMF_WIDE_DECODE_ROWS QWEN4_AOT QWEN_DENSE_AOT STEP5_AOT KIMIK3_AOT COORDINATOR_EXPERT_FAMILIES EXPERT_FAMILIES GENERIC_SPARK_COUNTS \
        EXL3_BITS VISION_ATTENTION_AOT CARGO_FEATURES; do
        [[ "$key" != KIMIK3_AOT || "${CUTEAFD_WIP_KIMIK3_AOT:-OFF}" != OFF ]] || continue
        local_name="CUTEAFD_WIP_$key"
        printf '%s=%s\n' "$local_name" "${!local_name:-<default>}"
      done
      # Keyed only when on, so every existing slot keeps its build key.
      [[ "${CUTEAFD_WIP_EXLR8_AOT:-OFF}" == OFF ]] || printf 'CUTEAFD_WIP_EXLR8_AOT=%s\n' "$CUTEAFD_WIP_EXLR8_AOT"
    } | sha256sum | awk '{print $1}'
  )"
  # Each role's key adds the dev image that builds it.
  wip_coordinator_key="$(printf '%s coordinator %s\n' "$wip_build_key" \
    "$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")" | sha256sum | awk '{print $1}')"
  if ((wip_spark_leg)); then
    wip_expert_key="$(printf '%s spark-expert %s\n' "$wip_build_key" \
      "$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")" | sha256sum | awk '{print $1}')"
  fi
  echo "WIP build key: $wip_build_key (coordinator $wip_coordinator_key${wip_expert_key:+, expert $wip_expert_key})"
fi

# wip_reuse_role ROLE HOST ROOT: adopt a verified slot with this key on HOST.
wip_reuse_role() {
  local role="$1" host="$2" root="$3" key="$4" found
  [[ -n "$key" ]] || return 1
  local -a find_cmd=(python3 - find --key "$key" --role "$role" --slot "$slot" --exclude-root "$root")
  local -a adopt_cmd=(python3 - adopt --key "$key" --role "$role" --root "$root"
    --slot "$slot" --instance "${WIP_INSTANCE:-}" --wip-root "${WIP_ROOT:-}")
  if [[ -z "$host" ]]; then
    found="$("${find_cmd[@]}" <"$repo_root/scripts/build/wip-slot-reuse.py")" || return 1
    "${adopt_cmd[@]}" --source "$found" <"$repo_root/scripts/build/wip-slot-reuse.py"
  else
    found="$(ssh -o BatchMode=yes "$host" "$(printf '%q ' "${find_cmd[@]}")" <"$repo_root/scripts/build/wip-slot-reuse.py")" || return 1
    ssh -o BatchMode=yes "$host" "$(printf '%q ' "${adopt_cmd[@]}" --source "$found")" <"$repo_root/scripts/build/wip-slot-reuse.py"
  fi
}

echo "== ensuring persistent WIP development containers =="
# The serving Sparks' containers, plus the build-only seed's own build container.
wip_hosts_for_containers=("${wip_container_hosts[@]}")
ensure_local_container
remote_pids=()
((wip_spark_leg)) || wip_hosts_for_containers=()
for host in "${wip_hosts_for_containers[@]}"; do
  ensure_remote_container "$host" &
  remote_pids+=("$!")
done
remote_failed=0
for pid in "${remote_pids[@]}"; do wait "$pid" || remote_failed=1; done
((remote_failed == 0)) || release_die "one or more Spark WIP containers could not be prepared"

wip_coordinator_processes_active() {
  docker exec "$coordinator_container" bash -lc '
for pid_file in /wip/run/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && exit 0
done
exit 1
'
}

wip_expert_processes_active() {
  local host
  ((wip_spark_leg)) || return 1
  for host in "${wip_container_hosts[@]}"; do
    if ssh -o BatchMode=yes "$host" docker exec -i "$spark_container" bash -s <<'CONTAINER'
for pid_file in /wip/run/*.pid; do
  [ -f "$pid_file" ] || continue
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && exit 0
done
exit 1
CONTAINER
    then
      return 0
    fi
  done
  return 1
}

if [[ "$role" == coordinator && -n "$from_slot" ]]; then
  release_stop_wip_coordinator
  wip_coordinator_processes_active &&
    release_die "a coordinator WIP process remains active; stop it before building"
  if wip_expert_processes_active; then
    echo "== preserving resident Spark experts during coordinator-only build =="
  fi
elif wip_coordinator_processes_active || wip_expert_processes_active; then
  release_die "a WIP CUTEAFD process is active; stop it before synchronizing or building"
fi

if [[ -n "$from_slot" ]]; then
  echo "== cloning WIP slot $from_slot -> $slot =="
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/coordinator"
test ! -e "/wip/slots/$to"
CONTAINER
  if ((wip_spark_leg)); then
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
test ! -e "/wip/slots/$to"
CONTAINER
  fi
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/coordinator" "/wip/slots/$to/coordinator"
python3 - "/wip/slots/$to/coordinator" "$to" <<'RESEAL'
import hashlib, json, pathlib, sys
# A cloned role keeps its measured artifacts but was sealed for the source slot:
# re-seal it as this slot (recording the source) so launch identity checks pass.
d, to = pathlib.Path(sys.argv[1]), sys.argv[2]
raw = (d / "META.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (d / "FINGERPRINT").read_text().strip(), "cloned role seal mismatch"
meta = json.loads(raw)
meta["cloned_from"] = meta.get("cloned_from") or meta["slot"]
meta["slot"] = to
body = json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n"
(d / "META.json").write_bytes(body)
(d / "FINGERPRINT").write_text(hashlib.sha256(body).hexdigest() + "\n")
RESEAL
CONTAINER
  if ((wip_spark_leg)); then
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
python3 - "/wip/slots/$to/spark-expert" "$to" <<'RESEAL'
import hashlib, json, pathlib, sys
# A cloned role keeps its measured artifacts but was sealed for the source slot:
# re-seal it as this slot (recording the source) so launch identity checks pass.
d, to = pathlib.Path(sys.argv[1]), sys.argv[2]
raw = (d / "META.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (d / "FINGERPRINT").read_text().strip(), "cloned role seal mismatch"
meta = json.loads(raw)
meta["cloned_from"] = meta.get("cloned_from") or meta["slot"]
meta["slot"] = to
body = json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n"
(d / "META.json").write_bytes(body)
(d / "FINGERPRINT").write_text(hashlib.sha256(body).hexdigest() + "\n")
RESEAL
CONTAINER
  fi
fi

# Reuse needs each role's slot on the host that builds it: raptor and the seed.
wip_reused_coordinator=0
wip_reused_expert=0
if [[ -n "$wip_build_key" && "$wip_reuse_mode" == on ]]; then
  reuse_started=$SECONDS
  seed_wip_root=""
  ((wip_spark_leg == 0)) ||
    seed_wip_root="$(ssh -o BatchMode=yes "$seed_host" "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/wip\"}}{{.Source}}{{end}}{{end}}' '$spark_container'")"
  if [[ "$role" != expert ]] && wip_reuse_role coordinator "" "$(realpath "$state_dir")" "$wip_coordinator_key"; then
    wip_reused_coordinator=1
  fi
  if [[ "$role" != coordinator && -n "$seed_wip_root" ]] && wip_reuse_role spark-expert "$seed_host" "$seed_wip_root" "$wip_expert_key"; then
    wip_reused_expert=1
  fi
  echo "WIP phase reuse: $((SECONDS - reuse_started))s (coordinator $wip_reused_coordinator, expert $wip_reused_expert)"
fi

sync_local_source() {
  docker exec "$coordinator_container" rm -rf /wip/source.next
  docker cp -a "$staging_dir/." "$coordinator_container:/wip/source.next"
  docker exec "$coordinator_container" bash -lc 'rm -rf /wip/source && mv /wip/source.next /wip/source'
}

sync_seed_source() {
  local remote_staging
  if [[ -n "${WIP_ROOT:-}" ]]; then
    remote_staging="$WIP_ROOT/source-staging"
  else
    remote_staging="$(ssh -o BatchMode=yes "$seed_host" 'printf "%s/.cuteafd-wip-source-staging" "$HOME"')${WIP_INSTANCE:+-$WIP_INSTANCE}"
  fi
  local sync=rsync started=$SECONDS
  if [[ "$wip_seed_rdma" == 1 ]] && command -v rdmasync >/dev/null 2>&1 && ssh -o BatchMode=yes "$seed_host" 'command -v rdmasync >/dev/null'; then
    sync=rdmasync
  fi
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_staging'"
  if [[ "$sync" == rdmasync ]]; then
    rdmasync -a --delete --rdma=required --rdma-show-config "$staging_dir/" "$seed_host:$remote_staging/"
  elif [[ "$wip_seed_rdma" == 0 ]]; then
    # 10 GbE seed: its LAN address, AES-GCM and no ssh compression.
    rsync -a --delete -e "$release_rsh ${release_bulk_ssh_opts[*]} -o HostName=$wip_seed_addr -o HostKeyAlias=$seed_host" \
      "$staging_dir/" "$seed_host:$remote_staging/"
  else
    rsync -a --delete "$staging_dir/" "$seed_host:$remote_staging/"
  fi
  echo "WIP phase expert source-sync: $((SECONDS - started))s ($sync${wip_seed_addr:+ via $wip_seed_addr})"
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' rm -rf /wip/source.next && docker cp -a '$remote_staging/.' '$spark_container:/wip/source.next' && docker exec '$spark_container' bash -lc 'rm -rf /wip/source && mv /wip/source.next /wip/source'"
}

# Native/AOT outputs are cached by input key (build-wip-artifacts.sh). The
# per-instance store is <root>/cache/native-store; the host-wide store is
# ~/.cache/cuteafd/native-store/<arch>. Entries move between them as hardlinks
# (same NVMe filesystem; immutable once published) and are verified against
# their NATIVE_SHA256SUMS by the build before reuse.
# native_store_sync HOST ROOT ARCH KEYFILE import|publish (HOST "" = local)
native_store_sync() {
  local host="$1" root="$2" arch="$3" output="$4" direction="$5"
  local script='
set -euo pipefail
root="$1"; arch="$2"; output="$3"; direction="$4"
[[ -s "$root/output/$output/.native-cache-key" ]] || exit 0
read -r scope key <"$root/output/$output/.native-cache-key"
[[ "$scope" =~ ^[a-z-]+-[0-9]+$ && "$key" =~ ^[0-9a-f]{64}$ ]] || exit 0
local_entry="$root/cache/native-store/$scope/$key"
global_root="$HOME/.cache/cuteafd/native-store/$arch/$scope"
global_entry="$global_root/$key"
if [[ "$direction" == import ]]; then
  [[ -f "$global_entry/COMPLETE" && ! -e "$local_entry" ]] || exit 0
  mkdir -p "$(dirname "$local_entry")"
  stage="$local_entry.import-$$"
  cp -al "$global_entry" "$stage" 2>/dev/null || cp -a "$global_entry" "$stage"
  mv -T "$stage" "$local_entry" 2>/dev/null || rm -rf "$stage"
  echo "native cache: imported $scope/$key from the host store"
else
  [[ -f "$local_entry/COMPLETE" && ! -e "$global_entry" ]] || exit 0
  mkdir -p "$global_root"
  stage="$global_root/.publish-$key-$$"
  cp -al "$local_entry" "$stage" 2>/dev/null || cp -a "$local_entry" "$stage"
  mv -T "$stage" "$global_entry" 2>/dev/null || rm -rf "$stage"
  echo "native cache: published $scope/$key to the host store"
  # Bound the host store: the most recently used entries per scope.
  find "$global_root" -mindepth 2 -maxdepth 2 -name COMPLETE -printf "%T@ %h\n" |
    sort -rn | tail -n +"$((${CUTEAFD_WIP_NATIVE_KEEP:-16} + 1))" | cut -d" " -f2- |
    while IFS= read -r stale; do rm -rf "$stale"; done
fi
'
  if [[ -z "$host" ]]; then
    bash -c "$script" native-store "$root" "$arch" "$output" "$direction" || true
  else
    ssh -o BatchMode=yes "$host" bash -s -- "$root" "$arch" "$output" "$direction" <<<"$script" || true
  fi
}

# A cluster tier for Spark native entries on sparknest, so every seed (and a
# fresh one such as kakapo) reuses an export any seed already built: one sealed
# zstd tar per key at build-cache/wip-native/<arch>/<scope>/<key>.tar.zst,
# imported into the host store before the host-to-instance import and published
# after the host publish. The newest CUTEAFD_WIP_NEST_NATIVE_KEEP (8) per scope
# survive (CUTEAFD_WIP_NEST_NATIVE=off skips the tier). Entries are re-verified
# against NATIVE_SHA256SUMS before reuse.
# native_store_nest HOST ARCH ROOT OUTPUT import|publish
wip_nest_settle_script='
nest_settle() {
  # A file written through the mount commits shortly after close; seal and
  # replicate need the committed object.
  local i
  for ((i = 0; i < 600; i++)); do
    [[ "$(nest where --json "$1" | python3 -c "import json,sys; print(json.load(sys.stdin)[\"writing\"])")" == 0 ]] && return 0
    sleep 0.5
  done
  echo "sparknest never committed $1" >&2
  return 1
}
'
native_store_nest() {
  local host="$1" arch="$2" root="$3" output="$4" direction="$5"
  [[ "${CUTEAFD_WIP_NEST_NATIVE:-on}" == on ]] || return 0
  local script="$wip_nest_settle_script"'
set -euo pipefail
arch="$1"; root="$2"; output="$3"; direction="$4"; keep="$5"
mountpoint -q /mnt/sparknest && command -v nest >/dev/null || exit 0
[[ -s "$root/output/$output/.native-cache-key" ]] || exit 0
read -r scope key <"$root/output/$output/.native-cache-key"
[[ "$scope" =~ ^[a-z-]+-[0-9]+$ && "$key" =~ ^[0-9a-f]{64}$ ]] || exit 0
nest_root="/mnt/sparknest/build-cache/wip-native/$arch/$scope"
archive="$nest_root/$key.tar.zst"
global_root="$HOME/.cache/cuteafd/native-store/$arch/$scope"
global_entry="$global_root/$key"
if [[ "$direction" == import ]]; then
  [[ -f "$archive" && ! -f "$global_entry/COMPLETE" && ! -f "$root/cache/native-store/$scope/$key/COMPLETE" ]] || exit 0
  mkdir -p "$global_root"
  stage="$global_root/.nest-$key-$$"
  rm -rf "$stage"; mkdir -p "$stage"
  if zstd -d -q <"$archive" | tar -C "$stage" -xf - && [[ -f "$stage/COMPLETE" ]]; then
    mv -T "$stage" "$global_entry" 2>/dev/null || rm -rf "$stage"
    touch "$archive" 2>/dev/null || true
    echo "native cache: imported $scope/$key from sparknest"
  else
    rm -rf "$stage"
  fi
else
  [[ -f "$global_entry/COMPLETE" ]] || exit 0
  if [[ -f "$archive" ]]; then touch "$archive" 2>/dev/null || true; exit 0; fi
  mkdir -p "$nest_root"
  partial="$nest_root/.$key.partial-$$"
  tar -C "$global_entry" -cf - . | zstd -T0 -3 -q >"$partial"
  mv -T "$partial" "$archive"
  nest_settle "$archive" && nest seal "$archive" >/dev/null
  echo "native cache: published $scope/$key to sparknest"
  # Bound the tier: the most recently used entries per scope.
  ls -1t "$nest_root"/*.tar.zst 2>/dev/null | tail -n +"$((keep + 1))" |
    while IFS= read -r stale; do nest rm "$stale" >/dev/null || true; done
  rm -f "$nest_root"/.*.partial-* 2>/dev/null || true
fi
'
  timeout --kill-after=10 900 ssh -o BatchMode=yes "$host" bash -s -- "$arch" "$root" "$output" "$direction" \
    "${CUTEAFD_WIP_NEST_NATIVE_KEEP:-8}" <<<"$script" || true
}

# native_store_has HOST ROOT OUTPUT: the per-instance store holds a complete entry
# for the key the rust phase recorded.
native_store_has() {
  local probe='read -r scope key <"$1/output/$2/.native-cache-key" && test -f "$1/cache/native-store/$scope/$key/COMPLETE"'
  if [[ -z "$1" ]]; then
    bash -c "$probe" native-store "$2" "$3" 2>/dev/null
  else
    ssh -o BatchMode=yes "$1" bash -c "$(printf '%q' "$probe")" native-store "$2" "$3" 2>/dev/null
  fi
}

# The seed Spark's lock: rhea, moa and the out-of-pool build seed kakapo have
# their own; a pool Spark is under sparks.
wip_seed_lock() {
  case "$1" in rhea|moa|kakapo) printf '%s\n' "$1" ;; *) printf 'sparks\n' ;; esac
}
# Try a configured remote exporter first, then queue one capped local turn.
# Dispatch runs only after the Spark lease and Cargo build slot are released.
wip_coordinator_export_modes() {
  case "${CUTEAFD_WIP_COORDINATOR_EXPORT:-auto}" in
    auto) [[ -z "${CUTEAFD_EXPORT_HOST:-}" ]] || printf 'remote ' ; printf 'capped\n' ;;
    capped) [[ -z "${CUTEAFD_EXPORT_HOST:-}" ]] || printf 'remote ' ; printf 'capped\n' ;;
    remote) printf 'remote wait-remote\n' ;;
    *) release_die "CUTEAFD_WIP_COORDINATOR_EXPORT must be auto, remote or capped" ;;
  esac
}

# wip_remote_export: the native key's entry from CUTEAFD_EXPORT_HOST, through
# sparknest, then restored here with REQUIRE_HIT (no GPU turn on raptor).
wip_remote_export() {
  local host="${CUTEAFD_EXPORT_HOST:-}" key env_file started=$SECONDS
  [[ -n "$host" ]] || return 75
  read -r _ key <"$state_dir/output/coordinator/.native-cache-key" || return 75
  env_file="$(mktemp "${TMPDIR:-/tmp}/wip-export-env.XXXXXXXX")"
  printf '%s\n' "${coordinator_build[@]}" | grep -E '^CUTEAFD_[A-Z0-9_]+=' >"$env_file"
  printf 'CUTEAFD_WIP_EXPORT_LOCKS=off\n' >>"$env_file"
  echo "WIP coordinator export: remote on $host${CUTEAFD_EXPORT_HOST_ADDR:+ ($CUTEAFD_EXPORT_HOST_ADDR)}"
  local layers status=0
  layers="$(docker image inspect -f '{{json .RootFS.Layers}}' "$COORDINATOR_DOCKER_DEV" | sha256sum | awk '{print $1}')"
  "$repo_root/scripts/build/export-remote.sh" "$host" "$staging_dir" "$key" "$env_file" \
    "$COORDINATOR_DOCKER_DEV" "$layers" || status=$?
  if ((status != 0)); then
    rm -f "$env_file"
    echo "WIP coordinator export: $host unavailable (export-remote exited $status); next mode" >&2
    return 75
  fi
  rm -f "$env_file"
  native_store_nest raptor x86_64 "$(realpath "$state_dir")" coordinator import
  native_store_sync "" "$state_dir" "$(uname -m)" coordinator import
  docker exec -e CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1 "${coordinator_build[@]:2}" native || return 75
  echo "WIP phase coordinator export-remote: $((SECONDS - started))s ($host)"
}

# wip_capped_export BUILD...: the native phase on a raptor GPU we hold: one
# whose lock the caller already holds, else a fair-queued GPU turn (the first
# configured GPU, held only for the export). A one-off container on
# that GPU (the WIP container is bound to COORDINATOR_GPU), admitted by the
# host's nvidia-smi and stopped by the watchdog if it outgrows its cap.
wip_capped_export() {
  local status=0
  CUTEAFD_CAPPED_BUILD="$(printf '%q ' "$@")" \
    python3 "$repo_root/scripts/build/export-placement.py" hold --gpus "${CUTEAFD_WIP_EXPORT_GPUS:-1,0}" --wait "${CUTEAFD_WIP_EXPORT_WAIT_S:-14400}" -- \
    bash -c "$(declare -f wip_capped_export_on release_gpu_device_args cuteafd_cdi_require cuteafd_cdi_gpu_args); $(declare -p repo_root coordinator_container COORDINATOR_DOCKER_DEV); wip_capped_export_on" ||
    status=$?
  ((status == 0)) || { echo "WIP coordinator export: capped export exited $status; next mode" >&2; return 75; }
}

# wip_capped_export_on: runs under export-placement.py hold, CUTEAFD_EXPORT_GPU set.
wip_capped_export_on() (
  local placement="$repo_root/scripts/build/export-placement.py" gpu cap jobs used started=$SECONDS
  gpu="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$CUTEAFD_EXPORT_GPU" '$1 == i {print $2}')"
  [[ -n "$gpu" ]] || return 75
  read -r cap jobs used < <(python3 "$placement" admit --gpu "$gpu" --headroom-mib "${CUTEAFD_WIP_EXPORT_HEADROOM_MIB:-4096}") ||
    { echo "WIP coordinator export: gpu$CUTEAFD_EXPORT_GPU has no room for a capped export" >&2; return 75; }
  echo "WIP coordinator export: capped on gpu$CUTEAFD_EXPORT_GPU (${cap} MiB, ${jobs} jobs; ${used} MiB in use)"
  # The coordinator build's environment (its -e pairs), in a one-off container
  # on that GPU sharing the WIP container's /wip and cache mounts.
  local -a build env_args=()
  eval "build=($CUTEAFD_CAPPED_BUILD)"
  local i
  for ((i = 2; i < ${#build[@]}; i++)); do
    [[ "${build[i]}" == -e ]] && env_args+=(-e "${build[i + 1]}")
  done
  local id status=0 gpu_args name="${coordinator_container}-export-$$"
  cleanup_capped() {
    trap '' TERM HUP INT QUIT
    local names
    # A failed daemon query is never proof that CUDA has stopped.
    while true; do
      if names="$(timeout 30 docker ps -a --filter "name=^/$name$" --format '{{.Names}}')"; then
        if [[ -z "$names" ]]; then return 0; fi
        if timeout 30 docker rm -f "$name" >/dev/null; then
          if names="$(timeout 30 docker ps -a --filter "name=^/$name$" --format '{{.Names}}')" && [[ -z "$names" ]]; then return 0; fi
        fi
      fi
      echo "Capped export cleanup unconfirmed; retaining GPU lock" >&2
      sleep 60
    done
  }
  trap cleanup_capped EXIT
  trap 'exit 143' TERM INT HUP QUIT
  mapfile -t gpu_args < <(release_gpu_device_args "$gpu")
  id="$(docker run -d --name "$name" --user "$(id -u):$(id -g)" "${gpu_args[@]}" \
    -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" "${env_args[@]}" \
    -e "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-${CARGO_BUILD_JOBS:-$(nproc)}}" -e "CUTEAFD_EXPORT_JOBS=$((cap / 2688))" \
    -e CUTEAFD_EXPORT_HOST_SLOT_DIR=/export-slots -e "CUTEAFD_EXPORT_HOST_WORKERS=$(nproc)" \
    -e "CUTEAFD_EXPORT_GPU_SLOT_DIR=/export-slots/gpu$CUTEAFD_EXPORT_GPU" -e "CUTEAFD_EXPORT_GPU_BUDGET_MIB=$cap" -e "CUDA_VISIBLE_DEVICES=$gpu" -e "NVIDIA_VISIBLE_DEVICES=$gpu" \
    --volumes-from "$coordinator_container" --entrypoint sleep "$COORDINATOR_DOCKER_DEV" infinity)" || return 75
  # The watchdog counts only this container's processes (its cgroup carries the ID).
  python3 "$placement" watch --gpu "$gpu" --cap-mib "$cap" --cgroup "$id" -- \
    docker exec "$id" timeout --kill-after=30 7200 /wip/source/scripts/build/build-wip-artifacts.sh \
    /wip/source coordinator 120 /wip/build/coordinator /wip/output/coordinator native || status=$?
  cleanup_capped
  trap - EXIT TERM INT HUP QUIT
  ((status == 0)) || return "$status"
  echo "WIP phase coordinator export-capped: $((SECONDS - started))s (gpu$CUTEAFD_EXPORT_GPU)"
)

build_coordinator() {
  echo "== incrementally building coordinator slot $slot =="
  local phase="${1:-all}"
  [[ "$phase" == native ]] || sync_local_source
  local image_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
  local -a cache_env=(-e CUTEAFD_KACHE= -e CUTEAFD_SCCACHE_CUDA=0 -e "CUTEAFD_BUILD_CACHES=${CUTEAFD_BUILD_CACHES:-on}")
  if [[ "${CUTEAFD_SCCACHE_CUDA:-0}" == 1 ]]; then
    if docker exec "$coordinator_container" test -d /opt/cuteafd-sccache-cache; then
      cache_env+=(-e CUTEAFD_SCCACHE_CUDA=1)
    else
      cuteafd_compiler_cache_warn 'WIP sccache mount absent; --recreate to enable CUDA caching'
    fi
  fi
  if [[ -n "${CUTEAFD_KACHE:-}" ]]; then
    if docker exec "$coordinator_container" test -x /opt/cuteafd-kache; then
      cache_env+=(-e CUTEAFD_KACHE=/opt/cuteafd-kache)
    else
      cuteafd_compiler_cache_warn 'WIP cache mounts absent; --recreate to enable caching'
    fi
  fi
  local -a coordinator_build=(docker exec \
    "${cache_env[@]}" \
    -e "CUTEAFD_WIP_DEV_IMAGE_ID=$image_id" -e "CUTEAFD_WIP_NATIVE_CACHE=${CUTEAFD_WIP_NATIVE_CACHE:-on}" \
    -e "CUTEAFD_WIP_HOST_ISA=${wip_coordinator_host_isa:-${CUTEAFD_WIP_HOST_ISA:-}}" \
    -e "CUTEAFD_WIP_CARGO_FEATURES=${wip_cargo_features:-}" \
    -e CUTEAFD_WIP_EXPORT_LOCKS=off -e "CUTEAFD_EXPORT_JOBS=${CUTEAFD_EXPORT_JOBS:-}" -e "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-}" -e "RUST_TEST_THREADS=${RUST_TEST_THREADS:-}" -e "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-}" \
    -e "CUTEAFD_WIP_EXL3_AOT=${CUTEAFD_WIP_EXL3_AOT:-ON}" \
    -e "CUTEAFD_WIP_EXL3_BITS=${CUTEAFD_WIP_EXL3_BITS:-}" \
    -e "CUTEAFD_WIP_VISION_ATTENTION_AOT=${CUTEAFD_WIP_VISION_ATTENTION_AOT:-}" \
    -e "CUTEAFD_WIP_NVFP4_AOT=${CUTEAFD_WIP_NVFP4_AOT:-ON}" \
    -e "CUTEAFD_WIP_AUDIO_AOT=$audio_aot" \
    -e "CUTEAFD_WIP_DSV4_MAX_CONTEXT=${CUTEAFD_WIP_DSV4_MAX_CONTEXT:-1048576}" \
    -e "CUTEAFD_WIP_DSV4_AOT=${CUTEAFD_WIP_DSV4_AOT:-OFF}" \
    -e "CUTEAFD_WIP_GLM_AOT=${CUTEAFD_WIP_GLM_AOT:-OFF}" \
    -e "CUTEAFD_WIP_CONTEXT_SPLIT_AOT=${CUTEAFD_WIP_CONTEXT_SPLIT_AOT:-ON}" \
    -e "CUTEAFD_WIP_MIMO_AOT=${CUTEAFD_WIP_MIMO_AOT:-OFF}" \
    -e "CUTEAFD_WIP_MIMO_GEOMETRIES=${CUTEAFD_WIP_MIMO_GEOMETRIES:-mimo}" \
    -e "CUTEAFD_WIP_GLMF_AOT=${CUTEAFD_WIP_GLMF_AOT:-OFF}" \
    -e "CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS=${CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS:-128}" \
    -e "CUTEAFD_WIP_QWEN4_AOT=${CUTEAFD_WIP_QWEN4_AOT:-OFF}" \
    -e "CUTEAFD_WIP_QWEN_DENSE_AOT=${CUTEAFD_WIP_QWEN_DENSE_AOT:-OFF}" \
    -e "CUTEAFD_WIP_STEP5_AOT=${CUTEAFD_WIP_STEP5_AOT:-OFF}" \
    -e "CUTEAFD_WIP_KIMIK3_AOT=${CUTEAFD_WIP_KIMIK3_AOT:-OFF}" \
    ${wip_exlr8_env[@]+"${wip_exlr8_env[@]}"} \
    -e "CUTEAFD_WIP_EXPERT_FAMILIES=${CUTEAFD_WIP_COORDINATOR_EXPERT_FAMILIES:-${CUTEAFD_WIP_EXPERT_FAMILIES:-}}" \
    -e "CUTEAFD_WIP_GENERIC_SPARK_COUNTS=${CUTEAFD_WIP_GENERIC_SPARK_COUNTS:-}" \
    -e "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=$bf16_families" \
    "$coordinator_container" \
    /wip/source/scripts/build/build-wip-artifacts.sh \
    /wip/source coordinator 120 /wip/build/coordinator /wip/output/coordinator)
  # Cargo first, holding no hardware; then the bounded native/AOT export.
  # A native-cache hit restores verified outputs without touching a GPU.
  # The raptor compile holds a build slot (scripts/build/build-slot.sh; slot 0 is
  # build.lock) only while Cargo runs, when the caller asks (wip-cards does);
  # the GPU export, reuse and staging hold none.
  local rust_status=0 attempt
  # One retry: a coordinator Rust exec has twice reported 127 although the phase
  # inside the container completed (open issue). The phase is idempotent and a
  # retry over a fresh Cargo target takes seconds; a real compile error fails twice.
  if [[ "$phase" != native ]]; then
    for attempt in 1 2; do
      rust_status=0
      if [[ "${CUTEAFD_WIP_BUILD_SLOT:-off}" == on ]]; then
        "$repo_root/scripts/build/build-slot.sh" "${coordinator_build[@]}" rust || rust_status=$?
      else
        "${coordinator_build[@]}" rust || rust_status=$?
      fi
      ((rust_status != 0 && attempt == 1)) || break
      echo "WIP coordinator Rust phase exited $rust_status; retrying once" >&2
      # The retry appends to the phase table, so the first attempt's time stays recorded.
      coordinator_build=("${coordinator_build[@]:0:2}" -e CUTEAFD_WIP_PHASE_APPEND=1 "${coordinator_build[@]:2}")
    done
    ((rust_status == 0)) || release_die "coordinator WIP Rust phase failed (exit $rust_status)"
  fi
  [[ "$phase" != rust ]] || return 0
  native_store_sync "" "${state_dir:-/nonexistent}" "$(uname -m)" coordinator import
  if [[ "${wip_export_locks:-off}" != on ]]; then
    "${coordinator_build[@]}" native
  elif native_store_has "" "${state_dir:-/nonexistent}" coordinator; then
    echo "WIP coordinator export: native cache hit, no GPU turn"
    # The build re-verifies the entry; one that vanished fails rather than exporting unlocked.
    docker exec -e CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1 "${coordinator_build[@]:2}" native ||
      release_die "coordinator WIP native restore failed"
  else
    # Remote first when configured; otherwise take one bounded fair GPU turn.
    # The seed lease and Cargo slot are already released before this phase.
    local mode exported=0
    for mode in $(wip_coordinator_export_modes); do
      case "$mode" in
        remote) wip_remote_export && exported=1 ;;
        capped) wip_capped_export "${coordinator_build[@]}" && exported=1 ;;
        wait-remote)
          local remote_deadline=$((SECONDS + ${CUTEAFD_WIP_EXPORT_WAIT_S:-14400}))
          while ((SECONDS < remote_deadline)); do
            sleep 60
            if wip_remote_export; then exported=1; break; fi
          done
          ;;

      esac
      ((exported == 0)) || { echo "WIP coordinator export: done by $mode"; break; }
    done
    ((exported)) || release_die "no coordinator export mode succeeded"
  fi
  native_store_sync "" "${state_dir:-/nonexistent}" "$(uname -m)" coordinator publish
  docker exec -e "CUTEAFD_WIP_BUILD_KEY=${wip_coordinator_key:-}" ${wip_exlr8_env[@]+"${wip_exlr8_env[@]}"} \
    -e "CUTEAFD_WIP_PROGRAM_FAMILY=${wip_program_family:-}" "$coordinator_container" \
    /wip/source/scripts/build/finalize-wip-slot.sh \
    /wip/source coordinator "$slot" /wip/output/coordinator \
    "$COORDINATOR_DOCKER_DEV" "$image_id"
}

build_expert() (
  echo "== incrementally building Spark expert slot $slot on $seed_host =="
  sync_seed_source
  local image_id
  image_id="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  local cache_wrapper= cuda_cache=0
  if [[ "${CUTEAFD_SCCACHE_CUDA:-0}" == 1 ]]; then
    if ssh -o BatchMode=yes "$seed_host" docker exec "$spark_container" test -d /opt/cuteafd-sccache-cache; then
      cuda_cache=1
    else
      cuteafd_compiler_cache_warn 'Spark WIP sccache mount absent; --recreate to enable CUDA caching'
    fi
  fi
  if [[ -n "${CUTEAFD_KACHE_SPARK:-}" ]]; then
    if ssh -o BatchMode=yes "$seed_host" docker exec "$spark_container" test -x /opt/cuteafd-kache; then
      cache_wrapper=/opt/cuteafd-kache
    else
      cuteafd_compiler_cache_warn 'Spark WIP cache mounts absent; --recreate to enable caching'
    fi
  fi
  # The role list and build-scope opt-ins travel inside a single quoted remote
  # command so a `tp2;tp3` value is never split by the remote shell.
  local build_command
  build_command="docker exec -e 'CUTEAFD_WIP_DEV_IMAGE_ID=$image_id' -e 'CUTEAFD_WIP_NATIVE_CACHE=${CUTEAFD_WIP_NATIVE_CACHE:-on}' -e 'CUTEAFD_EXPORT_JOBS=${CUTEAFD_EXPORT_JOBS:-}' -e 'CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-}' -e 'RUST_TEST_THREADS=${RUST_TEST_THREADS:-}' -e 'CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-}' -e 'CUTEAFD_BUILD_CACHES=${CUTEAFD_BUILD_CACHES:-on}' -e 'CUTEAFD_SCCACHE_CUDA=$cuda_cache' -e 'CUTEAFD_KACHE=$cache_wrapper' -e 'CUTEAFD_WIP_SPARK_TP_ROLES=$wip_spark_tp_roles' -e 'CUTEAFD_WIP_EXPERT_FAMILIES=${CUTEAFD_WIP_EXPERT_FAMILIES:-}' -e 'CUTEAFD_WIP_GENERIC_SPARK_COUNTS=${CUTEAFD_WIP_GENERIC_SPARK_COUNTS:-}' -e 'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=$bf16_families' -e 'CUTEAFD_WIP_EXL3_AOT=${CUTEAFD_WIP_EXL3_AOT:-ON}' -e 'CUTEAFD_WIP_EXL3_BITS=${CUTEAFD_WIP_EXL3_BITS:-}' -e 'CUTEAFD_WIP_NVFP4_AOT=${CUTEAFD_WIP_NVFP4_AOT:-ON}' -e 'CUTEAFD_WIP_AUDIO_AOT=$audio_aot'${wip_exlr8_env[1]:+ -e '${wip_exlr8_env[1]}'} '$spark_container' timeout --kill-after=30 7200 /wip/source/scripts/build/build-wip-artifacts.sh /wip/source expert 121 /wip/build/expert /wip/output/expert"
  timeout --kill-after=30 7260 ssh -o BatchMode=yes "$seed_host" "$build_command rust" ||
    release_die "Spark WIP Rust phase on $seed_host failed (exit $?)"
  local seed_root
  seed_root="$(ssh -o BatchMode=yes "$seed_host" "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/wip\"}}{{.Source}}{{end}}{{end}}' '$spark_container'")"
  native_store_nest "$seed_host" aarch64 "$seed_root" expert import
  native_store_sync "$seed_host" "$seed_root" aarch64 expert import
  if [[ "${wip_export_locks:-off}" != on ]]; then
    timeout --kill-after=30 7260 ssh -o BatchMode=yes "$seed_host" "$build_command native"
  elif native_store_has "$seed_host" "$seed_root" expert; then
    echo "WIP Spark export: native cache hit, no lock turn"
    timeout --kill-after=30 7260 ssh -o BatchMode=yes "$seed_host" "${build_command/docker exec /docker exec -e CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1 } native" ||
      release_die "Spark WIP native restore on $seed_host failed"
  else
    # The leg's lease already holds the seed's lock: the export runs inside it.
    echo "WIP Spark export: inside the ${CUTEAFD_WIP_SEED_LEASE:-seed} lease"
    timeout --kill-after=30 7260 ssh -o BatchMode=yes "$seed_host" "$build_command native" ||
      release_die "Spark WIP export on $seed_host failed"
  fi
  native_store_sync "$seed_host" "$seed_root" aarch64 expert publish
  native_store_nest "$seed_host" aarch64 "$seed_root" expert publish
  timeout --kill-after=30 300 ssh -o BatchMode=yes "$seed_host" docker exec -e "CUTEAFD_WIP_BUILD_KEY=${wip_expert_key:-}" ${wip_exlr8_env[@]+"${wip_exlr8_env[@]}"} "$spark_container" \
    /wip/source/scripts/build/finalize-wip-slot.sh \
    /wip/source spark-expert "$slot" /wip/output/expert \
    "$SPARK_EXPERT_DOCKER_DEV" "$image_id"
)

# Role builders end here.
wip_retain_counterpart() {
# A named slot is launchable only when both role artifacts are present. A
# role-specific rebuild must retain the cloned counterpart even if its build
# helper stages or replaces only the selected role.
if [[ -n "$from_slot" && "$role" == expert ]]; then
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
if [ ! -d "/wip/slots/$to/coordinator" ]; then
  test -d "/wip/slots/$from/coordinator"
  mkdir -p "/wip/slots/$to"
  cp -a "/wip/slots/$from/coordinator" "/wip/slots/$to/coordinator"
python3 - "/wip/slots/$to/coordinator" "$to" <<'RESEAL'
import hashlib, json, pathlib, sys
# A cloned role keeps its measured artifacts but was sealed for the source slot:
# re-seal it as this slot (recording the source) so launch identity checks pass.
d, to = pathlib.Path(sys.argv[1]), sys.argv[2]
raw = (d / "META.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (d / "FINGERPRINT").read_text().strip(), "cloned role seal mismatch"
meta = json.loads(raw)
meta["cloned_from"] = meta.get("cloned_from") or meta["slot"]
meta["slot"] = to
body = json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n"
(d / "META.json").write_bytes(body)
(d / "FINGERPRINT").write_text(hashlib.sha256(body).hexdigest() + "\n")
RESEAL
fi
CONTAINER
elif [[ -n "$from_slot" && "$role" == coordinator ]] && ((wip_spark_leg)); then
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
if [ ! -d "/wip/slots/$to/spark-expert" ]; then
  test -d "/wip/slots/$from/spark-expert"
  mkdir -p "/wip/slots/$to"
  cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
python3 - "/wip/slots/$to/spark-expert" "$to" <<'RESEAL'
import hashlib, json, pathlib, sys
# A cloned role keeps its measured artifacts but was sealed for the source slot:
# re-seal it as this slot (recording the source) so launch identity checks pass.
d, to = pathlib.Path(sys.argv[1]), sys.argv[2]
raw = (d / "META.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (d / "FINGERPRINT").read_text().strip(), "cloned role seal mismatch"
meta = json.loads(raw)
meta["cloned_from"] = meta.get("cloned_from") or meta["slot"]
meta["slot"] = to
body = json.dumps(meta, indent=2, sort_keys=True).encode() + b"\n"
(d / "META.json").write_bytes(body)
(d / "FINGERPRINT").write_text(hashlib.sha256(body).hexdigest() + "\n")
RESEAL
fi
CONTAINER
fi

}

# How a sealed Spark slot reaches the serving Sparks. A seed on the RoCE fabric
# streams to each (rdmapipe). A seed without RDMA writes one sealed zstd tar to
# sparknest, which replicates it to one serving Spark over its 10 GbE link and
# then on to the rest over RoCE; each Spark extracts from its local copy. With
# sparknest unmounted (or CUTEAFD_WIP_SEED_TRANSFER=stream) that seed relays one
# compressed stream to the first target, which fans out.
wip_slot_transfer_mode() {
  case "$wip_seed_transfer" in
    nest) printf 'nest\n'; return ;;
    stream) [[ "$wip_seed_rdma" == 1 ]] && printf 'stream\n' || printf 'relay\n'; return ;;
  esac
  [[ "$wip_seed_rdma" == 0 ]] || { printf 'stream\n'; return; }
  local host
  for host in "$seed_host" "${wip_target_hosts[@]}"; do
    ssh -o BatchMode=yes "$host" 'mountpoint -q /mnt/sparknest && command -v nest >/dev/null' ||
      { printf 'relay\n'; return; }
  done
  printf 'nest\n'
}

# distribute_expert_slot_from SRC HOST...: stream SRC's sealed slot to each HOST.
distribute_expert_slot_from() {
  local src="$1" host
  shift
  local -a pids=()
  for host in "$@"; do
    (
      set -o pipefail
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'rm -rf /wip/incoming/$slot.spark-expert && mkdir -p /wip/incoming/$slot.spark-expert'"
      stream_between_hosts "$src" "docker exec '$spark_container' tar -C '/wip/slots/$slot/spark-expert' -cf - ." \
        "$host" "docker exec -i '$spark_container' tar -C '/wip/incoming/$slot.spark-expert' -xf -"
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'mkdir -p /wip/slots/$slot; rm -rf /wip/slots/$slot/spark-expert; mv /wip/incoming/$slot.spark-expert /wip/slots/$slot/spark-expert'"
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark WIP slot distribution from $src failed"
}

# The instance's sparknest build area; wip-cards --cleanup removes it with the instance.
wip_nest_slot_dir() {
  printf '/mnt/sparknest/build-cache/wip/%s/%s\n' "${WIP_INSTANCE:-_default}" "$slot"
}

# distribute_expert_slot_nest FINGERPRINT HOST...
distribute_expert_slot_nest() {
  local fingerprint="$1" dir archive
  shift
  dir="$(wip_nest_slot_dir)"
  archive="$dir/$fingerprint.tar.zst"
  local started=$SECONDS
  # One archive per slot: an earlier fingerprint's copies go first.
  timeout --kill-after=10 1800 ssh -o BatchMode=yes "$seed_host" bash -s -- "$dir" "$archive" "$spark_container" "$slot" \
    <<<"$wip_nest_settle_script"'
set -euo pipefail
dir="$1"; archive="$2"; container="$3"; slot="$4"
mkdir -p "$dir"
for old in "$dir"/*.tar.zst; do
  [[ -e "$old" && "$old" != "$archive" ]] || continue
  nest rm "$old" >/dev/null
done
if [[ ! -f "$archive" ]]; then
  partial="$dir/.$(basename "$archive").partial-$$"
  docker exec "$container" tar -C "/wip/slots/$slot/spark-expert" -cf - . | zstd -T0 -1 -q >"$partial"
  mv -T "$partial" "$archive"
fi
nest_settle "$archive"
nest seal "$archive" >/dev/null
' || release_die "could not write the Spark WIP slot to sparknest from $seed_host"
  echo "WIP phase expert nest-write: $((SECONDS - started))s ($(ssh -o BatchMode=yes "$seed_host" "stat -c %s '$archive'") bytes)"
  started=$SECONDS
  # The first copy crosses the seed's link; the rest source from a RoCE host.
  timeout --kill-after=10 1800 ssh -o BatchMode=yes "$seed_host" "nest replicate --hosts '$1' --wait '$archive' >/dev/null" ||
    release_die "sparknest could not replicate $archive to $1"
  echo "WIP phase expert nest-first-copy: $((SECONDS - started))s ($1)"
  if (($# > 1)); then
    started=$SECONDS
    timeout --kill-after=10 1800 ssh -o BatchMode=yes "$seed_host" \
      "nest replicate --hosts '$(IFS=,; echo "${*:2}")' --wait '$archive' >/dev/null" ||
      release_die "sparknest could not replicate $archive to ${*:2}"
    echo "WIP phase expert nest-fanout: $((SECONDS - started))s (${*:2})"
  fi
  started=$SECONDS
  local host
  local -a pids=()
  for host in "$@"; do
    ssh -o BatchMode=yes "$host" bash -s -- "$archive" "$spark_container" "$slot" <<'EXTRACT' &
set -euo pipefail
archive="$1"; container="$2"; slot="$3"
docker exec "$container" bash -lc "rm -rf /wip/incoming/$slot.spark-expert && mkdir -p /wip/incoming/$slot.spark-expert"
zstd -d -q <"$archive" | docker exec -i "$container" tar -C "/wip/incoming/$slot.spark-expert" -xf -
docker exec "$container" bash -lc "mkdir -p /wip/slots/$slot; rm -rf /wip/slots/$slot/spark-expert; mv /wip/incoming/$slot.spark-expert /wip/slots/$slot/spark-expert"
EXTRACT
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark WIP slot extraction from sparknest failed"
  echo "WIP phase expert nest-extract: $((SECONDS - started))s"
}

distribute_expert_slot() {
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' test -s '/wip/slots/$slot/spark-expert/FINGERPRINT'" ||
    return 0
  local seed_fingerprint host
  local -a targets=()
  seed_fingerprint="$(ssh -o BatchMode=yes "$seed_host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
  for host in "${wip_target_hosts[@]}"; do
    [[ "$(ssh -o BatchMode=yes "$host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT' 2>/dev/null" || true)" == "$seed_fingerprint" ]] ||
      targets+=("$host")
  done
  ((${#targets[@]})) || { echo "Spark WIP slot $slot is already on every target host"; return 0; }
  local started=$SECONDS mode
  mode="$(wip_slot_transfer_mode)"
  echo "== distributing Spark WIP slot $slot from $seed_host to ${targets[*]} ($mode) =="
  case "$mode" in
    nest) distribute_expert_slot_nest "$seed_fingerprint" "${targets[@]}" ;;
    relay)
      # One compressed crossing of the seed's 10 GbE link, then fan out on RoCE.
      distribute_expert_slot_from "$seed_host" "${targets[0]}"
      ((${#targets[@]} == 1)) || distribute_expert_slot_from "${targets[0]}" "${targets[@]:1}"
      ;;
    stream) distribute_expert_slot_from "$seed_host" "${targets[@]}" ;;
  esac
  echo "WIP phase expert distribute: $((SECONDS - started))s ($mode, ${#targets[@]} hosts)"
  local expected actual
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
  for host in "${wip_target_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
    [[ "$actual" == "$expected" ]] || release_die "$host received a mismatched WIP slot fingerprint"
  done
}

wip_check_expert_readiness() {
if ((wip_spark_leg)) && [[ "$role" != coordinator || -n "$from_slot" ]]; then
ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$slot" <<'CONTAINER'
set -euo pipefail
slot="$1"
test -s "/wip/slots/$slot/spark-expert/FINGERPRINT"
test -s "/wip/slots/$slot/spark-expert/workspace/cuteafd.config"
CONTAINER
fi
}

# The seed container was stopped when its lease ended. Distribution uses only
# tar/file operations; stop the seed again even when transfer fails.
wip_distribute_sealed_expert() (
  ((wip_spark_leg)) || return 0
  local running
  running="$(ssh -o BatchMode=yes "$seed_host" "docker inspect -f '{{.State.Running}}' '$spark_container'")"
  if [[ "$running" != true ]]; then
    ssh -o BatchMode=yes "$seed_host" "docker start '$spark_container' >/dev/null"
    trap 'ssh -o BatchMode=yes "$seed_host" "docker stop -t 10 \"$spark_container\" >/dev/null"' EXIT
  fi
  wip_check_expert_readiness
  distribute_expert_slot
)

wip_build_selected_roles() {
  wip_build_coordinator=0
  wip_build_expert=0
  [[ "$role" == expert || "$wip_reused_coordinator" == 1 ]] || wip_build_coordinator=1
  [[ "$role" == coordinator || "$wip_reused_expert" == 1 ]] || wip_build_expert=1
  # No hardware or seed lease during Cargo; its build-slot helper exits here.
  ((wip_build_coordinator == 0)) || build_coordinator rust
  if ((wip_build_expert)); then
    wip_build_resource_acquire
    build_expert
  fi
  wip_retain_counterpart
  wip_resource_complete=1
  wip_build_resource_release
  echo "WIP build resources released before Spark distribution and coordinator export"
  wip_distribute_sealed_expert
  ((wip_build_coordinator == 0)) || build_coordinator native
}

wip_build_selected_roles

# wip-slot-readiness:start
# A fresh role-only build succeeds with its own artifacts. Clones still require
# the retained counterpart, and a both-role build must prove the complete pair.
if [[ "$role" != expert || -n "$from_slot" ]]; then
docker exec -i "$coordinator_container" bash -s -- "$slot" <<'CONTAINER'
set -euo pipefail
slot="$1"
test -s "/wip/slots/$slot/coordinator/FINGERPRINT"
test -s "/wip/slots/$slot/coordinator/workspace/cuteafd.config"
CONTAINER
fi
# wip-slot-readiness:end

release_record_wip_slot "$slot"
if [[ "$role" == both || -n "$from_slot" ]]; then
  echo "WIP slot '$slot' is ready. Launch it with: ./run.sh --wip '$slot' --restart"
else
  echo "WIP slot '$slot': $role artifacts built. Build the other role before launching a new slot."
fi
