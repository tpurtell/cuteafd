#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/build/compiler-cache.sh"
source "$repo_root/scripts/build/build-caches.sh"
cuteafd_build_cache_defaults
wip_toolchain_hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"
audio_aot="${CUTEAFD_WIP_AUDIO_AOT:-OFF}"
case "$audio_aot" in ON|OFF) ;; *) release_die "CUTEAFD_WIP_AUDIO_AOT must be ON or OFF, got: $audio_aot" ;; esac
bf16_families="${CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES:-}"
bf16_family_pattern='^(mimo|mimop|mimof|glm|glmf|qwen4)(;(mimo|mimop|mimof|glm|glmf|qwen4))*$'
[[ -z "$bf16_families" || "$bf16_families" =~ $bf16_family_pattern ]] ||
  release_die "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES must be a semicolon list of mimo, mimop, mimof, glm, glmf or qwen4"

usage() {
  cat <<'EOF'
Usage: ./wip.sh [--slot NAME] [--role coordinator|expert|both]
                [--from-slot NAME] [--config FILE] [--recreate] [--dry-run]

Synchronizes the current checkout into persistent development containers and
incrementally builds a named WIP slot. The coordinator container builds and
runs coordinator slots. The first configured Spark builds Spark slots, which
are copied directly and concurrently to the other persistent Spark WIP
containers.
An explicit SPARK_TP=2/3/6 topology builds the matching opt-in SM121 expert role;
CUTEAFD_WIP_SPARK_TP_ROLES=tp2;tp3;tp6 overrides that selection. The default
configuration builds no extra role and keeps the historical Spark TP4 shard.
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
case "$role" in coordinator|expert|both) ;; *) release_die "--role must be coordinator, expert, or both" ;; esac

release_load_config "$config"
case "${CUTEAFD_RELEASE_DEV_IMAGE_SOURCE:-registry}" in registry|build) ;; *) release_die "CUTEAFD_RELEASE_DEV_IMAGE_SOURCE must be registry or build" ;; esac
release_validate_wip_instance
release_need docker
release_need ssh
release_need rsync
release_need python3
release_need sha256sum
release_need install
release_need nvidia-smi

case "${CUTEAFD_WIP_EXPORT_LOCKS:-off}" in on|off) ;; *) release_die "CUTEAFD_WIP_EXPORT_LOCKS must be on or off" ;; esac
wip_export_locks=off
if [[ "${CUTEAFD_WIP_EXPORT_LOCKS:-off}" == on ]]; then
  wip_export_locks=on
  [[ "$COORDINATOR_GPU" =~ ^[01]$ ]] || release_die "export locking needs COORDINATOR_GPU=0 or 1"
  # Only the selected GPU is exposed to the coordinator build container.
  wip_export_lock="$HOME/.cache/cuteafd/gpu${COORDINATOR_GPU}.lock"
fi
mapfile -t wip_hosts < <(release_spark_values HOST)
((${#wip_hosts[@]})) || release_die "configuration has no active Spark hosts"
[[ "${#wip_hosts[@]}" == "$SPARK_COUNT" ]] ||
  release_die "SPARK_COUNT=$SPARK_COUNT does not match ${#wip_hosts[@]} configured Spark hosts"
seed_host="${wip_hosts[0]}"
wip_target_hosts=("${wip_hosts[@]:1}")

# Opt-in replicated-group Spark expert roles for the WIP slot. The default and
# explicit TP4xEP1 build no extra role; an explicit TP2/TP3/TP6 topology selects
# the matching SM121 role.
wip_spark_tp_roles="${CUTEAFD_WIP_SPARK_TP_ROLES:-}"
if [[ -z "$wip_spark_tp_roles" ]] && release_spark_topology_explicit; then
  case "$SPARK_TP" in
    2) wip_spark_tp_roles=tp2 ;;
    3) wip_spark_tp_roles=tp3 ;;
    4) wip_spark_tp_roles= ;;
    6) wip_spark_tp_roles=tp6 ;;
  esac
fi
if [[ -n "$wip_spark_tp_roles" ]]; then
  IFS=';' read -ra wip_spark_tp_role_list <<<"$wip_spark_tp_roles"
  for wip_spark_tp_role in "${wip_spark_tp_role_list[@]}"; do
    case "$wip_spark_tp_role" in
      tp2|tp3|tp6) ;;
      *) release_die "CUTEAFD_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got: $wip_spark_tp_role" ;;
    esac
  done
  unset wip_spark_tp_role wip_spark_tp_role_list
fi

if ((dry_run)); then
  echo "WIP dry-run passed; no container, image, SSH or build operation was performed."
  echo "  config: $RELEASE_CONFIG"
  echo "  slot: $slot (role $role)"
  echo "  containers: $(release_wip_container coordinator), $(release_wip_container spark-expert)"
  echo "  WIP root: ${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}"
  echo "  Spark hosts (${#wip_hosts[@]}): $(IFS=,; echo "${wip_hosts[*]}")"
  echo "  seed host: $seed_host"
  echo "  topology: tp=$(release_spark_tp) ep=$(release_spark_ep) explicit=$(release_spark_topology_explicit && echo 1 || echo 0)"
  echo "  V41 Spark expert roles: ${wip_spark_tp_roles:-<legacy TP4 only>}"
  echo "  EXL3 AOT: ${CUTEAFD_WIP_EXL3_AOT:-ON}; NVFP4 AOT: ${CUTEAFD_WIP_NVFP4_AOT:-ON}"
  echo "  Audio AOT: $audio_aot"
  cuteafd_build_cache_docker_args "${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}" /wip/home "$wip_toolchain_hash" dry >/dev/null
  echo "  Spark cache plan (host HOME template; states/admission and uid rechecked on each Spark, no SSH):"
  cuteafd_build_cache_docker_args "${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}" /wip/home "$wip_toolchain_hash" dry aarch64 >/dev/null
  exit 0
fi

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
release_resolve_coordinator_gpu_identity

coordinator_container="$(release_wip_container coordinator)"
spark_container="$(release_wip_container spark-expert)"
seed_host="$SPARK_0_HOST"
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
build_supervise 'WIP dev image' "$dev_image_logs" 0 \
  coordinator ensure_local_image cancel_local_dev_image coordinator.log \
  expert ensure_seed_image cancel_seed_dev_image expert.log

# stream_between_hosts SRC SRC_CMD DST DST_CMD: SRC_CMD's stdout into DST_CMD's stdin, over
# rdmapipe when both hosts have it, otherwise a plain ssh pipe relayed through this host (the
# fallback scripts/build/build-dev-images.sh uses for the development images).
stream_between_hosts() {
  local src="$1" src_cmd="$2" dst="$3" dst_cmd="$4"
  if ssh -o BatchMode=yes "$src" 'command -v rdmapipe >/dev/null' &&
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
  echo "== concurrently distributing Spark development image from $seed_host =="
  local -a pids=()
  for host in "${targets[@]}"; do
    (
      set -o pipefail
      stream_between_hosts "$seed_host" "docker image save '$SPARK_EXPERT_DOCKER_DEV'" "$host" 'docker image load'
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark development image distribution failed"
}

remove_wip_containers() {
  docker rm -f "$coordinator_container" >/dev/null 2>&1 || true
  local host
  for host in "${wip_hosts[@]}"; do
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
  for host in "${wip_hosts[@]}"; do
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
    rm -rf "$state_dir/$part"
    for host in "${wip_hosts[@]}"; do
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
    [[ "$coordinator_device" == "$RELEASE_COORDINATOR_GPU_UUID" ]] ||
      release_die "$coordinator_container is not bound to the configured physical GPU; rerun ./wip.sh --recreate"
  fi
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker image inspect -f '{{.Id}}' '$SPARK_EXPERT_DOCKER_DEV'")"
  for host in "${wip_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker inspect -f '{{.Image}}' '$spark_container' 2>/dev/null || true")"
    [[ -z "$actual" || "$actual" == "$expected" ]] ||
      release_die "$host $spark_container uses an old development image; rerun ./wip.sh --recreate"
  done
}

preflight_existing_container_images
distribute_spark_dev_image

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
    if [[ "${wip_export_locks:-off}" == on ]]; then
      [[ "$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/wip-export-gpu.lock"}}{{.Source}}{{end}}{{end}}' "$coordinator_container")" == "$wip_export_lock" ]] ||
        release_die "export lock mount missing or changed; recreate this task's WIP container"
    fi
    docker exec "$coordinator_container" mkdir -p /wip/build /wip/output /wip/slots /wip/incoming /wip/run /wip/cache
    return
  fi
  local -a args=(
    run -d --name "$coordinator_container" --restart no
    --label "io.cuteafd.build-caches=${CUTEAFD_BUILD_CACHES:-on}:$wip_toolchain_hash"
    --user "$(id -u):$(id -g)"
    -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
    --gpus device="$RELEASE_COORDINATOR_GPU_UUID"
    --net=host --ipc=host --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1 --cap-add IPC_LOCK
    -v "$hf_home:$hf_home:ro" -v "$hf_home:/root/.cache/huggingface:ro"
    -e "WIP_INSTANCE=${WIP_INSTANCE:-}" -e "WIP_ROOT=${WIP_ROOT:-}"
    -e HF_HOME="$hf_home"
    -e CUDA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
    -e NVIDIA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
  )
  if [[ "${wip_export_locks:-off}" == on ]]; then
    mkdir -p "$(dirname "$wip_export_lock")"
    touch "$wip_export_lock"
    args+=(-v "$wip_export_lock:/wip-export-gpu.lock" -e CUTEAFD_WIP_EXPORT_LOCKS=on -e CUTEAFD_WIP_EXPORT_LOCK_FILES=/wip-export-gpu.lock)
  fi
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
         ! scp -q "$repo_root/scripts/build/compiler-cache.sh" "$repo_root/scripts/build/assert-build-filesystem.py" "$repo_root/scripts/build/build-caches.sh" "$repo_root/scripts/build/build-cache-plan.py" "$host:$cache_staging/scripts/build/"; then
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
args=(
  run -d --name "$container" --restart no
  --label "io.cuteafd.build-caches=${CUTEAFD_BUILD_CACHES:-on}:${12:?}"
  --user "$(id -u):$(id -g)"
  -e HOME=/wip/home -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
  --gpus all --net=host --ipc=host --security-opt seccomp=unconfined
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

echo "== ensuring persistent WIP development containers =="
ensure_local_container
remote_pids=()
for host in "${wip_hosts[@]}"; do
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
  for host in "${wip_hosts[@]}"; do
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
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
test ! -e "/wip/slots/$to"
CONTAINER
  docker exec -i "$coordinator_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/coordinator" "/wip/slots/$to/coordinator"
CONTAINER
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
test -d "/wip/slots/$from/spark-expert"
mkdir -p "/wip/slots/$to"
cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
CONTAINER
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
  local sync=rsync
  if command -v rdmasync >/dev/null 2>&1 && ssh -o BatchMode=yes "$seed_host" 'command -v rdmasync >/dev/null'; then
    sync=rdmasync
  fi
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_staging'"
  if [[ "$sync" == rdmasync ]]; then
    rdmasync -a --delete --rdma=required --rdma-show-config "$staging_dir/" "$seed_host:$remote_staging/"
  else
    rsync -a --delete "$staging_dir/" "$seed_host:$remote_staging/"
  fi
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' rm -rf /wip/source.next && docker cp -a '$remote_staging/.' '$spark_container:/wip/source.next' && docker exec '$spark_container' bash -lc 'rm -rf /wip/source && mv /wip/source.next /wip/source'"
}

build_coordinator() {
  echo "== incrementally building coordinator slot $slot =="
  sync_local_source
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
  docker exec \
    "${cache_env[@]}" \
    -e "CUTEAFD_WIP_EXPORT_LOCKS=${wip_export_locks:-off}" -e "CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-}" -e "RUST_TEST_THREADS=${RUST_TEST_THREADS:-}" -e "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-}" \
    -e "CUTEAFD_WIP_EXL3_AOT=${CUTEAFD_WIP_EXL3_AOT:-ON}" \
    -e "CUTEAFD_WIP_NVFP4_AOT=${CUTEAFD_WIP_NVFP4_AOT:-ON}" \
    -e "CUTEAFD_WIP_AUDIO_AOT=$audio_aot" \
    -e "CUTEAFD_WIP_DSV4_MAX_CONTEXT=${CUTEAFD_WIP_DSV4_MAX_CONTEXT:-1048576}" \
    -e "CUTEAFD_WIP_DSV4_AOT=${CUTEAFD_WIP_DSV4_AOT:-OFF}" \
    -e "CUTEAFD_WIP_GLM_AOT=${CUTEAFD_WIP_GLM_AOT:-OFF}" \
    -e "CUTEAFD_WIP_MIMO_AOT=${CUTEAFD_WIP_MIMO_AOT:-OFF}" \
    -e "CUTEAFD_WIP_MIMO_GEOMETRIES=${CUTEAFD_WIP_MIMO_GEOMETRIES:-mimo}" \
    -e "CUTEAFD_WIP_GLMF_AOT=${CUTEAFD_WIP_GLMF_AOT:-OFF}" \
    -e "CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS=${CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS:-128}" \
    -e "CUTEAFD_WIP_QWEN4_AOT=${CUTEAFD_WIP_QWEN4_AOT:-OFF}" \
    -e "CUTEAFD_WIP_EXPERT_FAMILIES=${CUTEAFD_WIP_EXPERT_FAMILIES:-}" \
    -e "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=$bf16_families" \
    "$coordinator_container" \
    /wip/source/scripts/build/build-wip-artifacts.sh \
    /wip/source coordinator 120 /wip/build/coordinator /wip/output/coordinator
  docker exec "$coordinator_container" \
    /wip/source/scripts/build/finalize-wip-slot.sh \
    /wip/source coordinator "$slot" /wip/output/coordinator \
    "$COORDINATOR_DOCKER_DEV" "$image_id"
}

build_expert() {
  echo "== incrementally building Spark expert slot $slot on $seed_host =="
  if [[ "${wip_export_locks:-off}" == on ]]; then
    local spark_lock
    case "$seed_host" in rhea|moa) spark_lock="$HOME/.cache/cuteafd/$seed_host.lock" ;; *) spark_lock="$HOME/.cache/cuteafd/sparks.lock" ;; esac
    flock -n "$spark_lock" true || release_die "refusing Spark export on $seed_host: $spark_lock is held (choose an idle seed)"
  fi
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
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec -e 'CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-}' -e 'RUST_TEST_THREADS=${RUST_TEST_THREADS:-}' -e 'CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-}' -e 'CUTEAFD_BUILD_CACHES=${CUTEAFD_BUILD_CACHES:-on}' -e 'CUTEAFD_SCCACHE_CUDA=$cuda_cache' -e 'CUTEAFD_KACHE=$cache_wrapper' -e 'CUTEAFD_WIP_SPARK_TP_ROLES=$wip_spark_tp_roles' -e 'CUTEAFD_WIP_EXPERT_FAMILIES=${CUTEAFD_WIP_EXPERT_FAMILIES:-}' -e 'CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=$bf16_families' -e 'CUTEAFD_WIP_EXL3_AOT=${CUTEAFD_WIP_EXL3_AOT:-ON}' -e 'CUTEAFD_WIP_NVFP4_AOT=${CUTEAFD_WIP_NVFP4_AOT:-ON}' -e 'CUTEAFD_WIP_AUDIO_AOT=$audio_aot' '$spark_container' /wip/source/scripts/build/build-wip-artifacts.sh /wip/source expert 121 /wip/build/expert /wip/output/expert"
  ssh -o BatchMode=yes "$seed_host" docker exec "$spark_container" \
    /wip/source/scripts/build/finalize-wip-slot.sh \
    /wip/source spark-expert "$slot" /wip/output/expert \
    "$SPARK_EXPERT_DOCKER_DEV" "$image_id"
}

case "$role" in
  coordinator) build_coordinator ;;
  expert) build_expert ;;
  both)
    build_coordinator &
    coordinator_build_pid=$!
    if ! build_expert; then
      wait "$coordinator_build_pid" || true
      release_die "Spark WIP build failed"
    fi
    wait "$coordinator_build_pid" || release_die "coordinator WIP build failed"
    ;;
esac

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
fi
CONTAINER
elif [[ -n "$from_slot" && "$role" == coordinator ]]; then
  ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$from_slot" "$slot" <<'CONTAINER'
set -euo pipefail
from="$1"
to="$2"
if [ ! -d "/wip/slots/$to/spark-expert" ]; then
  test -d "/wip/slots/$from/spark-expert"
  mkdir -p "/wip/slots/$to"
  cp -a "/wip/slots/$from/spark-expert" "/wip/slots/$to/spark-expert"
fi
CONTAINER
fi

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
if [[ "$role" != coordinator || -n "$from_slot" ]]; then
ssh -o BatchMode=yes "$seed_host" docker exec -i "$spark_container" bash -s -- "$slot" <<'CONTAINER'
set -euo pipefail
slot="$1"
test -s "/wip/slots/$slot/spark-expert/FINGERPRINT"
test -s "/wip/slots/$slot/spark-expert/workspace/cuteafd.config"
CONTAINER
fi
# wip-slot-readiness:end

distribute_expert_slot() {
  ssh -o BatchMode=yes "$seed_host" \
    "docker exec '$spark_container' test -s '/wip/slots/$slot/spark-expert/FINGERPRINT'" ||
    return 0
  echo "== concurrently distributing Spark WIP slot $slot from $seed_host =="
  local -a pids=()
  local host
  for host in "${wip_target_hosts[@]}"; do
    (
      set -o pipefail
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'rm -rf /wip/incoming/$slot.spark-expert && mkdir -p /wip/incoming/$slot.spark-expert'"
      stream_between_hosts "$seed_host" "docker exec '$spark_container' tar -C '/wip/slots/$slot/spark-expert' -cf - ." \
        "$host" "docker exec -i '$spark_container' tar -C '/wip/incoming/$slot.spark-expert' -xf -"
      ssh -o BatchMode=yes "$host" \
        "docker exec '$spark_container' bash -lc 'mkdir -p /wip/slots/$slot; rm -rf /wip/slots/$slot/spark-expert; mv /wip/incoming/$slot.spark-expert /wip/slots/$slot/spark-expert'"
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || release_die "Spark WIP slot distribution failed"
  local expected actual
  expected="$(ssh -o BatchMode=yes "$seed_host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
  for host in "${wip_target_hosts[@]}"; do
    actual="$(ssh -o BatchMode=yes "$host" "docker exec '$spark_container' cat '/wip/slots/$slot/spark-expert/FINGERPRINT'")"
    [[ "$actual" == "$expected" ]] || release_die "$host received a mismatched WIP slot fingerprint"
  done
}

distribute_expert_slot
release_record_wip_slot "$slot"
if [[ "$role" == both || -n "$from_slot" ]]; then
  echo "WIP slot '$slot' is ready. Launch it with: ./run.sh --wip '$slot' --restart"
else
  echo "WIP slot '$slot': $role artifacts built. Build the other role before launching a new slot."
fi
