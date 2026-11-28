#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/build/compiler-cache.sh"
source "$repo_root/scripts/build/build-caches.sh"
cuteafd_build_cache_defaults
native_build_jobs="${CUTEAFD_RELEASE_NATIVE_BUILD_JOBS:-}"
[[ -z "$native_build_jobs" || "$native_build_jobs" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_NATIVE_BUILD_JOBS must be a positive integer"
remote_min_free_gib="${CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB-60}"
[[ "$remote_min_free_gib" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB must be a positive integer"
# Keep the GiB-to-KiB comparison inside Bash's signed integer range.
((${#remote_min_free_gib} <= 9)) ||
  release_die "CUTEAFD_RELEASE_REMOTE_MIN_FREE_GIB is too large"
native_build_env_args=()
[[ -z "$native_build_jobs" ]] || native_build_env_args=(-e "CMAKE_BUILD_PARALLEL_LEVEL=$native_build_jobs")

usage() {
  cat <<'EOF'
Usage: ./build.sh [--config FILE] [--spark-hosts HOST,...] [--scope SCOPE] [--dry-run]

The coordinator and Spark legs build concurrently: the coordinator image locally
and the Spark image natively over SSH on the first configured Spark. Set
CUTEAFD_RELEASE_SEQUENTIAL=1 to build coordinator then Spark for debugging.
Each leg has its own log file; both must finish before artifacts are exported.
It exports both release artifact sets to dist/
and distributes the Spark inference image to all configured Spark hosts.
Use --spark-hosts ostrich,dodo to build and distribute only on available hosts;
this does not change the serving topology.
Release images derive the Spark TP roles required by the shipping matrix.
CUTEAFD_RELEASE_SPARK_TP_ROLES adds extra roles; --scope selects a smaller
family/card set or explicit for a raw-knob experimental build.
--dry-run validates the configuration, host set and role plan without touching
Docker, SSH, submodules, hardware locks or any image.
The build takes ~/.cache/cuteafd/build.lock for its whole run, so two release
builds serialize; CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS bounds that wait
(default 1200). It never takes sparks.lock or gpu1.lock, and it never touches
a GPU: both coordinator (SM120) and Spark (SM121) AOT exports run CPU-only
(CUDA_VISIBLE_DEVICES empty). Set CUTEAFD_RELEASE_EXPORT_TIMEOUT_SECONDS
(default 7200) to bound each export.
Set CUTEAFD_RELEASE_NATIVE_BUILD_JOBS to a positive integer to bound concurrent
native compile/export jobs on both hosts (e.g. 1 for the full family matrix on
unified-memory Sparks). Unset keeps the build tool's existing concurrency.
Set CUTEAFD_RELEASE_SSH_CONFIG to an ssh config file that every remote step should
use (default empty: stock OpenSSH resolution, so a build host's ~/.ssh/config keeps
working, with BatchMode forced either way). Pass /dev/null to discard a system
ssh_config that OpenSSH refuses to read - but -F replaces the whole config chain, so
that also drops your own host aliases; prefer a file containing just
    Include ~/.ssh/config
and pass that instead. The setting reaches ssh, rsync and rdmasync alike, and is
exported to child build scripts.
Set CUTEAFD_RELEASE_BUILD_ROOT to one unique absolute path, on a real writable
filesystem with room for a source copy plus objects, to hold the container build
roots of both roles instead of their default /tmp scratch. The same path is bound
into each container, created and filesystem-guarded on the coordinator host and the
seed Spark before any compile, and it is per-task: never share one between builds.
CUTEAFD_RELEASE_REMOTE_BUILD_DIR selects where the Spark seed host stages the source
tree and builds the expert images (default: a cuteafd-release-build directory in the
seed host's own home). Use a fresh one per release so a previous staging tree cannot
be reused.
CUTEAFD_RELEASE_SSH_CONFIG, CUTEAFD_RELEASE_BUILD_ROOT and
CUTEAFD_RELEASE_REMOTE_BUILD_DIR must each be a canonical absolute path built from
letters, digits, dot, underscore, plus and minus - no spaces, dot segments, trailing
slashes or shell metacharacters - because they reach remote shells and bind mounts.
Both artifact containers run as the invoking user (--user UID:GID) with USER,
LOGNAME, HOME and the Cargo/TorchInductor caches supplied explicitly, so their
Cargo target dirs and staging stay deletable without sudo. The home is
BUILD_ROOT/container-home when a build root is set and /tmp/cuteafd-home inside
the container otherwise.

Release exports derive from CUTEAFD_RELEASE_MATRIX (default: the shipping matrix
reported by scripts/build/release-scope.py default-matrix). --scope / CUTEAFD_RELEASE_SCOPE selects all
(default), comma-separated family or entry names, or explicit (raw knobs only).
Raw CUTEAFD_RELEASE_* export knobs add to selected cards; OFF/empty cannot remove
required exports. Narrow or explicit builds are for development, not release tags.
Both image inventories must cover the selected cards before complete tags update.


Images are labelled with the checkout's Git revision (HEAD), even when the
tree has local changes. Dirty checkouts still get an automatic source manifest
under .cuteafd-release/ so local and remote inventories can be verified; keep
source files unchanged during the build, including when building from a Git
worktree. The compiler receives a copy without Git metadata after host-side
submodule verification; Docker image assembly still reads the live checkout. CUTEAFD_RELEASE_SOURCE_MANIFEST can
supply an existing manifest. Source archives without .git must provide
CUTEAFD_RELEASE_ENGINE_REVISION (a 40-hex Git revision).

Development images default to ghcr.io/tpurtell/cuteafd-dev:tc-<hash>-<arch>,
where dev-toolchain.py hashes the checkout's exact toolchain inputs. Both hosts
use a cached matching image or pull it, verifying its hash, architecture and
pinned base-digest label. Missing images, pull failures and mismatches fall back
to Dockerfile.dev locally, with the reason logged. Set
CUTEAFD_RELEASE_DEV_IMAGE_SOURCE=build to force local dev builds on both hosts
(default registry). Compilation uses the admitted immutable image ID.
CUTEAFD_RELEASE_DEV_IMAGE=sha256:ID overrides only the coordinator, keeping its
existing live toolchain/source checks and image-ID-bound provenance proof.
Missing proof or mismatches refuse this explicit override rather than falling back.
Both legs' exact image IDs, registry references/digests and toolchain hashes ship
in dist/DEV_IMAGE_REUSE.json (checksummed by dist/SHA256SUMS) and image labels.
EOF
}

config="$repo_root/cuteafd.config"
build_hosts_csv=""
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --spark-hosts)
      build_hosts_csv="${2:?--spark-hosts requires a comma-separated host list}"
      shift 2
      ;;
    --scope)
      export CUTEAFD_RELEASE_SCOPE="${2:?--scope requires all, explicit, families or entry names}"
      shift 2
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
      release_die "unknown build argument: $1"
      ;;
  esac
done

release_scope="${CUTEAFD_RELEASE_SCOPE:-all}"
release_matrix="${CUTEAFD_RELEASE_MATRIX:-$repo_root/$(python3 "$repo_root/scripts/build/release-scope.py" default-matrix)}"
release_matrix_sha256="$(sha256sum "$release_matrix" | awk '{print $1}')"
release_scope_exports="$(python3 "$repo_root/scripts/build/release-scope.py" derive \
  --matrix "$release_matrix" --scope "$release_scope" --shell)"
eval "$release_scope_exports"
printf -v release_scope_assignment 'export CUTEAFD_RELEASE_SCOPE=%q' "$release_scope"
release_scope_exports+=$'\n'"$release_scope_assignment"
printf -v release_matrix_assignment 'export CUTEAFD_RELEASE_MATRIX_SHA256=%q' "$release_matrix_sha256"
release_scope_exports+=$'\n'"$release_matrix_assignment"
release_scope_requirements="$(python3 "$repo_root/scripts/build/release-scope.py" requirements \
  --matrix "$release_matrix" --scope "$release_scope")"
bf16_families="${CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES:-}"
audio_aot="${CUTEAFD_RELEASE_AUDIO_AOT:-ON}"
case "$audio_aot" in ON|OFF) ;; *) release_die "CUTEAFD_RELEASE_AUDIO_AOT must be ON or OFF, got: $audio_aot" ;; esac
bf16_family_pattern='^(mimo|mimop|mimof|glm|glmf|qwen4)(;(mimo|mimop|mimof|glm|glmf|qwen4))*$'
[[ -z "$bf16_families" || "$bf16_families" =~ $bf16_family_pattern ]] ||
  release_die "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES must be a semicolon list of mimo, mimop, mimof, glm, glmf or qwen4"


release_load_config "$config"
release_select_build_hosts "$build_hosts_csv"
release_need docker
release_need ssh
release_need rsync
release_need sha256sum
release_need python3
release_need install

release_version="${COORDINATOR_DOCKER_INFERENCE##*:}"
spark_release_version="${SPARK_EXPERT_DOCKER_INFERENCE##*:}"
[[ "$release_version" != "$COORDINATOR_DOCKER_INFERENCE" &&
  "$release_version" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]] ||
  release_die "coordinator inference image must use a valid release tag"
[[ "$spark_release_version" == "$release_version" ]] ||
  release_die "coordinator and Spark inference image release tags must match"

# Spark expert roles are universal by default: the ARM64 (SM121) image carries
# every other native Spark TP shard on top of the always-built TP4 shard, so one
# published pair serves every planned Spark count and ./run.sh requires the role
# the plan names. The x86_64 coordinator needs no Spark role: roles are
# expert-only. CUTEAFD_RELEASE_SPARK_TP_ROLES is an explicit SUBSET override for
# a bounded A/B or a legacy TP4-only rebuild (empty).
# release-spark-tp-roles:start
release_universal_spark_tp_roles="$(release_spark_tp_roles_default)"
# `${VAR-default}`, not `:-`, so an explicitly empty override stays the legacy
# TP4-only request rather than falling back to the universal set.
spark_tp_roles="$(release_spark_tp_roles_canonical \
  "${CUTEAFD_RELEASE_SPARK_TP_ROLES-$release_universal_spark_tp_roles}")"
[[ "$spark_tp_roles" == "$release_universal_spark_tp_roles" ]] ||
  echo "== NON-UNIVERSAL expert role subset '${spark_tp_roles:-<none>}'; this pair cannot serve every planned Spark count =="
# Repeated by --dry-run and the build summary so a subset build is never mistaken
# for the universal default.
spark_tp_roles_note='universal, covers every planned Spark count'
[[ "$spark_tp_roles" == "$release_universal_spark_tp_roles" ]] ||
  spark_tp_roles_note='explicit subset, not universal'
# release-spark-tp-roles:end

# release-build-transport:start
# The SSH option set and the canonical-path validator live in
# scripts/lib/release-common.sh, so a release build and the Spark-facing helpers cannot
# drift apart: stock resolution unless CUTEAFD_RELEASE_SSH_CONFIG names a config file,
# BatchMode forced always, and a bad value refused here before any host is reached.
# What stays here is the release-only setting: where a container leg keeps its
# writable build root.
release_configure_ssh_transport

# Optional unique writable build root for the release container legs. Unset keeps
# the historical in-container /tmp scratch; a set value must be the identical path
# inside the container and on the host, because the artifact compiler guards the
# path it writes and that guard resolves the filesystem behind the string it is
# given. It is per-task: two concurrent builds must never share one root.
release_build_root="${CUTEAFD_RELEASE_BUILD_ROOT:-}"
release_validate_path_setting CUTEAFD_RELEASE_BUILD_ROOT "$release_build_root"
if [[ -n "$release_build_root" ]]; then
  # A root inside the source tree would be staged into its own copy and then
  # guarded as if it were the source, so the two must stay disjoint. The remote
  # value is still unknown here; the leg that knows it repeats this check.
  release_path_within "$release_build_root" "$repo_root" &&
    release_die "CUTEAFD_RELEASE_BUILD_ROOT must not be $repo_root or inside it"
  release_build_root_args=(
    -v "$release_build_root:$release_build_root"
    -e "CUTEAFD_RELEASE_BUILD_ROOT=$release_build_root"
  )
else
  release_build_root_args=()
fi

# Create and probe the build root on the machine that will run the container.
# HOST is empty for the local leg; SCRIPT_DIR is the tree holding
# assert-build-filesystem.py there (the local checkout or the staged remote copy).
# Both operands are %q-quoted, so a validated path cannot be reinterpreted by the
# shell that runs the command.
release_prepare_build_root() {
  [[ -n "$release_build_root" ]] || return 0
  local host="$1" script_dir="$2" quoted_root quoted_dir command
  printf -v quoted_root '%q' "$release_build_root"
  printf -v quoted_dir '%q' "$script_dir"
  command="mkdir -p $quoted_root && python3 $quoted_dir/scripts/build/assert-build-filesystem.py $quoted_root"
  if [[ -n "$host" ]]; then
    release_ssh "$host" "$command" ||
      release_die "$host release build root is not a safe writable filesystem: $release_build_root"
  else
    bash -c "$command" ||
      release_die "release build root is not a safe writable filesystem: $release_build_root"
  fi
}
# release-build-transport:end

# release-build-container-user:start
# Every build container that compiles into a bind-mounted host directory runs as
# the invoking user, so its Cargo target dirs and staging stay deletable without
# sudo. UIDs differ per host (raptor 1000, the Sparks 1001), so the identity is
# taken on the machine that runs docker: here for the coordinator leg, and inside
# the remote heredoc for the Spark leg.
#
# `--user` bypasses the image's passwd lookup, so the env a non-passwd UID needs
# is supplied explicitly: USER/LOGNAME for getpass and Torch Dynamo, a writable
# HOME, and the two cache roots. The image's CARGO_HOME (/opt/cargo) is
# root-owned and therefore not writable by that UID, and the dev image ships no
# warm crate registry for the export, so the build's cargo home is relocated
# under the writable home. With a relocated build root the home lives inside it,
# which keeps the caches on the task's fast build filesystem and inside the
# directory the build already owns and cleans up; otherwise it is the
# container's own /tmp.
release_build_container_home() {
  local build_root="${1:-}"
  if [[ -n "$build_root" ]]; then
    printf '%s\n' "$build_root/container-home"
  else
    printf '%s\n' "/tmp/cuteafd-home"
  fi
}

# One argument per line so a call site can mapfile the group into its own array.
release_build_container_user_args_render() {
  local container_home
  container_home="$(release_build_container_home "${1:-}")"
  printf '%s\n' \
    --user "$(id -u):$(id -g)" \
    -e "HOME=$container_home" \
    -e "USER=$(id -un)" \
    -e "LOGNAME=$(id -un)" \
    -e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor" \
    -e "CARGO_HOME=$container_home/cargo"
}
# release-build-container-user:end

# release-build-budget:start
# Settings that only size waits and budgets, resolved before the dry-run so
# --dry-run reports the plan this build would follow. Nothing here takes a lock,
# a device or a container.
release_build_lock_timeout="${CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS:-1200}"
[[ "$release_build_lock_timeout" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS must be positive seconds"
# release-build-budget:end

release_sequential="${CUTEAFD_RELEASE_SEQUENTIAL:-0}"
[[ "$release_sequential" == 0 || "$release_sequential" == 1 ]] ||
  release_die "CUTEAFD_RELEASE_SEQUENTIAL must be 0 or 1"
release_leg_plan="coordinator and Spark legs build concurrently; shared export/distribution waits for both"
[[ "$release_sequential" == 0 ]] ||
  release_leg_plan="sequential: coordinator then Spark (CUTEAFD_RELEASE_SEQUENTIAL=1)"

release_dev_image_source="${CUTEAFD_RELEASE_DEV_IMAGE_SOURCE:-registry}"
case "$release_dev_image_source" in registry|build) ;; *) release_die "CUTEAFD_RELEASE_DEV_IMAGE_SOURCE must be registry or build" ;; esac

if ((dry_run)); then
  release_toolchain_hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"
  echo "  release scope: $release_scope"
  echo "  release matrix: $release_matrix"
  printf '%s\n' "$release_scope_exports"
  echo "Build dry-run passed; no image, container, SSH or submodule was touched."
  echo "  config: $RELEASE_CONFIG"
  echo "  build hosts (${#RELEASE_BUILD_HOSTS[@]}): $(IFS=,; echo "${RELEASE_BUILD_HOSTS[*]}")"
  echo "  seed host: ${RELEASE_BUILD_HOSTS[0]:-}"
  if [[ -n "${CUTEAFD_RELEASE_DEV_IMAGE:-}" ]]; then
    echo "  coordinator dev image: verify explicit override $CUTEAFD_RELEASE_DEV_IMAGE"
  elif [[ "$release_dev_image_source" == registry ]]; then
    echo "  coordinator dev image: pull ghcr.io/tpurtell/cuteafd-dev:tc-$release_toolchain_hash-amd64 on $(hostname) if missing; verify or build locally"
  else
    echo "  coordinator dev image: build Dockerfile.dev locally on $(hostname)"
  fi
  if [[ "$release_dev_image_source" == registry ]]; then
    echo "  Spark dev image: pull ghcr.io/tpurtell/cuteafd-dev:tc-$release_toolchain_hash-arm64 on ${RELEASE_BUILD_HOSTS[0]:-} if missing; verify or build locally"
  else
    echo "  Spark dev image: build Dockerfile.dev locally on ${RELEASE_BUILD_HOSTS[0]:-}"
  fi
  echo "  build legs: $release_leg_plan"
  echo "  coordinator export cache container args (ID resolved at image admission):"
  cuteafd_export_cache_docker_args '${COORDINATOR_DOCKER_DEV}' /output/EXPORT_CACHE_STATS.jsonl
  echo "  Spark export cache container args (ID resolved on seed at image admission):"
  cuteafd_export_cache_docker_args '${dev_image}' /output/EXPORT_CACHE_STATS.jsonl
  echo "  release tag: $release_version"
  echo "  V41 Spark expert roles: ${spark_tp_roles:-<legacy TP4 only>} ($spark_tp_roles_note)"
  echo "  coordinator image: $COORDINATOR_DOCKER_INFERENCE"
  echo "  spark image: $SPARK_EXPERT_DOCKER_INFERENCE"
  echo "  ssh config: ${release_ssh_config:-<stock>}"
  echo "  release build root: ${release_build_root:-<container /tmp>}"
  echo "  build container user: $(id -un) ($(id -u):$(id -g))"
  echo "  build container home: $(release_build_container_home "$release_build_root")"
  echo "  build lock: $HOME/.cache/cuteafd/build.lock (waited up to ${release_build_lock_timeout}s; no hardware lock is taken)"
  echo "  AOT exports: CPU-only (CUDA_VISIBLE_DEVICES empty; no GPU or hardware lock required)"
  cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash" dry >/dev/null
  echo "  Spark cache plan (host HOME shown as template; states/admission rechecked on Spark, no SSH):"
  cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash" dry aarch64 >/dev/null
  exit 0
fi

# Build under temporary tags; a failed inventory never replaces a complete pair.
complete_coordinator_image="$COORDINATOR_DOCKER_INFERENCE"
complete_spark_image="$SPARK_EXPERT_DOCKER_INFERENCE"
COORDINATOR_DOCKER_INFERENCE+="-pending-$$"
SPARK_EXPERT_DOCKER_INFERENCE+="-pending-$$"
# Keep a single parent EXIT cleanup through export, verification and promotion.
release_pending_cleanup() {
  [[ -z "${coordinator_container:-}" ]] || docker rm -f "$coordinator_container" >/dev/null 2>&1 || true
  docker image rm "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 || true
  local host
  for host in "${RELEASE_BUILD_HOSTS[@]}"; do
    timeout 30s ssh "${release_ssh_opts[@]}" "$host" docker image rm "$SPARK_EXPERT_DOCKER_INFERENCE" >/dev/null 2>&1 || true
  done
}
trap release_pending_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
release_toolchain_hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"

prepare_pinned_source_dependencies() {
  local git_root=""
  if command -v git >/dev/null 2>&1; then
    git_root="$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null || true)"
  fi

  if [[ -n "$git_root" &&
    "$(cd "$git_root" && pwd -P)" == "$(cd "$repo_root" && pwd -P)" ]]; then
    echo "== preparing pinned source dependencies =="
    git -C "$repo_root" submodule sync -- \
      third_party/sparkinfer third_party/xgrammar
    git -C "$repo_root" submodule update --init --checkout -- \
      third_party/sparkinfer third_party/xgrammar
    git -C "$repo_root/third_party/xgrammar" submodule sync -- \
      3rdparty/dlpack
    git -C "$repo_root/third_party/xgrammar" submodule update --init --checkout -- \
      3rdparty/dlpack
  fi

  [[ -f "$repo_root/third_party/sparkinfer/b12x/__init__.py" ]] ||
    release_die "SparkInfer source is missing; initialize third_party/sparkinfer"
  [[ -f "$repo_root/third_party/xgrammar/include/xgrammar/compiler.h" ]] ||
    release_die "XGrammar source is missing; initialize third_party/xgrammar"
  [[ -f "$repo_root/third_party/xgrammar/3rdparty/dlpack/include/dlpack/dlpack.h" ]] ||
    release_die "XGrammar DLPack source is missing; initialize third_party/xgrammar/3rdparty/dlpack"
}

release_need flock
release_need timeout
release_need mkfifo

# One build at a time, and never a hardware lock: a build is CPU work plus
# CPU-only AOT exports. Holding sparks.lock/gpu1.lock across a compile, crate
# download or image assembly would block serving for no reason, so the build
# takes ~/.cache/cuteafd/build.lock instead. It is acquired before the submodule
# refresh so two builds cannot stage over each other's checkout, and held
# through image assembly and distribution: the slot isolation that keeps
# artifacts apart does not cover the shared checkout or the docker daemon.
release_build_lock_dir="$HOME/.cache/cuteafd"
mkdir -p "$release_build_lock_dir"
exec 9>"$release_build_lock_dir/build.lock"
echo "== waiting for the release build lock (${release_build_lock_timeout}s): $release_build_lock_dir/build.lock =="
flock -w "$release_build_lock_timeout" 9 ||
  release_die "timed out waiting for $release_build_lock_dir/build.lock; another release build is running"
echo "== release build lock held ($(date -Is)) =="

prepare_pinned_source_dependencies

export_timeout="${CUTEAFD_RELEASE_EXPORT_TIMEOUT_SECONDS:-7200}"
[[ "$export_timeout" =~ ^[1-9][0-9]*$ ]] || release_die "CUTEAFD_RELEASE_EXPORT_TIMEOUT_SECONDS must be positive seconds"
# Unique names let timeout/signal cleanup remove only this build's containers.
export_container="cuteafd-release-export-$(hostname)-$$"

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
detected_engine_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
engine_source_dirty=0
if [[ -n "$detected_engine_commit" &&
  -n "$(git -C "$repo_root" status --porcelain 2>/dev/null || true)" ]]; then
  engine_source_dirty=1
fi
source_manifest="${CUTEAFD_RELEASE_SOURCE_MANIFEST:-}"
source_manifest_sha256=""
if [[ -z "$source_manifest" ]] && ((engine_source_dirty)); then
  # Build provenance is generated by the build, not an extra manual prerequisite.
  # This directory is excluded from both the inventory and remote source sync.
  mkdir -p "$repo_root/.cuteafd-release/source-manifests"
  source_manifest="$(mktemp "$repo_root/.cuteafd-release/source-manifests/source.XXXXXXXX.sha256")"
  python3 "$repo_root/scripts/build/verify-release-source-manifest.py" \
    --source "$repo_root" --write "$source_manifest"
  echo "== recorded current checkout: $source_manifest =="
fi
if [[ -n "$source_manifest" ]]; then
  source_manifest="$(
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' \
      "$source_manifest"
  )"
  [[ -f "$source_manifest" ]] ||
    release_die "release source manifest not found: $source_manifest"
  source_manifest_sha256="$(sha256sum "$source_manifest" | awk '{print $1}')"
fi

verify_local_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  python3 "$repo_root/scripts/build/verify-release-source-manifest.py" \
    --source "$repo_root" \
    --manifest "$source_manifest" ||
    release_die "release source differs from $source_manifest"
}

verify_remote_source_manifest() {
  [[ -n "$source_manifest" ]] || return 0
  local current_source_manifest_sha256
  current_source_manifest_sha256="$(
    sha256sum "$source_manifest" | awk '{print $1}'
  )"
  [[ "$current_source_manifest_sha256" == "$source_manifest_sha256" ]] ||
    release_die "release source manifest changed during the build"
  local remote_dir_quoted
  printf -v remote_dir_quoted '%q' "$remote_dir"
  release_ssh "$seed_host" \
    "cd $remote_dir_quoted && python3 scripts/build/verify-release-source-manifest.py --source . --manifest -" \
    <"$source_manifest" ||
    release_die "$seed_host staged source differs from $source_manifest"
}

verify_local_source_manifest
sparkinfer_commit="$(
  python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --print-revision
)"
python3 "$repo_root/scripts/build/verify-xgrammar-source.py" \
  --source "$repo_root/third_party/xgrammar" \
  --lock "$repo_root/third_party/xgrammar.lock.json"
engine_revision_override="${CUTEAFD_RELEASE_ENGINE_REVISION:-}"
if [[ -n "$engine_revision_override" ]]; then
  [[ "$engine_revision_override" =~ ^[0-9a-f]{40}(-dirty-[0-9a-f]{12})?$ ]] ||
    release_die "CUTEAFD_RELEASE_ENGINE_REVISION must be REVISION or REVISION-dirty-MANIFEST12"
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "CUTEAFD_RELEASE_ENGINE_REVISION requires CUTEAFD_RELEASE_SOURCE_MANIFEST"
  engine_commit="$engine_revision_override"
elif [[ -z "$detected_engine_commit" ]]; then
  release_die "source snapshot has no Git metadata; set CUTEAFD_RELEASE_ENGINE_REVISION and CUTEAFD_RELEASE_SOURCE_MANIFEST"
elif ((engine_source_dirty)); then
  # Label the Git revision; the build does not encode local changes.
  echo "== note: building $detected_engine_commit with uncommitted local changes =="
  engine_commit="$detected_engine_commit"
else
  engine_commit="$detected_engine_commit"
fi
if [[ "$engine_commit" == *-dirty-* ]]; then
  [[ -n "$source_manifest_sha256" ]] ||
    release_die "dirty engine revision requires CUTEAFD_RELEASE_SOURCE_MANIFEST"
  [[ "$engine_commit" == *"-dirty-${source_manifest_sha256:0:12}" ]] ||
    release_die "dirty engine revision suffix does not match the source manifest"
fi
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.cuteafd.source-manifest.sha256=$source_manifest_sha256"
  )
fi

hosts_csv="$(IFS=,; echo "${RELEASE_BUILD_HOSTS[*]}")"
seed_host="${RELEASE_BUILD_HOSTS[0]}"
remote_dir="${CUTEAFD_RELEASE_REMOTE_BUILD_DIR:-}"
artifact_dir="$repo_root/.cuteafd-release-image"

echo "== validating native Spark build hosts =="
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  release_ssh -o ConnectTimeout=10 "$host" bash -s <<'REMOTE'
set -euo pipefail
command -v docker >/dev/null
command -v setsid >/dev/null
docker info >/dev/null
test "$(uname -m)" = "aarch64"
REMOTE
  echo "  $host: ssh/docker/aarch64 ready"
done

release_sync_program=rsync
if command -v rdmasync >/dev/null 2>&1 &&
  release_ssh "$seed_host" "command -v rdmasync >/dev/null 2>&1"; then
  release_sync_program=rdmasync
  echo "== using RDMA source/artifact synchronization =="
fi

release_sync() {
  # Transport options live in one place: rsync and rdmasync both accept the
  # remote shell as a single --rsh value, so the ssh config override reaches the
  # source and artifact copies as well as the explicit ssh calls above. A bare
  # "-F" argument here would instead be their own --filter=dir-merge option.
  if [[ "$release_sync_program" == rdmasync ]]; then
    rdmasync -a --rdma=required --rdma-show-config --rsh="$release_rsh" "$@"
  else
    rsync -a --rsh="$release_rsh" "$@"
  fi
}

if [[ -z "$remote_dir" ]]; then
  remote_dir="$(
    release_ssh "$seed_host" \
      'printf "%s/cuteafd-release-build" "$HOME"'
  )"
fi

# The Spark staging directory is embedded in remote shell command strings and in
# Docker bind sources, so it gets the same canonical treatment as every other
# setting; the default derived from the seed host's own $HOME is checked too rather
# than assumed to be simple.
release_validate_path_setting CUTEAFD_RELEASE_REMOTE_BUILD_DIR "$remote_dir"
if [[ -n "$release_build_root" ]]; then
  # The remote source tree and the remote scratch must stay disjoint: a build root
  # inside the staged source would be copied into its own build directory, and a
  # source tree inside the scratch would be deleted with it.
  release_path_within "$release_build_root" "$remote_dir" &&
    release_die "CUTEAFD_RELEASE_BUILD_ROOT must not be $remote_dir or inside it"
  release_path_within "$remote_dir" "$release_build_root" &&
    release_die "the Spark staging directory must not be inside CUTEAFD_RELEASE_BUILD_ROOT"
fi

local_free_kib="$(df -Pk "$repo_root" | awk 'NR==2 {print $4}')"
((local_free_kib >= 60 * 1024 * 1024)) || release_die "local build needs at least 60 GiB free"
# Measure the filesystem that really holds the Spark staging directory (it may
# resolve off a small $HOME, e.g. a symlink onto the Spark /scratch).
remote_free_kib="$(
  release_ssh "$seed_host" bash -s -- "$remote_dir" <<'REMOTE'
p="$(readlink -m "$1")"
while [ ! -e "$p" ]; do p="$(dirname "$p")"; done
df -Pk "$(readlink -f "$p")" | awk 'NR == 2 { print $4 }'
REMOTE
)"
((remote_free_kib >= $remote_min_free_gib * 1024 * 1024)) || release_die "$seed_host build needs at least $remote_min_free_gib GiB free"

# A relocated build root is its own filesystem decision: the two checks above
# cover the source tree and the Spark home, not the scratch the container writes.
# The remote mkdir is where the directory starts to exist, so the same path can be
# bind-mounted into the container without Docker creating it as root-owned later.
if [[ -n "$release_build_root" ]]; then
  # Create first: `df` on a path that does not exist yields no capacity, which an
  # arithmetic comparison would silently read as zero and report as a space problem.
  mkdir -p "$release_build_root" ||
    release_die "cannot create release build root: $release_build_root"
  local_root_free_kib="$(df -Pk "$release_build_root" | awk 'NR==2 {print $4}')"
  ((local_root_free_kib >= 60 * 1024 * 1024)) ||
    release_die "release build root $release_build_root needs at least 60 GiB free"
  printf -v release_build_root_quoted '%q' "$release_build_root"
  remote_root_free_kib="$(
    release_ssh "$seed_host" \
      "mkdir -p $release_build_root_quoted && df -Pk $release_build_root_quoted | awk 'NR == 2 { print \$4 }'"
  )"
  ((remote_root_free_kib >= $remote_min_free_gib * 1024 * 1024)) ||
    release_die "$seed_host release build root $release_build_root needs at least $remote_min_free_gib GiB free"
fi

# Only the coordinator writes local artifacts/reuse proof. Allocate the proof path
# before forking so the shared tail can read it without importing worker variables.
release_log_root="${release_build_root:-$HOME/.cache/cuteafd/builds/release-source}"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$release_log_root"
mkdir -p "$release_log_root"
release_leg_log_dir="$(mktemp -d "$release_log_root/build-legs.XXXXXXXX")"
release_dev_reuse_manifest="$release_leg_log_dir/coordinator-dev-image.json"

build_coordinator_release() (
release_source_dir=""
coordinator_export_container="$export_container-coordinator"
release_export_cleanup() {
  docker rm -f "$coordinator_export_container" >/dev/null 2>&1 || true
}
trap 'release_export_cleanup; rm -rf "$release_source_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
release_dev_reuse_label_args=()
if [[ -n "${CUTEAFD_RELEASE_DEV_IMAGE:-}" ]]; then
  echo "== verifying reused coordinator development image: $CUTEAFD_RELEASE_DEV_IMAGE =="
  COORDINATOR_DOCKER_DEV="$(python3 "$repo_root/scripts/build/verify-release-dev-image.py" \
    --source "$repo_root" --image "$CUTEAFD_RELEASE_DEV_IMAGE" \
    --output "$release_dev_reuse_manifest")" || release_die "coordinator dev image reuse verification failed"
  python3 "$repo_root/scripts/build/select-dev-image.py" record-override \
    --source "$repo_root" --manifest "$release_dev_reuse_manifest"
else
  COORDINATOR_DOCKER_DEV="$(python3 "$repo_root/scripts/build/select-dev-image.py" select \
    --source "$repo_root" --arch amd64 \
    --engine-commit "$engine_commit" --mode "$release_dev_image_source" \
    --output "$release_dev_reuse_manifest")"
fi
python3 "$repo_root/scripts/build/select-dev-image.py" labels --manifest "$release_dev_reuse_manifest" >"$release_leg_log_dir/coordinator-dev-labels"
mapfile -t release_dev_reuse_label_args <"$release_leg_log_dir/coordinator-dev-labels"

echo "== compiling coordinator release artifacts in CPU-only development container =="
mkdir -p "$artifact_dir"
release_prepare_build_root "" "$repo_root"
# Git worktree submodule gitfiles refer outside /source. Verify them on the host
# above, then give the compiler a clean copy whose tree locks remain authoritative.
release_source_parent="${release_build_root:-$HOME/.cache/cuteafd/builds/release-source}"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$release_source_parent"
mkdir -p "$release_source_parent"
release_source_dir="$(mktemp -d "$release_source_parent/coordinator-source.XXXXXXXX")"
"$repo_root/scripts/build/stage-release-source.sh" "$repo_root" "$release_source_dir"
coordinator_export_container="$export_container-coordinator"
echo "== coordinator AOT export on $(hostname) at $(date -Is): CPU-only, no hardware lock =="
release_build_user_args=()
mapfile -t release_build_user_args < <(release_build_container_user_args_render "$release_build_root")
# Killing a docker client does not stop its container; the leg's EXIT trap owns
# the named export container, including failures before docker run.
# --foreground keeps timeout and its client inside the cancellable leg group.
coordinator_export_status=0
compiler_cache_args=()
cache_plan="$(cuteafd_build_cache_docker_args "${release_build_root:-$release_source_parent}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash")" || release_die "cache plan failed"
mapfile -t compiler_cache_args <<<"$cache_plan"
mapfile -t export_cache_args < <(cuteafd_export_cache_docker_args "$COORDINATOR_DOCKER_DEV" /output/EXPORT_CACHE_STATS.jsonl)
timeout --foreground "$export_timeout" docker run --rm --name "$coordinator_export_container" \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e "CUDA_VISIBLE_DEVICES=" \
  -e "NVIDIA_VISIBLE_DEVICES=void" \
  "${release_build_user_args[@]}" \
  "${compiler_cache_args[@]}" \
  "${export_cache_args[@]}" \
  -e "CUTEAFD_RELEASE_COORDINATOR_FEATURES=${CUTEAFD_RELEASE_COORDINATOR_FEATURES:-}" \
  -e "CUTEAFD_RELEASE_EXPERT_FAMILIES=${CUTEAFD_RELEASE_EXPERT_FAMILIES:-}" \
  -e "CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS=${CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS:-}" \
  -e "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$bf16_families" \
  -e "CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot" \
  -e "CUTEAFD_RELEASE_VISION_ATTENTION_AOT=${CUTEAFD_RELEASE_VISION_ATTENTION_AOT:-ON}" \
  ${native_build_env_args[@]+"${native_build_env_args[@]}"} \
  -e "CUTEAFD_RELEASE_DSV4_AOT=${CUTEAFD_RELEASE_DSV4_AOT:-ON}" \
  -e "CUTEAFD_RELEASE_DSV4_MAX_CONTEXT=${CUTEAFD_RELEASE_DSV4_MAX_CONTEXT:-1048576}" \
  -e "CUTEAFD_RELEASE_GLM_AOT=${CUTEAFD_RELEASE_GLM_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_CONTEXT_SPLIT_AOT=${CUTEAFD_RELEASE_CONTEXT_SPLIT_AOT:-ON}" \
  -e "CUTEAFD_RELEASE_MIMO_AOT=${CUTEAFD_RELEASE_MIMO_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_MIMO_GEOMETRIES=${CUTEAFD_RELEASE_MIMO_GEOMETRIES:-mimo,mimo2,mimop,mimop2}" \
  -e "CUTEAFD_RELEASE_GLMF_AOT=${CUTEAFD_RELEASE_GLMF_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_QWEN4_AOT=${CUTEAFD_RELEASE_QWEN4_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_QWEN_DENSE_AOT=${CUTEAFD_RELEASE_QWEN_DENSE_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_STEP5_AOT=${CUTEAFD_RELEASE_STEP5_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_KIMIK3_AOT=${CUTEAFD_RELEASE_KIMIK3_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_MUSE_AOT=${CUTEAFD_RELEASE_MUSE_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_ML4_AOT=${CUTEAFD_RELEASE_ML4_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_MINIMAX_AOT=${CUTEAFD_RELEASE_MINIMAX_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_INKLING2_AOT=${CUTEAFD_RELEASE_INKLING2_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_INKLING2_MINI_AOT=${CUTEAFD_RELEASE_INKLING2_MINI_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_HY4F_AOT=${CUTEAFD_RELEASE_HY4F_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_HY4_AOT=${CUTEAFD_RELEASE_HY4_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_EXLR8_AOT=${CUTEAFD_RELEASE_EXLR8_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_EXLR8_FAMILIES=${CUTEAFD_RELEASE_EXLR8_FAMILIES:-}" \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$release_source_dir:/source:ro" \
  -v "$artifact_dir:/output" \
  "$COORDINATOR_DOCKER_DEV" \
  /source/scripts/build/build-release-artifacts.sh /source coordinator 120 /output ||
  coordinator_export_status=$?
(( coordinator_export_status == 0 )) ||
  release_die "coordinator AOT export failed (exit $coordinator_export_status) on $(hostname) at $(date -Is)"
rm -rf "$release_source_dir"
release_source_dir=""

echo "== building coordinator inference image: $COORDINATOR_DOCKER_INFERENCE =="
DOCKER_BUILDKIT=1 docker build \
  "${release_source_label_args[@]}" \
  "${release_dev_reuse_label_args[@]}" \
  --label "io.cuteafd.release.scope=$release_scope" \
  --label "io.cuteafd.release.matrix.sha256=$release_matrix_sha256" \
  --build-arg CUTEAFD_ROLE=coordinator \
  --build-arg CUDA_ARCH=120 \
  --build-arg CUTEAFD_ENGINE_COMMIT="$engine_commit" \
  --build-arg CUTEAFD_SPARKINFER_COMMIT="$sparkinfer_commit" \
  --build-arg CUTEAFD_RELEASE_VERSION="$release_version" \
  --build-arg CUTEAFD_SPARK_TP_ROLES= \
  -f "$repo_root/docker/Dockerfile.release" \
  -t "$COORDINATOR_DOCKER_INFERENCE" \
  "$repo_root"
)

build_spark_release() (
echo "== staging native Spark build on $seed_host:$remote_dir =="
printf -v remote_dir_quoted '%q' "$remote_dir"
release_ssh "$seed_host" "mkdir -p $remote_dir_quoted"
release_sync --delete \
  --exclude '.git' \
  --exclude '.venv*/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.cuteafd-cache/' \
  --exclude '.cuteafd-release/' \
  --exclude '.cuteafd-release-image/' \
  --exclude '.cuteafd-wip/' \
  --exclude 'dist/' \
  --exclude 'rust/target/' \
  --exclude 'native/build*/' \
  "$repo_root/" "$seed_host:$remote_dir/"
# The broad staging sync protects excluded paths from deletion. Reconcile the
# pinned source separately so bytecode left by an earlier build cannot survive
# merely because it is now excluded.
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '.venv*/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/sparkinfer/" \
  "$seed_host:$remote_dir/third_party/sparkinfer/"
release_sync --delete --delete-excluded \
  --exclude '.git' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  "$repo_root/third_party/xgrammar/" \
  "$seed_host:$remote_dir/third_party/xgrammar/"
verify_remote_source_manifest

# Prepared before the leg below so the quoted heredoc region stays exactly the
# argument transport it is tested as: creating and probing the root is a host-side
# step, not part of what the remote shell receives.
release_prepare_build_root "$seed_host" "$remote_dir"
echo "== building Spark development and inference images natively on $seed_host =="
build_spark_release_leg() {
  local phase="$1"
  timeout --foreground "$export_timeout" ssh "${release_ssh_opts[@]}" "$seed_host" setsid --wait bash -s -- \
  "$remote_dir" "$SPARK_EXPERT_DOCKER_DEV" "$SPARK_EXPERT_DOCKER_INFERENCE" \
  "$engine_commit" "$sparkinfer_commit" "$release_version" \
  "${source_manifest_sha256:-__legacy__}" "$(r="${spark_tp_roles//;/,}"; echo "${r:-__legacy__}")" \
  "${release_build_root:-__legacy__}" \
  "$(f="${CUTEAFD_RELEASE_EXPERT_FAMILIES:-}"; f="${f//;/,}"; echo "${f:-__legacy__}")" \
  "$(f="${bf16_families//;/,}"; echo "${f:-__legacy__}")" \
  "$phase" "$export_container-expert" "${native_build_jobs:-__legacy__}" \
  "$(printf '%q' "${CUTEAFD_KACHE_SPARK:-__legacy__}")" \
  "$(printf '%q' "${CUTEAFD_KACHE_REMOTE:-__legacy__}")" \
  "$(printf '%q' "${CUTEAFD_KACHE_SPARK_CACHE_DIR:-__legacy__}")" "${CUTEAFD_SCCACHE_CUDA:-0}" "${release_dev_image_source:-registry}" "$audio_aot" "${CUTEAFD_BUILD_CACHES:-on}" \
  "$(printf '%s' "$release_scope_exports" | base64 -w0)" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
dev_image="$2"
inference_image="$3"
engine_commit="$4"
sparkinfer_commit="$5"
release_version="$6"
# SSH reconstructs a shell command and can omit an empty argument, so every
# optional trailing value is passed as a non-empty sentinel and decoded here.
# Both source_manifest_sha256 and spark_tp_roles are optional: sentinels keep
# the two from shifting into each other's position.
source_manifest_sha256="${7-__legacy__}"
# The role list travels as a comma list so a remote shell cannot split it at a
# semicolon; it is restored to the CMake semicolon list here.
spark_tp_roles="${8-__legacy__}"
# The relocated build root is optional and uses the same sentinel convention.
# It is already created and filesystem-guarded by the caller, on this host.
release_build_root="${9-__legacy__}"
# Extra routed-expert kernel families (FAMILY:ROLE), comma-encoded like the roles.
expert_families="${10-__legacy__}"
[[ "$expert_families" != "__legacy__" ]] || expert_families=
expert_families="${expert_families//,/;}"
# Optional BF16-input siblings travel with a sentinel and the same comma encoding.
bf16_families="${11-__legacy__}"
[[ "$bf16_families" != "__legacy__" ]] || bf16_families=
bf16_families="${bf16_families//,/;}"
[[ "$source_manifest_sha256" != "__legacy__" ]] || source_manifest_sha256=
[[ "$spark_tp_roles" != "__legacy__" ]] || spark_tp_roles=
[[ "$release_build_root" != "__legacy__" ]] || release_build_root=
spark_tp_roles="${spark_tp_roles//,/;}"
release_build_root_args=()
if [[ -n "$release_build_root" ]]; then
  release_build_root_args=(
    -v "$release_build_root:$release_build_root"
    -e "CUTEAFD_RELEASE_BUILD_ROOT=$release_build_root"
  )
fi
release_source_label_args=()
if [[ -n "$source_manifest_sha256" ]]; then
  release_source_label_args+=(
    --label "io.cuteafd.source-manifest.sha256=$source_manifest_sha256"
  )
fi
cd "$remote_dir"
compiler_cache_args=()
export CUTEAFD_BUILD_CACHES="${21:-on}"
eval "$(printf '%s' "${22:?}" | base64 -d)"
release_scope="$CUTEAFD_RELEASE_SCOPE"
release_matrix_sha256="$CUTEAFD_RELEASE_MATRIX_SHA256"
[[ "${15:-__legacy__}" == __legacy__ ]] || export CUTEAFD_KACHE="${15}"
[[ "${17:-__legacy__}" == __legacy__ ]] || export CUTEAFD_KACHE_CACHE_DIR="${17}"
export CUTEAFD_SCCACHE_CUDA="${18:-1}"
source scripts/build/build-caches.sh
cuteafd_build_cache_defaults
phase="${12:?}"
# The audio tower switch reaches the remote leg as its own argument (ON/OFF).
audio_aot="${20:-ON}"
export_container="${13:?}"
native_build_jobs="${14-__legacy__}"
[[ "$native_build_jobs" != "__legacy__" ]] || native_build_jobs=
native_build_env_args=()
[[ -z "$native_build_jobs" ]] || native_build_env_args=(-e "CMAKE_BUILD_PARALLEL_LEVEL=$native_build_jobs")
# release-spark-process-group:start
# Each SSH phase has its own session. A separate cancellation SSH can stop its
# docker clients/children even when closing the original SSH did not deliver HUP.
process_dir="$remote_dir/.cuteafd-release"
process_file="$process_dir/$export_container.pid"
cancel_file="$process_dir/$export_container.cancel"
mkdir -p "$process_dir"
[[ ! -e "$cancel_file" ]] || exit 143
printf '%s\n' "$$" >"$process_file"
cleanup_spark_phase() {
  trap '' HUP INT TERM
  kill -TERM -- "-$$" 2>/dev/null || true
  docker rm -f "$export_container" >/dev/null 2>&1 || true
  rm -f "$process_file"
}
trap cleanup_spark_phase EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
[[ ! -e "$cancel_file" ]] || exit 143
# release-spark-process-group:end
dev_manifest="$process_dir/$export_container.dev-image.json"
if [[ "$phase" == dev ]]; then
python3 scripts/build/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json \
  --require-no-python-cache
python3 scripts/build/select-dev-image.py select \
  --source "$remote_dir" --arch arm64 \
  --engine-commit "$engine_commit" --mode "${19:-registry}" --output "$dev_manifest"
else
dev_image="$(python3 scripts/build/select-dev-image.py image --manifest "$dev_manifest")"
fi
if [[ "$phase" == export ]]; then
echo "== Spark AOT export on $(hostname) at $(date -Is): CPU-only, no hardware lock =="
mkdir -p .cuteafd-release-image
# This host's invoking user, not root: the container writes a Cargo target dir
# and staging into bind-mounted host paths (see the coordinator note in build.sh).
# A --user UID has no passwd entry here, so the identity and the cache roots are
# passed explicitly; the artifact compiler creates them before Cargo runs.
container_home=/tmp/cuteafd-home
[[ -z "$release_build_root" ]] || container_home="$release_build_root/container-home"
cache_plan="$(cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$container_home" "$(python3 scripts/build/dev-toolchain.py)")" || exit 2
mapfile -t compiler_cache_args <<<"$cache_plan"
mapfile -t export_cache_args < <(cuteafd_export_cache_docker_args "$dev_image" /output/EXPORT_CACHE_STATS.jsonl)
spark_export_status=0
docker run --rm --name "$export_container" \
  "${compiler_cache_args[@]}" \
  "${export_cache_args[@]}" \
  --user "$(id -u):$(id -g)" \
  -e "HOME=$container_home" \
  -e "USER=$(id -un)" \
  -e "LOGNAME=$(id -un)" \
  -e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor" \
  -e "CARGO_HOME=$container_home/cargo" \
  -e "CUDA_VISIBLE_DEVICES=" \
  -e "NVIDIA_VISIBLE_DEVICES=void" \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e "CUTEAFD_RELEASE_SPARK_TP_ROLES=$spark_tp_roles" \
  -e "CUTEAFD_RELEASE_EXPERT_FAMILIES=$expert_families" \
  -e "CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS=${CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS:-}" \
  -e "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$bf16_families" \
  -e "CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot" \
  -e "CUTEAFD_RELEASE_VISION_ATTENTION_AOT=${CUTEAFD_RELEASE_VISION_ATTENTION_AOT:-ON}" \
  -e "CUTEAFD_RELEASE_EXLR8_AOT=${CUTEAFD_RELEASE_EXLR8_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_EXLR8_FAMILIES=${CUTEAFD_RELEASE_EXLR8_FAMILIES:-}" \
  ${native_build_env_args[@]+"${native_build_env_args[@]}"} \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$remote_dir:/source:ro" \
  -v "$remote_dir/.cuteafd-release-image:/output" \
  "$dev_image" \
  /source/scripts/build/build-release-artifacts.sh /source expert 121 /output ||
  spark_export_status=$?
(( spark_export_status == 0 )) ||
  { echo "$(hostname): the Spark AOT export failed (exit $spark_export_status) at $(date -Is)" >&2; exit "$spark_export_status"; }
fi
if [[ "$phase" == image ]]; then
python3 scripts/build/select-dev-image.py labels --manifest "$dev_manifest" >"$dev_manifest.labels"
mapfile -t release_dev_reuse_label_args <"$dev_manifest.labels"
DOCKER_BUILDKIT=1 docker build \
  "${release_source_label_args[@]}" \
  "${release_dev_reuse_label_args[@]}" \
  --label "io.cuteafd.release.scope=$release_scope" \
  --label "io.cuteafd.release.matrix.sha256=$release_matrix_sha256" \
  --build-arg CUTEAFD_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg CUTEAFD_ENGINE_COMMIT="$engine_commit" \
  --build-arg CUTEAFD_SPARKINFER_COMMIT="$sparkinfer_commit" \
  --build-arg CUTEAFD_RELEASE_VERSION="$release_version" \
  --build-arg CUTEAFD_SPARK_TP_ROLES="$spark_tp_roles" \
  -f docker/Dockerfile.release \
  -t "$inference_image" .
fi
REMOTE
}
build_spark_release_leg dev
build_spark_release_leg export
build_spark_release_leg image
verify_remote_source_manifest
)

# release-build-cancellation:start
release_cancel_coordinator_build() {
  # Also clean from the supervisor, outside the killed group: a worker trap may
  # itself have been terminated or a Docker client may have hung during cleanup.
  timeout 20 docker rm -f "$export_container-coordinator" >/dev/null 2>&1 || true
}

release_cancel_remote_build() {
  # A new bounded SSH owns cancellation; killing an SSH client alone is not a
  # reliable remote process/container cleanup. Only this build's names are used.
  timeout 30 ssh "${release_ssh_opts[@]}" -o ConnectTimeout=5 "$seed_host" bash -s -- \
    "$remote_dir" "$export_container-expert" <<'CANCEL'
set -euo pipefail
process_dir="$1/.cuteafd-release"
container="$2"
mkdir -p "$process_dir"
# Close the race with a phase whose SSH arrived just as cancellation began.
: >"$process_dir/$container.cancel"
pid=""
[[ ! -f "$process_dir/$container.pid" ]] || read -r pid <"$process_dir/$container.pid" || true
if [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] &&
   grep -zFq -- "$container" "/proc/$pid/cmdline"; then
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((i=0; i<5; i++)); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
fi
docker rm -f "$container" >/dev/null 2>&1 || true
rm -f "$process_dir/$container.pid"
CANCEL
}
# release-build-cancellation:end

# release-build-supervision:start
source "$repo_root/scripts/lib/build-supervision.sh"
release_build_legs() {
  echo "== $release_leg_plan =="
  build_supervise 'release build' "$release_leg_log_dir" "$release_sequential" \
    coord build_coordinator_release release_cancel_coordinator_build coordinator.log \
    spark build_spark_release release_cancel_remote_build spark.log
}
# release-build-supervision:end

release_build_legs

echo "== exporting release binaries =="
mkdir -p "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert"
find "$repo_root/dist/coordinator" "$repo_root/dist/spark-expert" \
  -mindepth 1 -delete
echo "== checking selected release matrix against built image inventories =="
python3 "$repo_root/scripts/build/release-scope.py" check-image --role coordinator \
  --image "$COORDINATOR_DOCKER_INFERENCE" --requirements "$release_scope_requirements"
release_ssh "$seed_host" python3 "$remote_dir/scripts/build/release-scope.py" check-image \
  --role expert --image "$SPARK_EXPERT_DOCKER_INFERENCE" --requirements "$(printf '%q' "$release_scope_requirements")"

coordinator_container="$(docker create "$COORDINATOR_DOCKER_INFERENCE")"

docker cp "$coordinator_container:/opt/cuteafd/bin/cuteafd" "$repo_root/dist/coordinator/cuteafd"
docker cp "$coordinator_container:/opt/cuteafd/lib/libcuteafd_native.so" "$repo_root/dist/coordinator/libcuteafd_native.so"
# Host-ISA variants (scripts/build/host-isa.sh): dist/coordinator/x86-64-v4/{cuteafd,libcuteafd_native.so}.
for host_isa in x86-64-v4; do
  mkdir -p "$repo_root/dist/coordinator/$host_isa"
  docker cp "$coordinator_container:/opt/cuteafd/bin/$host_isa/cuteafd" "$repo_root/dist/coordinator/$host_isa/cuteafd"
  docker cp "$coordinator_container:/opt/cuteafd/lib/$host_isa/libcuteafd_native.so" "$repo_root/dist/coordinator/$host_isa/libcuteafd_native.so"
done
docker cp "$coordinator_container:/opt/cuteafd/lib/exl3" "$repo_root/dist/coordinator/exl3"
docker cp "$coordinator_container:/opt/cuteafd/lib/fp8" "$repo_root/dist/coordinator/fp8"
docker cp "$coordinator_container:/opt/cuteafd/share/V41_EXPERT_AOT.json" "$repo_root/dist/coordinator/V41_EXPERT_AOT.json"
docker cp "$coordinator_container:/opt/cuteafd/share/V41_EXPERT_WORKSPACES" "$repo_root/dist/coordinator/V41_EXPERT_WORKSPACES"
docker cp "$coordinator_container:/opt/cuteafd/share/V41_EXPERT_TP_AOT.json" "$repo_root/dist/coordinator/V41_EXPERT_TP_AOT.json"
docker cp "$coordinator_container:/opt/cuteafd/share/V41_FP8_AOT.json" "$repo_root/dist/coordinator/V41_FP8_AOT.json"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/THIRD_PARTY_NOTICES.md" \
  "$repo_root/dist/coordinator/THIRD_PARTY_NOTICES.md"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/SPARKINFER_PROVENANCE.json" \
  "$repo_root/dist/coordinator/SPARKINFER_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/licenses/sparkinfer/LICENSE" \
  "$repo_root/dist/coordinator/SPARKINFER_LICENSE"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/SPARKINFER_SHA256SUMS" \
  "$repo_root/dist/coordinator/SPARKINFER_SHA256SUMS"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/XGRAMMAR_PROVENANCE.json" \
  "$repo_root/dist/coordinator/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/licenses/xgrammar/LICENSE" \
  "$repo_root/dist/coordinator/XGRAMMAR_LICENSE"
docker cp \
  "$coordinator_container:/opt/cuteafd/share/XGRAMMAR_SHA256SUMS" \
  "$repo_root/dist/coordinator/XGRAMMAR_SHA256SUMS"
docker rm "$coordinator_container" >/dev/null
coordinator_container=""

release_ssh "$seed_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote_dir/dist/spark-expert" <<'REMOTE'
set -euo pipefail
image="$1"
destination="$2"
mkdir -p "$destination"
find "$destination" -mindepth 1 -delete
container="$(docker create "$image")"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker cp "$container:/opt/cuteafd/bin/cuteafd" "$destination/cuteafd"
docker cp "$container:/opt/cuteafd/lib/libcuteafd_native.so" "$destination/libcuteafd_native.so"
docker cp "$container:/opt/cuteafd/lib/exl3" "$destination/exl3"
docker cp "$container:/opt/cuteafd/lib/fp8" "$destination/fp8"
docker cp "$container:/opt/cuteafd/share/V41_EXPERT_AOT.json" "$destination/V41_EXPERT_AOT.json"
docker cp "$container:/opt/cuteafd/share/V41_EXPERT_WORKSPACES" "$destination/V41_EXPERT_WORKSPACES"
docker cp "$container:/opt/cuteafd/share/V41_EXPERT_TP_AOT.json" "$destination/V41_EXPERT_TP_AOT.json"
docker cp "$container:/opt/cuteafd/share/V41_FP8_AOT.json" "$destination/V41_FP8_AOT.json"
docker cp \
  "$container:/opt/cuteafd/share/THIRD_PARTY_NOTICES.md" \
  "$destination/THIRD_PARTY_NOTICES.md"
docker cp \
  "$container:/opt/cuteafd/share/SPARKINFER_PROVENANCE.json" \
  "$destination/SPARKINFER_PROVENANCE.json"
docker cp \
  "$container:/opt/cuteafd/share/licenses/sparkinfer/LICENSE" \
  "$destination/SPARKINFER_LICENSE"
docker cp \
  "$container:/opt/cuteafd/share/SPARKINFER_SHA256SUMS" \
  "$destination/SPARKINFER_SHA256SUMS"
docker cp \
  "$container:/opt/cuteafd/share/XGRAMMAR_PROVENANCE.json" \
  "$destination/XGRAMMAR_PROVENANCE.json"
docker cp \
  "$container:/opt/cuteafd/share/licenses/xgrammar/LICENSE" \
  "$destination/XGRAMMAR_LICENSE"
docker cp \
  "$container:/opt/cuteafd/share/XGRAMMAR_SHA256SUMS" \
  "$destination/XGRAMMAR_SHA256SUMS"
docker rm "$container" >/dev/null
trap - EXIT
REMOTE
release_sync --delete \
  "$seed_host:$remote_dir/dist/spark-expert/" \
  "$repo_root/dist/spark-expert/"
dist_source_manifest=()
release_sync "$seed_host:$remote_dir/.cuteafd-release/$export_container-expert.dev-image.json" \
  "$release_leg_log_dir/spark-dev-image.json"
python3 "$repo_root/scripts/build/select-dev-image.py" merge \
  --coordinator "$release_dev_reuse_manifest" --spark "$release_leg_log_dir/spark-dev-image.json" \
  --output "$repo_root/dist/DEV_IMAGE_REUSE.json"
dist_source_manifest+=(DEV_IMAGE_REUSE.json)
# Receipts are informational: never retain a stale prior build's stats.
rm -f "$repo_root/dist/EXPORT_CACHE_STATS.coordinator.jsonl" "$repo_root/dist/EXPORT_CACHE_STATS.spark.jsonl"
if install -m 0644 "$artifact_dir/EXPORT_CACHE_STATS.jsonl" "$repo_root/dist/EXPORT_CACHE_STATS.coordinator.jsonl"; then
  dist_source_manifest+=(EXPORT_CACHE_STATS.coordinator.jsonl)
else
  echo "WARNING: coordinator export cache receipt unavailable" >&2
fi
if release_sync "$seed_host:$remote_dir/.cuteafd-release-image/EXPORT_CACHE_STATS.jsonl" \
  "$repo_root/dist/EXPORT_CACHE_STATS.spark.jsonl"; then
  dist_source_manifest+=(EXPORT_CACHE_STATS.spark.jsonl)
else
  rm -f "$repo_root/dist/EXPORT_CACHE_STATS.spark.jsonl"
  echo "WARNING: Spark export cache receipt unavailable" >&2
fi
if [[ -n "$source_manifest" ]]; then
  install -m 0644 "$source_manifest" "$repo_root/dist/SOURCE_SHA256SUMS"
  dist_source_manifest+=(SOURCE_SHA256SUMS)
fi
for role in coordinator spark-expert; do
  python3 "$repo_root/scripts/build/sparkinfer-release-provenance.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --license "$repo_root/dist/$role/SPARKINFER_LICENSE" \
    --notices "$repo_root/dist/$role/THIRD_PARTY_NOTICES.md" \
    --verify "$repo_root/dist/$role/SPARKINFER_PROVENANCE.json"
  (
    # A scope with no EXL3 family ships an empty exl3/ (see build-release-artifacts.sh).
    if compgen -G "$repo_root/dist/$role/exl3/*" >/dev/null; then
      python3 "$repo_root/python/tools/aot/package_exl3_aot.py" verify --package "$repo_root/dist/$role/exl3" --sparkinfer-revision "$sparkinfer_commit"
    fi
    cd "$repo_root/dist/$role"
    sha256sum -c SPARKINFER_SHA256SUMS
    sha256sum -c XGRAMMAR_SHA256SUMS
  )
done
# EXL3 manifests live either flat in the exl3 root (legacy single family) or
# nested per decoder-tier family (multi-family release images). Collect whichever
# layout the build produced so the checksum list never names a missing path.
dist_exl3_manifests=()
for role in coordinator spark-expert; do
  if [[ -f "$repo_root/dist/$role/exl3/manifest.json" ]]; then
    dist_exl3_manifests+=("$role/exl3/manifest.json")
  else
    for manifest in "$repo_root/dist/$role"/exl3/exl3-*/manifest.json; do
      [[ -f "$manifest" ]] || continue
      dist_exl3_manifests+=("${manifest#"$repo_root/dist/"}")
    done
  fi
done
(
  cd "$repo_root/dist"
  sha256sum \
    coordinator/cuteafd coordinator/libcuteafd_native.so coordinator/x86-64-v4/cuteafd coordinator/x86-64-v4/libcuteafd_native.so coordinator/V41_EXPERT_AOT.json coordinator/V41_EXPERT_TP_AOT.json coordinator/V41_FP8_AOT.json \
    coordinator/THIRD_PARTY_NOTICES.md \
    coordinator/SPARKINFER_PROVENANCE.json \
    coordinator/SPARKINFER_LICENSE \
    coordinator/SPARKINFER_SHA256SUMS \
    coordinator/XGRAMMAR_PROVENANCE.json \
    coordinator/XGRAMMAR_LICENSE \
    coordinator/XGRAMMAR_SHA256SUMS \
    spark-expert/cuteafd spark-expert/libcuteafd_native.so spark-expert/V41_EXPERT_AOT.json spark-expert/V41_EXPERT_TP_AOT.json spark-expert/V41_FP8_AOT.json \
    spark-expert/THIRD_PARTY_NOTICES.md \
    spark-expert/SPARKINFER_PROVENANCE.json \
    spark-expert/SPARKINFER_LICENSE \
    spark-expert/SPARKINFER_SHA256SUMS \
    spark-expert/XGRAMMAR_PROVENANCE.json \
    spark-expert/XGRAMMAR_LICENSE \
    spark-expert/XGRAMMAR_SHA256SUMS \
    $(find coordinator/V41_EXPERT_WORKSPACES spark-expert/V41_EXPERT_WORKSPACES -type f | LC_ALL=C sort) \
    "${dist_exl3_manifests[@]}" \
    "${dist_source_manifest[@]}" >SHA256SUMS
  sha256sum -c SHA256SUMS
)
verify_local_source_manifest
verify_remote_source_manifest

echo "== distributing fresh Spark inference image =="
for host in "${RELEASE_BUILD_HOSTS[@]:1}"; do
  # A stopped expert container can still reference the prior image ID. Force
  # removal only untags that image while preserving the referenced layers, so
  # ensure_image cannot mistake the stale tag for the fresh seed image.
  release_ssh "$host" "docker image rm --force '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null 2>&1 || true"
done
rdmapipe_ready=1
# A single build host has nobody to copy to; do not require rdmapipe then.
((${#RELEASE_BUILD_HOSTS[@]} > 1)) || rdmapipe_ready=2
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  ((rdmapipe_ready != 2)) || break
  if ! release_ssh "$host" "command -v rdmapipe >/dev/null 2>&1"; then
    rdmapipe_ready=0
    break
  fi
done
if ((rdmapipe_ready == 2)); then
  echo "== single Spark build host: nothing to distribute =="
elif ((rdmapipe_ready)); then
  echo "== concurrently distributing Spark image over RDMA =="
  image_copy_pids=()
  image_copy_hosts=()
  printf -v spark_image_quoted '%q' "$SPARK_EXPERT_DOCKER_INFERENCE"
  for host in "${RELEASE_BUILD_HOSTS[@]:1}"; do
    (
      set -o pipefail
      echo "== RDMA image copy $seed_host -> $host =="
      release_ssh "$seed_host" \
        "set -o pipefail; docker image save $spark_image_quoted | rdmapipe --send" |
        release_ssh "$host" \
          "set -o pipefail; rdmapipe --recv | docker image load"
      echo "== RDMA image copy $seed_host -> $host complete =="
    ) &
    image_copy_pids+=("$!")
    image_copy_hosts+=("$host")
  done
  image_copy_failed=0
  for index in "${!image_copy_pids[@]}"; do
    if ! wait "${image_copy_pids[$index]}"; then
      echo "RDMA image copy to ${image_copy_hosts[$index]} failed" >&2
      image_copy_failed=1
    fi
  done
  ((image_copy_failed == 0)) || release_die "concurrent RDMA Spark image distribution failed"
else
  # Without rdmapipe on every build host, stream the same `docker image save |
  # load` over ssh through this host. Slower, same bytes, same identity gates below.
  echo "== rdmapipe missing on a build host: distributing Spark image over ssh (docker save | load) =="
  printf -v spark_image_quoted '%q' "$SPARK_EXPERT_DOCKER_INFERENCE"
  for host in "${RELEASE_BUILD_HOSTS[@]:1}"; do
    echo "== ssh image copy $seed_host -> $host =="
    (
      set -o pipefail
      release_ssh "$seed_host" "set -o pipefail; docker image save $spark_image_quoted" |
        release_ssh "$host" "set -o pipefail; docker image load"
    ) || release_die "ssh Spark image distribution to $host failed"
    echo "== ssh image copy $seed_host -> $host complete =="
  done
fi

coordinator_revision="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_revision" == "$engine_commit" ]] || release_die "coordinator image revision mismatch"
coordinator_version="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.version"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_version" == "$release_version" ]] || release_die "coordinator image version mismatch"
coordinator_sparkinfer_revision="$(
  docker image inspect -f '{{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
  release_die "coordinator image SparkInfer revision mismatch"
if [[ -n "$source_manifest_sha256" ]]; then
  coordinator_source_manifest="$(
    docker image inspect -f '{{index .Config.Labels "io.cuteafd.source-manifest.sha256"}}' \
      "$COORDINATOR_DOCKER_INFERENCE"
  )"
  [[ "$coordinator_source_manifest" == "$source_manifest_sha256" ]] ||
    release_die "coordinator image source manifest mismatch"
fi
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  revision="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$revision" == "$engine_commit" ]] || release_die "$host Spark image revision mismatch: $revision"
  spark_version="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.version\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_version" == "$release_version" ]] ||
    release_die "$host Spark image version mismatch: $spark_version"
  spark_sparkinfer_revision="$(
    release_ssh "$host" \
      "docker image inspect -f '{{index .Config.Labels \"io.cuteafd.sparkinfer.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_sparkinfer_revision" == "$sparkinfer_commit" ]] ||
    release_die "$host Spark image SparkInfer revision mismatch: $spark_sparkinfer_revision"
  if [[ -n "$source_manifest_sha256" ]]; then
    spark_source_manifest="$(
      release_ssh "$host" \
        "docker image inspect -f '{{index .Config.Labels \"io.cuteafd.source-manifest.sha256\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
    )"
    [[ "$spark_source_manifest" == "$source_manifest_sha256" ]] ||
      release_die "$host Spark image source manifest mismatch: $spark_source_manifest"
  fi
  # The advertised role set must equal what was requested and actually built;
  # this is the identity the release launcher verifies before an explicit
  # topology launch, so a mismatch is a hard build failure.
  spark_role_label="$(
    release_ssh "$host" \
      "docker image inspect -f '{{or (index .Config.Labels \"io.cuteafd.spark_tp_roles\") (index .Config.Labels \"io.cuteafd.v41.spark_tp_roles\")}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$spark_role_label" != "<no value>" ]] || spark_role_label=
  # release-spark-tp-roles-postcheck:start
  # One canonicalizer for both sides of the comparison, so a permutation or a
  # stray separator cannot make an equal set look unequal (or the reverse).
  [[ "$(release_spark_tp_roles_canonical "$spark_role_label" \
    "io.cuteafd.spark_tp_roles")" == "$spark_tp_roles" ]] ||
    release_die "$host Spark image advertises expert roles '$spark_role_label', expected exactly '$spark_tp_roles'"
  # release-spark-tp-roles-postcheck:end
done

echo "Build complete."
echo "  revision:    $engine_commit"
echo "  SparkInfer:  $sparkinfer_commit"
if [[ -n "$source_manifest_sha256" ]]; then
  echo "  source:      $source_manifest_sha256"
fi
echo "  expert roles: ${spark_tp_roles:-<none>} ($spark_tp_roles_note)"
echo "  coordinator: $COORDINATOR_DOCKER_INFERENCE"
echo "  spark:       $SPARK_EXPERT_DOCKER_INFERENCE"
echo "  artifacts:   $repo_root/dist"
for leg in coordinator spark; do
  echo "  $leg export cache (hits, misses, bypassed; invocation and program layers):"
  if [[ -f "$repo_root/dist/EXPORT_CACHE_STATS.$leg.jsonl" ]]; then
    python3 "$repo_root/scripts/build/export-cache.py" stats "$repo_root/dist/EXPORT_CACHE_STATS.$leg.jsonl"
  else
    echo "WARNING: $leg export cache receipt unavailable" >&2
  fi
done

# Publish the requested local tags only after both inventories and identity gates pass.
docker tag "$COORDINATOR_DOCKER_INFERENCE" "$complete_coordinator_image"
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  release_ssh "$host" docker tag "$SPARK_EXPERT_DOCKER_INFERENCE" "$complete_spark_image"
done
echo "Release scope $release_scope verified; complete image tags updated."
