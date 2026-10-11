#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/build/compiler-cache.sh"
source "$repo_root/scripts/build/build-caches.sh"
cuteafd_build_cache_defaults
bf16_families="${CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES:-}"
audio_aot="${CUTEAFD_RELEASE_AUDIO_AOT:-ON}"
case "$audio_aot" in ON|OFF) ;; *) release_die "CUTEAFD_RELEASE_AUDIO_AOT must be ON or OFF, got: $audio_aot" ;; esac
native_build_jobs="${CUTEAFD_RELEASE_NATIVE_BUILD_JOBS:-}"
[[ -z "$native_build_jobs" || "$native_build_jobs" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_NATIVE_BUILD_JOBS must be a positive integer"
native_build_env_args=()
[[ -z "$native_build_jobs" ]] || native_build_env_args=(-e "CMAKE_BUILD_PARALLEL_LEVEL=$native_build_jobs")
bf16_family_pattern='^(mimo|mimop|mimof|glm|glmf|qwen4)(;(mimo|mimop|mimof|glm|glmf|qwen4))*$'
[[ -z "$bf16_families" || "$bf16_families" =~ $bf16_family_pattern ]] ||
  release_die "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES must be a semicolon list of mimo, mimop, mimof, glm, glmf or qwen4"

usage() {
  cat <<'EOF'
Usage: ./build.sh [--config FILE] [--spark-hosts HOST,...] [--dry-run]

The coordinator and Spark legs build concurrently: the coordinator image locally
and the Spark image natively over SSH on the first configured Spark. Set
CUTEAFD_RELEASE_SEQUENTIAL=1 to build coordinator then Spark for debugging.
Each leg has its own log file; both must finish before artifacts are exported.
It exports both release artifact sets to dist/
and distributes the Spark inference image to all configured Spark hosts.
Use --spark-hosts ostrich,dodo to build and distribute only on available hosts;
this does not change the serving topology.
Release images are universal by default: the ARM64 (SM121) Spark expert image
carries the TP2, TP3 and TP6 replicated-group shards on top of the always-built
TP4 shard, so one pair serves every approved native topology and ./run.sh selects
the mode with SPARK_TP/SPARK_EP. The x86_64 coordinator image needs no Spark role.
Set CUTEAFD_RELEASE_SPARK_TP_ROLES to an explicit subset (for example tp6, or empty
for the historical TP4-only shard) for a bounded topology A/B or a legacy rebuild.
--dry-run validates the configuration, host set and role plan without touching
Docker, SSH, submodules, hardware locks or any image.
The build takes ~/.cache/cuteafd/build.lock for its whole run, so two release
builds serialize; CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS bounds that wait
(default 1200). It never takes sparks.lock or gpu1.lock, and it does not pin
GPU0: a compile, a download or an image assembly must not block serving. The two
AOT exports take no lock at all. The coordinator export picks the least-used RTX
with at most CUTEAFD_RELEASE_IDLE_GPU_LIMIT_MIB (default 512) in use, waiting
bounded by CUTEAFD_RELEASE_IDLE_WAIT_SECONDS (default 300), pins it by UUID,
stops the export if the device grows past CUTEAFD_RELEASE_EXPORT_GPU_LIMIT_MIB
(default 8192) while it runs, and logs host, device and time; a Spark export
waits for a host with no serving worker container and at least
CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB (default 100) of free CUDA memory. Set
CUTEAFD_RELEASE_EXPORT_TIMEOUT_SECONDS (default 7200) to bound each export, and
CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS (default 15) to change how often the
export's device is checked.
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

Families beyond V4.1 are opt-in. CUTEAFD_RELEASE_{DSV4,GLM,MIMO,GLMF,QWEN4}_AOT=ON
put each family's coordinator programs in the coordinator image, and
CUTEAFD_RELEASE_EXPERT_FAMILIES (FAMILY:ROLE list, ';'-separated) adds routed-expert
packages to both images, each keeping its architecture's entries.
CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES (e.g. mimo, or mimo;glm) adds optional
BF16-input Spark siblings for the requested FAMILY:fp8 packages. It does not
change serving precision defaults. Empty builds the existing artifact set.
CUTEAFD_RELEASE_AUDIO_AOT=OFF leaves the optional audio tower out of both images
(default ON: AUDIO=auto serves it for the qualified MiMo V2.6 checkpoints).
CUTEAFD_RELEASE_MIMO_GEOMETRIES picks the MiMo program geometries (default
mimo,mimo2,mimop,mimop2: V2 Flash, V2.6 Pro and their two-GPU head splits). p7's set, everything scripts/launch/run-family.sh
serves (V4 Pro EXL3 K2, GLM 5.3 EXL3 K4 and FP8, GLM 5.3 Flash, MiMo V2 Flash
FP8, MiMo V2.6 Pro MXFP4 (tp1 coordinator, tp6/tp2 Spark), Qwen 3.8 Flash Next
EXL3 K4.25):
    CUTEAFD_RELEASE_GLM_AOT=ON CUTEAFD_RELEASE_GLMF_AOT=ON CUTEAFD_RELEASE_MIMO_AOT=ON
    CUTEAFD_RELEASE_QWEN4_AOT=ON CUTEAFD_RELEASE_MIMO_GEOMETRIES=mimo,mimo2,mimop,mimop2
    CUTEAFD_RELEASE_EXPERT_FAMILIES='dsv4f:rtx_backbone;dsv4f:rtx_tp2;dsv4p:rtx_backbone;dsv4p:rtx_tp2;dsv4p:exl3-k23;glm:exl3-k45;glm:fp8;mimo:fp8;mimop:fp8;qwen4:exl3-k45'
(p6 was the same without QWEN4, mimop, mimop:fp8 and qwen4:exl3-k45.)

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

# Spark expert roles are universal by default: the ARM64 (SM121) image carries the
# TP2, TP3 and TP6 replicated-group shards on top of the always-built TP4 shard,
# so one published pair serves every approved native topology (TP4EP1, TP2EP2,
# TP2EP3, TP3EP2, TP6EP1) and ./run.sh selects the mode. The x86_64 coordinator
# needs no Spark role: roles are expert-only, so it is topology-independent.
# CUTEAFD_RELEASE_SPARK_TP_ROLES is an explicit SUBSET override for a bounded
# topology A/B or a legacy TP4-only rebuild (empty).
# release-spark-tp-roles:start
release_universal_spark_tp_roles="$(release_spark_tp_roles_default)"
# `${VAR-default}`, not `:-`, so an explicitly empty override stays the legacy
# TP4-only request rather than falling back to the universal set.
spark_tp_roles="$(release_spark_tp_roles_canonical \
  "${CUTEAFD_RELEASE_SPARK_TP_ROLES-$release_universal_spark_tp_roles}")"
[[ "$spark_tp_roles" == "$release_universal_spark_tp_roles" ]] ||
  echo "== NON-UNIVERSAL expert role subset '${spark_tp_roles:-<none>}'; this pair cannot serve every approved native topology =="
# Repeated by --dry-run and the build summary so a subset build is never mistaken
# for the universal default.
spark_tp_roles_note='universal, covers every approved native topology'
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
# Settings that only size waits and guards, resolved before the dry-run so
# --dry-run reports the plan this build would follow. Nothing here takes a lock,
# a device or a container.
release_build_lock_timeout="${CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS:-1200}"
[[ "$release_build_lock_timeout" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS must be positive seconds"
release_idle_wait_seconds="${CUTEAFD_RELEASE_IDLE_WAIT_SECONDS:-300}"
[[ "$release_idle_wait_seconds" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_IDLE_WAIT_SECONDS must be positive seconds"
release_idle_gpu_limit_mib="${CUTEAFD_RELEASE_IDLE_GPU_LIMIT_MIB:-512}"
[[ "$release_idle_gpu_limit_mib" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_IDLE_GPU_LIMIT_MIB must be positive MiB"
export_gpu_limit_mib="${CUTEAFD_RELEASE_EXPORT_GPU_LIMIT_MIB:-8192}"
[[ "$export_gpu_limit_mib" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_EXPORT_GPU_LIMIT_MIB must be positive MiB"
release_export_gpu_poll_seconds="${CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS:-15}"
[[ "$release_export_gpu_poll_seconds" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS must be positive seconds"
spark_export_min_free_gib="${CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB:-100}"
[[ "$spark_export_min_free_gib" =~ ^[1-9][0-9]*$ ]] ||
  release_die "CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB must be positive GiB"
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
  echo "  release tag: $release_version"
  echo "  V41 Spark expert roles: ${spark_tp_roles:-<legacy TP4 only>} ($spark_tp_roles_note)"
  echo "  coordinator image: $COORDINATOR_DOCKER_INFERENCE"
  echo "  spark image: $SPARK_EXPERT_DOCKER_INFERENCE"
  echo "  ssh config: ${release_ssh_config:-<stock>}"
  echo "  release build root: ${release_build_root:-<container /tmp>}"
  echo "  build container user: $(id -un) ($(id -u):$(id -g))"
  echo "  build container home: $(release_build_container_home "$release_build_root")"
  echo "  build lock: $HOME/.cache/cuteafd/build.lock (waited up to ${release_build_lock_timeout}s; no hardware lock is taken)"
  echo "  AOT export GPU guard: least-used RTX with <=${release_idle_gpu_limit_mib} MiB used, waited ${release_idle_wait_seconds}s, pinned by UUID, stopped past ${export_gpu_limit_mib} MiB"
  echo "  Spark AOT export guard: no serving worker and >=${spark_export_min_free_gib} GiB free CUDA memory, same wait"
  cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash" dry >/dev/null
  echo "  Spark cache plan (host HOME shown as template; states/admission rechecked on Spark, no SSH):"
  cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash" dry aarch64 >/dev/null
  exit 0
fi

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

# One build at a time, and never a hardware lock: a build is CPU work plus two
# short AOT exports that merely need *a* GPU (see the guard below). Holding
# sparks.lock/gpu1.lock across a 15-minute compile, a crate download or an image
# assembly would block serving for no reason, so the build takes
# ~/.cache/cuteafd/build.lock instead. It is acquired before the submodule
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

# release-aot-export-guard:start
# AOT exports run outside the hardware locks (USING_AGENTS.md): the CuTe/Triton
# exporters query the device at compile time, so the export container needs a
# GPU, but not a lock, and must not wait behind or block a serving job.
#
# The coordinator export picks the least-used RTX with at most 512 MiB in use,
# waiting bounded by CUTEAFD_RELEASE_IDLE_WAIT_SECONDS (default 300, giving way
# but not failing if a measurement is finishing). The choice is pinned by UUID
# both for docker and inside the container, so no in-container index can drift
# onto another device. A watchdog then stops the export if its own device is
# taken over mid-compile. Host, device and time are logged so a concurrent
# measurement that saw an unexpected export can be explained after the fact.
release_select_idle_export_gpu() {
  local deadline=$((SECONDS + release_idle_wait_seconds))
  local last_report="" index uuid used best_index="" best_uuid="" best_used
  while :; do
    best_used=$((release_idle_gpu_limit_mib + 1))
    while IFS=',' read -r index uuid used; do
      index="${index// /}"
      uuid="${uuid// /}"
      used="${used// /}"
      [[ "$index" =~ ^[0-9]+$ && "$uuid" == GPU-* && "$used" =~ ^[0-9]+$ ]] || continue
      (( used <= release_idle_gpu_limit_mib )) || continue
      if (( used < best_used )); then
        best_index="$index"
        best_uuid="$uuid"
        best_used="$used"
      fi
    done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "$best_uuid" ]]; then
      printf '%s\n' "$best_index $best_uuid $best_used"
      return 0
    fi
    last_report="$(
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
        paste -sd';' - || true
    )"
    (( SECONDS < deadline )) || break
    echo "$(hostname): waiting for an idle RTX for the AOT export ($(date -Is)): ${last_report:-nvidia-smi reported no devices}" >&2
    sleep 15
  done
  release_die "no idle RTX (at most ${release_idle_gpu_limit_mib} MiB used) for the AOT export within ${release_idle_wait_seconds}s on $(hostname): ${last_report:-nvidia-smi unavailable}; stop the concurrent run or raise CUTEAFD_RELEASE_IDLE_WAIT_SECONDS"
}

# Stop the export when its device stops being idle. Backgrounded for the length
# of the export; its stdout/stderr carry the reason the build failed.
release_watch_export_gpu() {
  local uuid="$1" container="$2"
  local used
  while sleep "$release_export_gpu_poll_seconds"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$uuid" 2>/dev/null || true)"
    used="${used// /}"
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    (( used <= export_gpu_limit_mib )) || {
      echo "$(hostname): AOT export device $uuid now has ${used} MiB in use (limit ${export_gpu_limit_mib} MiB): a concurrent job took it; stopping $container" >&2
      docker rm -f "$container" >/dev/null 2>&1 || true
      return 1
    }
  done
}

release_stop_export_watchdog() {
  local pid="${export_watchdog_pid:-}"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  export_watchdog_pid=""
}
# release-aot-export-guard:end

release_need nvidia-smi

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
remote_free_kib="$(
  release_ssh "$seed_host" bash -s <<'REMOTE'
df -Pk "$HOME" | awk 'NR == 2 { print $4 }'
REMOTE
)"
((remote_free_kib >= 60 * 1024 * 1024)) || release_die "$seed_host build needs at least 60 GiB free"

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
  ((remote_root_free_kib >= 60 * 1024 * 1024)) ||
    release_die "$seed_host release build root $release_build_root needs at least 60 GiB free"
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
trap 'release_stop_export_watchdog; release_export_cleanup; rm -rf "$release_source_dir"' EXIT
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
    --source "$repo_root" --arch amd64 --local-tag "$COORDINATOR_DOCKER_DEV" \
    --engine-commit "$engine_commit" --mode "$release_dev_image_source" \
    --output "$release_dev_reuse_manifest")"
fi
python3 "$repo_root/scripts/build/select-dev-image.py" labels --manifest "$release_dev_reuse_manifest" >"$release_leg_log_dir/coordinator-dev-labels"
mapfile -t release_dev_reuse_label_args <"$release_leg_log_dir/coordinator-dev-labels"

echo "== compiling coordinator release artifacts in GPU-enabled development container =="
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
export_gpu_pick="$(release_select_idle_export_gpu)" || exit 2
read -r export_gpu_index export_gpu_uuid export_gpu_used <<<"$export_gpu_pick"
echo "== coordinator AOT export on $(hostname) GPU $export_gpu_index ($export_gpu_uuid) at $(date -Is): ${export_gpu_used} MiB in use, no hardware lock =="
release_build_user_args=()
mapfile -t release_build_user_args < <(release_build_container_user_args_render "$release_build_root")
# Killing a docker client does not stop its container; the leg's EXIT trap owns
# both the named export and its watchdog, including failures before docker run.
# --foreground keeps timeout and its client inside the cancellable leg group.
release_watch_export_gpu "$export_gpu_uuid" "$coordinator_export_container" &
export_watchdog_pid=$!
coordinator_export_status=0
compiler_cache_args=()
cache_plan="$(cuteafd_build_cache_docker_args "${release_build_root:-$release_source_parent}" "$(release_build_container_home "$release_build_root")" "$release_toolchain_hash")" || release_die "cache plan failed"
mapfile -t compiler_cache_args <<<"$cache_plan"
timeout "$export_timeout" --foreground docker run --rm --name "$coordinator_export_container" \
  --gpus "device=$export_gpu_uuid" \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e "CUDA_VISIBLE_DEVICES=$export_gpu_uuid" \
  -e "NVIDIA_VISIBLE_DEVICES=$export_gpu_uuid" \
  "${release_build_user_args[@]}" \
  "${compiler_cache_args[@]}" \
  -e "CUTEAFD_RELEASE_EXPERT_FAMILIES=${CUTEAFD_RELEASE_EXPERT_FAMILIES:-}" \
  -e "CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS=${CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS:-}" \
  -e "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$bf16_families" \
  -e "CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot" \
  ${native_build_env_args[@]+"${native_build_env_args[@]}"} \
  -e "CUTEAFD_RELEASE_DSV4_MAX_CONTEXT=${CUTEAFD_RELEASE_DSV4_MAX_CONTEXT:-1048576}" \
  -e "CUTEAFD_RELEASE_GLM_AOT=${CUTEAFD_RELEASE_GLM_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_MIMO_AOT=${CUTEAFD_RELEASE_MIMO_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_MIMO_GEOMETRIES=${CUTEAFD_RELEASE_MIMO_GEOMETRIES:-mimo,mimo2,mimop,mimop2}" \
  -e "CUTEAFD_RELEASE_GLMF_AOT=${CUTEAFD_RELEASE_GLMF_AOT:-OFF}" \
  -e "CUTEAFD_RELEASE_QWEN4_AOT=${CUTEAFD_RELEASE_QWEN4_AOT:-OFF}" \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$release_source_dir:/source:ro" \
  -v "$artifact_dir:/output" \
  "$COORDINATOR_DOCKER_DEV" \
  /source/scripts/build/build-release-artifacts.sh /source coordinator 120 /output ||
  coordinator_export_status=$?
release_stop_export_watchdog
(( coordinator_export_status == 0 )) ||
  release_die "coordinator AOT export failed (exit $coordinator_export_status); it ran on $(hostname) GPU $export_gpu_index ($export_gpu_uuid) at $(date -Is)"
rm -rf "$release_source_dir"
release_source_dir=""

echo "== building coordinator inference image: $COORDINATOR_DOCKER_INFERENCE =="
DOCKER_BUILDKIT=1 docker build \
  "${release_source_label_args[@]}" \
  "${release_dev_reuse_label_args[@]}" \
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
  timeout "$export_timeout" --foreground ssh "${release_ssh_opts[@]}" "$seed_host" setsid --wait bash -s -- \
  "$remote_dir" "$SPARK_EXPERT_DOCKER_DEV" "$SPARK_EXPERT_DOCKER_INFERENCE" \
  "$engine_commit" "$sparkinfer_commit" "$release_version" \
  "$EXL3_PAIRED_TP4" "${source_manifest_sha256:-__legacy__}" "$(r="${spark_tp_roles//;/,}"; echo "${r:-__legacy__}")" \
  "${release_build_root:-__legacy__}" \
  "$(f="${CUTEAFD_RELEASE_EXPERT_FAMILIES:-}"; f="${f//;/,}"; echo "${f:-__legacy__}")" \
  "$(f="${bf16_families//;/,}"; echo "${f:-__legacy__}")" \
  "$phase" "$export_container-expert" "${native_build_jobs:-__legacy__}" \
  "$(printf '%q' "${CUTEAFD_KACHE_SPARK:-__legacy__}")" \
  "$(printf '%q' "${CUTEAFD_KACHE_REMOTE:-__legacy__}")" \
  "$(printf '%q' "${CUTEAFD_KACHE_SPARK_CACHE_DIR:-__legacy__}")" "${CUTEAFD_SCCACHE_CUDA:-0}" "${release_dev_image_source:-registry}" "$audio_aot" "${CUTEAFD_BUILD_CACHES:-on}" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
dev_image="$2"
inference_image="$3"
engine_commit="$4"
sparkinfer_commit="$5"
release_version="$6"
exl3_paired_tp4="$7"
# SSH reconstructs a shell command and can omit an empty argument, so every
# optional trailing value is passed as a non-empty sentinel and decoded here.
# Both source_manifest_sha256 and spark_tp_roles are optional: sentinels keep
# the two from shifting into each other's position.
source_manifest_sha256="${8-__legacy__}"
# The role list travels as a comma list so a remote shell cannot split it at a
# semicolon; it is restored to the CMake semicolon list here.
spark_tp_roles="${9-__legacy__}"
# The relocated build root is optional and uses the same sentinel convention.
# It is already created and filesystem-guarded by the caller, on this host.
release_build_root="${10-__legacy__}"
# Extra routed-expert kernel families (FAMILY:ROLE), comma-encoded like the roles.
expert_families="${11-__legacy__}"
[[ "$expert_families" != "__legacy__" ]] || expert_families=
expert_families="${expert_families//,/;}"
# Optional BF16-input siblings travel with a sentinel and the same comma encoding.
bf16_families="${12-__legacy__}"
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
export CUTEAFD_BUILD_CACHES="${22:-on}"
[[ "${16:-__legacy__}" == __legacy__ ]] || export CUTEAFD_KACHE="${16}"
[[ "${18:-__legacy__}" == __legacy__ ]] || export CUTEAFD_KACHE_CACHE_DIR="${18}"
export CUTEAFD_SCCACHE_CUDA="${19:-1}"
source scripts/build/build-caches.sh
cuteafd_build_cache_defaults
phase="${13:?}"
# The audio tower switch reaches the remote leg as its own argument (ON/OFF).
audio_aot="${21:-ON}"
export_container="${14:?}"
native_build_jobs="${15-__legacy__}"
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
  docker rm -f "$export_container" "$export_container-probe" >/dev/null 2>&1 || true
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
  --source "$remote_dir" --arch arm64 --local-tag "$dev_image" \
  --engine-commit "$engine_commit" --mode "${20:-registry}" --output "$dev_manifest"
else
dev_image="$(python3 scripts/build/select-dev-image.py image --manifest "$dev_manifest")"
fi
if [[ "$phase" == export ]]; then
# AOT export outside the hardware locks (USING_AGENTS.md): it needs a GPU to
# query at compile time, not a lock, and it may not run beside a serving job on
# a Spark whose 121 GiB are the serving budget. Wait bounded for an idle host --
# no serving worker container and at least 100 GiB of CUDA memory free, read
# with CUDA because GB10 nvidia-smi memory reads are N/A -- then watch the host
# while the export runs and stop it if a serving container appears. Host and
# time are logged so a concurrent measurement can be explained afterwards.
spark_export_wait="${CUTEAFD_RELEASE_IDLE_WAIT_SECONDS:-300}"
[[ "$spark_export_wait" =~ ^[1-9][0-9]*$ ]] ||
  { echo "CUTEAFD_RELEASE_IDLE_WAIT_SECONDS must be positive seconds" >&2; exit 2; }
spark_export_min_free_gib="${CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB:-100}"
[[ "$spark_export_min_free_gib" =~ ^[1-9][0-9]*$ ]] ||
  { echo "CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB must be positive GiB" >&2; exit 2; }
spark_export_min_free_bytes=$((spark_export_min_free_gib * 1024 * 1024 * 1024))
spark_export_poll_seconds="${CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS:-15}"
[[ "$spark_export_poll_seconds" =~ ^[1-9][0-9]*$ ]] ||
  { echo "CUTEAFD_RELEASE_EXPORT_GPU_POLL_SECONDS must be positive seconds" >&2; exit 2; }
spark_export_deadline=$((SECONDS + spark_export_wait))
# Release workers are named cuteafd-spark-expert-HOST-PORT; a persistent WIP slot
# (cuteafd-spark-expert-wip) is idle by design and does not block an export --
# the CUDA memory gate below is what catches a WIP container actually in use.
spark_serving_containers() {
  docker ps -q --filter 'name=^cuteafd-spark-expert-[a-z0-9_.-]+-[0-9]+$' 2>/dev/null || true
}
while :; do
  spark_serving="$(spark_serving_containers)"
  if [[ -n "$spark_serving" ]]; then
    spark_wait_reason="serving container(s) running: $(tr '\n' ' ' <<<"$spark_serving")"
  else
    spark_cuda_free="$(docker run --rm --name "$export_container-probe" --gpus all \
      -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e HOME=/tmp/home \
      -e TORCHINDUCTOR_CACHE_DIR=/tmp/home/torchinductor -e TRITON_CACHE_DIR=/tmp/home/triton \
      --entrypoint python3 "$dev_image" \
      -c 'import torch; print(torch.cuda.mem_get_info()[0])' 2>/dev/null || true)"
    if [[ "$spark_cuda_free" =~ ^[0-9]+$ ]] && (( spark_cuda_free >= spark_export_min_free_bytes )); then
      echo "== Spark AOT export on $(hostname) at $(date -Is): ${spark_cuda_free} bytes CUDA memory free, no serving container, no hardware lock =="
      break
    fi
    spark_wait_reason="CUDA memory free ${spark_cuda_free:-unreadable}, needs ${spark_export_min_free_bytes} bytes"
  fi
  (( SECONDS < spark_export_deadline )) || {
    echo "$(hostname): no idle Spark within ${spark_export_wait}s ($spark_wait_reason); the export needs no serving container and ${spark_export_min_free_gib} GiB free CUDA memory" >&2
    exit 2
  }
  echo "$(hostname): waiting for an idle Spark export host ($(date -Is)): $spark_wait_reason"
  sleep 15
done
mkdir -p .cuteafd-release-image
# This host's invoking user, not root: the container writes a Cargo target dir
# and staging into bind-mounted host paths (see the coordinator note in build.sh).
# A --user UID has no passwd entry here, so the identity and the cache roots are
# passed explicitly; the artifact compiler creates them before Cargo runs.
container_home=/tmp/cuteafd-home
[[ -z "$release_build_root" ]] || container_home="$release_build_root/container-home"
cache_plan="$(cuteafd_build_cache_docker_args "${release_build_root:-$HOME/.cache/cuteafd/builds/release-cache-fallback}" "$container_home" "$(python3 scripts/build/dev-toolchain.py)")" || exit 2
mapfile -t compiler_cache_args <<<"$cache_plan"
# The export is stopped by name from three directions: the contention watchdog
# while it runs, and this shell's own EXIT/HUP when the caller's timeout kills
# the ssh client (a dying docker client does not stop its container).
docker run --rm --name "$export_container" \
  "${compiler_cache_args[@]}" \
  --user "$(id -u):$(id -g)" \
  -e "HOME=$container_home" \
  -e "USER=$(id -un)" \
  -e "LOGNAME=$(id -un)" \
  -e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor" \
  -e "CARGO_HOME=$container_home/cargo" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e "CUTEAFD_RELEASE_EXL3_PAIRED_TP4=$exl3_paired_tp4" \
  -e "CUTEAFD_RELEASE_SPARK_TP_ROLES=$spark_tp_roles" \
  -e "CUTEAFD_RELEASE_EXPERT_FAMILIES=$expert_families" \
  -e "CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS=${CUTEAFD_RELEASE_GENERIC_SPARK_COUNTS:-}" \
  -e "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$bf16_families" \
  -e "CUTEAFD_RELEASE_AUDIO_AOT=$audio_aot" \
  ${native_build_env_args[@]+"${native_build_env_args[@]}"} \
  ${release_build_root_args[@]+"${release_build_root_args[@]}"} \
  -v "$remote_dir:/source:ro" \
  -v "$remote_dir/.cuteafd-release-image:/output" \
  "$dev_image" \
  /source/scripts/build/build-release-artifacts.sh /source expert 121 /output &
spark_export_container_pid=$!
(
  while sleep "$spark_export_poll_seconds"; do
    spark_serving="$(spark_serving_containers)"
    [[ -z "$spark_serving" ]] || {
      echo "$(hostname): a serving container started during the AOT export ($(tr '\n' ' ' <<<"$spark_serving")): stopping $export_container" >&2
      docker rm -f "$export_container" >/dev/null 2>&1 || true
      exit 1
    }
  done
) &
spark_export_watchdog_pid=$!
spark_export_status=0
wait "$spark_export_container_pid" || spark_export_status=$?
kill "$spark_export_watchdog_pid" 2>/dev/null || true
wait "$spark_export_watchdog_pid" 2>/dev/null || true
(( spark_export_status == 0 )) ||
  { echo "$(hostname): the Spark AOT export failed (exit $spark_export_status) at $(date -Is)" >&2; exit "$spark_export_status"; }
fi
if [[ "$phase" == image ]]; then
python3 scripts/build/select-dev-image.py labels --manifest "$dev_manifest" >"$dev_manifest.labels"
mapfile -t release_dev_reuse_label_args <"$dev_manifest.labels"
DOCKER_BUILDKIT=1 docker build \
  "${release_source_label_args[@]}" \
  "${release_dev_reuse_label_args[@]}" \
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
docker rm -f "$container" "$container-probe" >/dev/null 2>&1 || true
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
coordinator_container="$(docker create "$COORDINATOR_DOCKER_INFERENCE")"
trap 'docker rm -f "$coordinator_container" >/dev/null 2>&1 || true' EXIT
docker cp "$coordinator_container:/opt/cuteafd/bin/cuteafd" "$repo_root/dist/coordinator/cuteafd"
docker cp "$coordinator_container:/opt/cuteafd/lib/libcuteafd_native.so" "$repo_root/dist/coordinator/libcuteafd_native.so"
docker cp "$coordinator_container:/opt/cuteafd/lib/exl3" "$repo_root/dist/coordinator/exl3"
docker cp "$coordinator_container:/opt/cuteafd/lib/fp8" "$repo_root/dist/coordinator/fp8"
docker cp "$coordinator_container:/opt/cuteafd/share/V41_EXPERT_AOT.json" "$repo_root/dist/coordinator/V41_EXPERT_AOT.json"
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
trap - EXIT

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
    python3 "$repo_root/python/tools/aot/package_exl3_aot.py" verify --package "$repo_root/dist/$role/exl3" --sparkinfer-revision "$sparkinfer_commit"
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
    coordinator/cuteafd coordinator/libcuteafd_native.so coordinator/V41_EXPERT_AOT.json coordinator/V41_EXPERT_TP_AOT.json coordinator/V41_FP8_AOT.json \
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
for host in "${RELEASE_BUILD_HOSTS[@]}"; do
  if ! release_ssh "$host" "command -v rdmapipe >/dev/null 2>&1"; then
    rdmapipe_ready=0
    break
  fi
done
if ((rdmapipe_ready)); then
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
  release_die "rdmapipe is required on every Spark build host to distribute the Spark image"
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
