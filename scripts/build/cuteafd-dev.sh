#!/usr/bin/env bash
set -euo pipefail

role="${1:-coordinator}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

case "$role" in
  coordinator)
    image="${CUTEAFD_COORDINATOR_DOCKER_DEV:-cuteafd-coordinator-dev}"
    ;;
  expert|spark)
    image="${CUTEAFD_SPARK_EXPERT_DOCKER_DEV:-cuteafd-spark-expert-dev}"
    ;;
  dev|cpu)
    image="${CUTEAFD_COORDINATOR_DOCKER_DEV:-cuteafd-coordinator-dev}"
    ;;
  *)
    echo "unknown role: $role" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
source "$repo_root/scripts/lib/release-common.sh"
release_ensure_dev_image "$image"
python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" \
  --source "$repo_root/third_party/sparkinfer" --lock "$repo_root/third_party/sparkinfer.lock.json"

docker_args=(
  run --rm
  -v "$repo_root:/workspace/cuteafd"
  -v "$hf_home:$hf_home:ro"
  -v "$hf_home:/root/.cache/huggingface:ro"
  -e HF_HOME="$hf_home"
)
# The host hub is usually a symlink into the shared store; mount its target too,
# or the link dangles inside the container.
if [[ -L "$hf_home/hub" ]]; then
  hub_target="$(readlink -f "$hf_home/hub")"
  [[ -d "$hub_target" ]] && docker_args+=(-v "$hub_target:$hub_target:ro")
fi
docker_args+=(
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e CUTEAFD_MODEL_ID="${CUTEAFD_MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
  -e CUTEAFD_MODEL_REVISION="${CUTEAFD_MODEL_REVISION:-}"
)

# Non-root container (scripts/agents/codex-preamble.md): this container compiles
# into the bind-mounted checkout, so it runs as the invoking user and its output
# -- Cargo target dirs, build trees -- is deletable without sudo. `--user`
# bypasses the image's passwd lookup, so the identity and the caches are passed
# explicitly: the image's CARGO_HOME (/opt/cargo) is root-owned and not writable
# by that UID, and a Spark UID has no passwd entry for getpass or Torch Dynamo to
# find. Tools create the home and their cache leaves on demand.
container_home="${CUTEAFD_DEV_CONTAINER_HOME:-/tmp/cuteafd-dev-home}"
docker_args+=(
  --user "$(id -u):$(id -g)"
  -e "HOME=$container_home"
  -e "USER=$(id -un)"
  -e "LOGNAME=$(id -un)"
  -e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor"
  -e "TRITON_CACHE_DIR=$container_home/triton"
  -e "TORCH_EXTENSIONS_DIR=$container_home/torch-extensions"
  -e "XDG_CACHE_HOME=$container_home/.cache"
  -e "B12X_ROCE_CACHE_DIR=$container_home/roce"
  -e "PYTHONPATH=/workspace/cuteafd/third_party/sparkinfer:/workspace/cuteafd/python/reference/cuteafd_reference:/workspace/cuteafd/python/reference"
  -e "CARGO_HOME=$container_home/cargo"
)

# Host build slots (scripts/build/build-slot.sh) exclude through these lock
# files; mount them at the same path so a compile inside takes the real slot.
# Only the slot files are shared (created first, so the container never makes
# a private copy); the hardware locks and the fair queue stay host-only.
lock_dir="$HOME/.cache/cuteafd"
mkdir -p "$lock_dir"
# The slot files live in a private, writable directory at the same path inside
# (a bind-mounted file under an unmounted parent is not writable for flock's
# open in a uid container); each slot file is the host's real lock inode.
container_lock_dir="/tmp/cuteafd-build-locks"
docker_args+=(-e "CUTEAFD_LOCK_DIR=$container_lock_dir" --tmpfs "$container_lock_dir:mode=1777")
for ((slot = 0; slot < ${CUTEAFD_BUILD_SLOTS:-5}; slot++)); do
  slot_name="build.lock"
  ((slot == 0)) || slot_name="build.lock.$slot"
  touch "$lock_dir/$slot_name"
  docker_args+=(-v "$lock_dir/$slot_name:$container_lock_dir/$slot_name")
done
if [[ -n "${CUTEAFD_DEV_TARGET_DIR:-}" ]]; then
  python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$CUTEAFD_DEV_TARGET_DIR"
  mkdir -p "$CUTEAFD_DEV_TARGET_DIR"
  docker_args+=(--mount "type=bind,src=$CUTEAFD_DEV_TARGET_DIR,dst=/opt/cuteafd-target"
                -e CARGO_TARGET_DIR=/opt/cuteafd-target)
fi
# Sealed WIP artifacts for tests that need a built native library and packages:
# mounted read-only at the same absolute path, so package paths resolve unchanged.
if [[ -n "${CUTEAFD_DEV_ARTIFACTS:-}" ]]; then
  [[ -d "$CUTEAFD_DEV_ARTIFACTS" ]] || { echo "CUTEAFD_DEV_ARTIFACTS is not a directory: $CUTEAFD_DEV_ARTIFACTS" >&2; exit 2; }
  artifacts="$(cd "$CUTEAFD_DEV_ARTIFACTS" && pwd -P)"
  docker_args+=(--mount "type=bind,src=$artifacts,dst=$artifacts,readonly"
                -e CUTEAFD_DEV_ARTIFACTS="$artifacts")
fi
# Memory cap for the whole container (e.g. 16g), so a test on a shared Spark's
# unified memory cannot crowd out the host's services; swap is capped equally.
if [[ -n "${CUTEAFD_DEV_MEMORY:-}" ]]; then
  [[ "$CUTEAFD_DEV_MEMORY" =~ ^[0-9]+[kmg]?$ ]] || { echo "CUTEAFD_DEV_MEMORY must be a docker size such as 16g: $CUTEAFD_DEV_MEMORY" >&2; exit 2; }
  docker_args+=(--memory "$CUTEAFD_DEV_MEMORY" --memory-swap "$CUTEAFD_DEV_MEMORY")
fi
source "$repo_root/scripts/build/compiler-cache.sh"
source "$repo_root/scripts/build/build-caches.sh"
export CUTEAFD_KACHE="${CUTEAFD_KACHE-}" CUTEAFD_SCCACHE_CUDA="${CUTEAFD_SCCACHE_CUDA-0}"
cuteafd_build_cache_defaults
compiler_cache_args=()
cache_plan="$(cuteafd_build_cache_docker_args "${CUTEAFD_DEV_TARGET_DIR:-$HOME/.cache/cuteafd/builds/dev-cache-fallback}" "$container_home" "$(python3 "$repo_root/scripts/build/dev-toolchain.py")")" || exit 2
mapfile -t compiler_cache_args <<<"$cache_plan"
docker_args+=("${compiler_cache_args[@]}")

if [[ "$role" == "expert" || "$role" == "spark" ]]; then
  source "$repo_root/scripts/lib/cdi-gpus.sh"
  cuteafd_cdi_gpu_args all || exit 2
  docker_args+=("${CUTEAFD_CDI_ARGS[@]}")
elif [[ "$role" != cpu ]]; then
  source "$repo_root/scripts/lib/release-common.sh"
  release_load_config "${CUTEAFD_CONFIG:-$repo_root/cuteafd.config}"
  release_need nvidia-smi
  release_resolve_coordinator_gpu_identity
  # Under hwlock, the GPU this container gets must be one the caller holds: the
  # config (not the environment) selects it, so a mismatch is a wrong config.
  if [[ -n "${CUTEAFD_HW_LOCKS_HELD:-}" && ",$CUTEAFD_HW_LOCKS_HELD," != *",gpu$RELEASE_COORDINATOR_GPU_HOST_INDEX,"* ]]; then
    echo "coordinator GPU $RELEASE_COORDINATOR_GPU_HOST_INDEX is not among held locks ($CUTEAFD_HW_LOCKS_HELD); set COORDINATOR_GPU in CUTEAFD_CONFIG" >&2
    exit 2
  fi
  source "$repo_root/scripts/lib/cdi-gpus.sh"
  cuteafd_cdi_gpu_args "$RELEASE_COORDINATOR_GPU_UUID" || exit 2
  docker_args+=(
    "${CUTEAFD_CDI_ARGS[@]}"
    -e CUDA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
    -e NVIDIA_VISIBLE_DEVICES="$RELEASE_COORDINATOR_GPU_UUID"
  )
fi

if [[ "$repo_root" != "/workspace/cuteafd" ]]; then
  docker_args+=(-v "$repo_root:$repo_root")
fi
# Worktree and submodule gitfiles resolve through the original checkout's metadata.
# A staged copy without Git metadata (verified against the lock above) mounts as is.
if git_common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
    && [[ "$git_common_dir" != "$repo_root/.git" ]]; then
  docker_args+=(-v "$git_common_dir:$git_common_dir:ro")
fi
# Relative gitfiles (worktrees, submodules) resolve only from the checkout's real
# path, not the /workspace/cuteafd alias: point the pinned SparkInfer tools at the
# real-path copy (mounted above), still verified against the lock file.
if [[ "$repo_root" != "/workspace/cuteafd" ]]; then
  docker_args+=(-e "CUTEAFD_SPARKINFER_SOURCE_DIR=$repo_root/third_party/sparkinfer"
                -e "CUTEAFD_SPARKINFER_LOCK_FILE=$repo_root/third_party/sparkinfer.lock.json")
fi

if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-it)
fi

if [[ "$role" == "expert" || "$role" == "spark" ]]; then
  docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
  if [[ -e /dev/infiniband ]]; then
    docker_args+=(--device=/dev/infiniband)
  fi
fi

if [[ "$role" == "coordinator" || "$role" == "dev" ]]; then
  if [[ "${CUTEAFD_DOCKER_HOST_NETWORK:-0}" == "1" ]]; then
    docker_args+=(--net=host --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK)
    if [[ -e /dev/infiniband ]]; then
      docker_args+=(--device=/dev/infiniband)
    fi
  fi
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

if [[ -n "${CUTEAFD_KACHE:-}" || "${CUTEAFD_SCCACHE_CUDA:-0}" == 1 ]]; then
  # Apply inside the matching toolchain, then preserve the caller's argv exactly.
  set -- bash -c 'source /workspace/cuteafd/scripts/build/compiler-cache.sh;
    cuteafd_compiler_cache_setup "${CARGO_TARGET_DIR:-$HOME/compiler-cache-build}";
    exec "$@"' cuteafd-cache "$@"
fi
exec docker "${docker_args[@]}" "$image" "$@"
