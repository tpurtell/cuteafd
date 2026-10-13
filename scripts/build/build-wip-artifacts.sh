#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-wip-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH BUILD_DIR OUTPUT_DIR [all|rust|native|export]" >&2
  exit 2
}

[[ $# -eq 5 || $# -eq 6 ]] || usage
phase="${6:-all}"
# Name the failing line and command: a container exec only reports an exit code.
trap 'status=$?; echo "build-wip-artifacts.sh: line $LINENO: \"$BASH_COMMAND\" exited $status (role ${2:-?}, phase $phase)" >&2' ERR
# export: only the native/AOT outputs and their native-store entry, no Cargo and
# no daemon (an export host such as oliver; the slot's own native phase then
# restores the entry with CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1).
case "$phase" in all|rust|native|export) ;; *) usage ;; esac
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
build_dir="$(realpath -m "$4")"
output_dir="$(realpath -m "$5")"
# Check source and destinations before creating files or invoking Cargo. This
# also rejects NTFS exposed under a container alias such as /scratch.
python3 "$(dirname "$0")/assert-build-filesystem.py" "$source_dir" "$build_dir" "$output_dir" "${CARGO_HOME:-$HOME/.cargo}" "${TMPDIR:-/tmp}"

case "$role" in
  coordinator)
    coordinator_aot=ON
    xgrammar=ON
    ;;
  expert)
    coordinator_aot=OFF
    xgrammar=OFF
    ;;
  *)
    echo "ROLE must be coordinator or expert" >&2
    exit 2
    ;;
esac
spark_tp_roles="${CUTEAFD_WIP_SPARK_TP_ROLES:-}"
if [[ -n "$spark_tp_roles" ]]; then
  IFS=';' read -ra spark_tp_role_list <<<"$spark_tp_roles"
  for spark_tp_role in "${spark_tp_role_list[@]}"; do
    case "$spark_tp_role" in
      tp2|tp3|tp6) ;;
      *) echo "CUTEAFD_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got: $spark_tp_role" >&2; exit 2 ;;
    esac
  done
  [[ "$role" == expert ]] ||
    { echo "CUTEAFD_WIP_SPARK_TP_ROLES is only valid for the expert role" >&2; exit 2; }
fi

# Extra routed-expert kernel families (FAMILY:ROLE list, e.g. dsv4f:spark),
# validated by native/cmake/shared/expert_families.cmake. Empty keeps the V4.1 image.
# Both builds take the same list; CMake keeps the entries for its architecture.
expert_families="${CUTEAFD_WIP_EXPERT_FAMILIES:-}"
[[ "$role" != expert || ";$expert_families;" != *";glmfdense:"* ]] || { echo "CUTEAFD_WIP_EXPERT_FAMILIES: glmfdense is coordinator-only; remove it from the Spark expert scope" >&2; exit 2; }
# Optional Spark siblings retain unquantized BF16 expert inputs. Empty keeps
# the existing artifact set; requesting one also requires its main FP8 family.
bf16_families="${CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES:-}"
IFS=';' read -ra bf16_family_list <<<"$bf16_families"
for bf16_family in "${bf16_family_list[@]}"; do
  case "$bf16_family" in
    mimo|mimop|mimof|glm|glmf|qwen4) ;;
    *) echo "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES: unknown family $bf16_family" >&2; exit 2 ;;
  esac
  [[ ";$expert_families;" == *";$bf16_family:fp8;"* ]] ||
    { echo "CUTEAFD_WIP_FP8_MOE_BF16_FAMILIES=$bf16_family needs $bf16_family:fp8 in CUTEAFD_WIP_EXPERT_FAMILIES" >&2; exit 2; }
done
# Official-only WIP builds may skip the EXL3 quantization AOT entirely. The
# default stays ON so every existing slot and script is byte-compatible; the
# native expert path does not require the EXL3 package.
exl3_aot="${CUTEAFD_WIP_EXL3_AOT:-ON}"
nvfp4_aot="${CUTEAFD_WIP_NVFP4_AOT:-ON}"
audio_aot="${CUTEAFD_WIP_AUDIO_AOT:-OFF}"
case "$audio_aot" in ON|OFF) ;; *) echo "CUTEAFD_WIP_AUDIO_AOT must be ON or OFF, got: $audio_aot" >&2; exit 2 ;; esac
case "$exl3_aot" in ON|OFF) ;; *) echo "CUTEAFD_WIP_EXL3_AOT must be ON or OFF, got: $exl3_aot" >&2; exit 2 ;; esac
case "$nvfp4_aot" in ON|OFF) ;; *) echo "CUTEAFD_WIP_NVFP4_AOT must be ON or OFF, got: $nvfp4_aot" >&2; exit 2 ;; esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a CUTEAFD source tree: $source_dir" >&2
  exit 2
}

python3 "$source_dir/scripts/build/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"
# GLM's include_bytes! inputs are compiled into the daemon in both roles.
transformers_source_digest="$(python3 "$source_dir/scripts/build/verify-transformers-source.py" \
  --source "$source_dir/third_party/transformers" \
  --lock "$source_dir/third_party/transformers.lock.json" --print-source-digest)"
if [[ "$xgrammar" == ON ]]; then
  python3 "$source_dir/scripts/build/verify-xgrammar-source.py" \
    --source "$source_dir/third_party/xgrammar" \
    --lock "$source_dir/third_party/xgrammar.lock.json"
fi

mkdir -p "$build_dir" "$output_dir"
if [[ "${CUTEAFD_BUILD_CACHES:-on}" == off ]]; then
  cold_cache="$(mktemp -d "$build_dir/cold-cache.XXXXXXXX")"
  export CARGO_HOME="$cold_cache/cargo"
  export CUTEAFD_KACHE= CUTEAFD_SCCACHE_CUDA=0
  export TORCH_EXTENSIONS_DIR="$cold_cache/torch-extensions" XDG_CACHE_HOME="$cold_cache/xdg"
  export B12X_ROCE_CACHE_DIR="$cold_cache/roce" TRITON_CACHE_DIR="$cold_cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$cold_cache/torchinductor"
fi
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$build_dir/cache/torchinductor}"
mkdir -p "${CARGO_HOME:-$HOME/.cargo}" "$TORCHINDUCTOR_CACHE_DIR"
source "$(dirname "${BASH_SOURCE[0]}")/build-caches.sh"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$build_dir/cache/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$build_dir/cache/xdg}"
export B12X_ROCE_CACHE_DIR="${B12X_ROCE_CACHE_DIR:-$build_dir/cache/roce}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$build_dir/cache/triton}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$B12X_ROCE_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYO3_PYTHON=python3
export PYTHONPATH="$source_dir/third_party/sparkinfer:$source_dir/python/reference/cuteafd_reference:$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}"
source "$(dirname "${BASH_SOURCE[0]}")/compiler-cache.sh"
cuteafd_compiler_cache_setup "$build_dir"
# kache restores hardlinks; plain and cached Cargo must not share outputs.
export CARGO_TARGET_DIR="$build_dir/cargo-target$( [[ "${CUTEAFD_KACHE_MODE:-disabled}" != enabled ]] || printf -- '-kache' )"
[[ "$phase" == native || "$phase" == export ]] || cuteafd_build_cache_cargo_offline "$source_dir/rust/Cargo.toml"
if [[ "${CUTEAFD_WIP_EXPORT_LOCKS:-off}" == on || "$phase" == native || "$phase" == export ]] && ! cuteafd_compiler_cache_check_cmake_compilers "$build_dir/native"; then
  # Copied configure metadata can name a different compiler shim. Objects and
  # AOT outputs remain reusable; only the stale configure identity is discarded.
  rm -f "$build_dir/native/CMakeCache.txt"
  rm -rf "$build_dir/native/CMakeFiles"
fi
cuteafd_compiler_cache_check_cmake_compilers "$build_dir/native"
compiler_cache_cmake_args=()
mapfile -t compiler_cache_cmake_args < <(cuteafd_compiler_cache_cmake_args "$build_dir/native")

# The WIP sync chain (rsync -a + docker cp) can leave source mtimes older
# than the previous build's fingerprints; cargo/ninja then silently reuse
# stale objects and the slot ships binaries that do not match the frozen
# source. Fingerprint each build tree's content and, only where it changed,
# refresh mtimes so the dependency trackers see the new content. Python
# tools drive the AOT exports, so a python change also refreshes native/.
wip_tree_fingerprint() {
  find "$@" -type f \( -name '*.rs' -o -name '*.toml' -o -name '*.lock' -o -name '*.cu' -o -name '*.cc' -o -name '*.h' -o -name '*.cmake' -o -name 'CMakeLists.txt' -o -name '*.py' \) \
    -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}
wip_rust_fingerprint="$(printf '%s %s\n' "$(wip_tree_fingerprint "$source_dir/rust")" "$transformers_source_digest" | sha256sum | awk '{print $1}')"
wip_native_fingerprint="$(wip_tree_fingerprint "$source_dir/native" "$source_dir/python")"
wip_fingerprint_marker="$build_dir/.source-content-fingerprint"
wip_previous_fingerprint="$(cat "$wip_fingerprint_marker" 2>/dev/null || true)"
wip_previous_rust="$(cut -d' ' -f1 <<<"$wip_previous_fingerprint")"
wip_previous_native="$(cut -d' ' -f2 <<<"$wip_previous_fingerprint")"
if [[ -z "$wip_previous_fingerprint" || "$wip_previous_rust" != "$wip_rust_fingerprint" ]]; then
  find "$source_dir/rust" "$source_dir/third_party/transformers/src" -type f -exec touch {} +
fi
if [[ -z "$wip_previous_fingerprint" || "$wip_previous_native" != "$wip_native_fingerprint" ]]; then
  find "$source_dir/native" "$source_dir/python" -type f -exec touch {} +
fi
wip_current_fingerprint="$wip_rust_fingerprint $wip_native_fingerprint"

if [[ -n "${CARGO_BUILD_JOBS:-}" ]]; then
  renice -n 19 -p "$$" >/dev/null
else
  unset CARGO_BUILD_JOBS
fi
[[ -n "${RUST_TEST_THREADS:-}" ]] || unset RUST_TEST_THREADS
[[ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]] || unset CMAKE_BUILD_PARALLEL_LEVEL
rust_phase_marker="$CARGO_TARGET_DIR/.wip-rust-phase-fingerprint"
wip_phase_log="$output_dir/BUILD_PHASES.tsv"
wip_phase_mark() {
  # PHASE SECONDS [DETAIL]: one row per completed phase, read by finalize-wip-slot.sh.
  printf '%s\t%s\t%s\t%s\n' "$role" "$1" "$2" "${3:-}" >>"$wip_phase_log"
  printf 'WIP phase %s %s: %ss%s\n' "$role" "$1" "$2" "${3:+ ($3)}"
}
# A fresh build starts the table; the native phase and a retried Rust phase append.
[[ "$phase" == native || "$phase" == export || "${CUTEAFD_WIP_PHASE_APPEND:-0}" == 1 ]] || : >"$wip_phase_log"

# Native outputs (library, AOT exports, packages, manifests) are a pure function
# of their inputs: native/, python/, the build helpers, the pinned submodule
# trees, every scope switch, role, architecture and the dev toolchain. A hit
# restores the previous outputs and skips CMake and every GPU export; Cargo
# still builds the daemon. The store is shared by instances on this host.
# wip.sh imports and publishes entries between this per-instance store
# (/wip/cache/native-store, on the host's NVMe build root) and the host-wide
# ~/.cache/cuteafd/native-store/<arch>, by hardlink, so instances share them.
native_store_root="${CUTEAFD_WIP_NATIVE_STORE:-$(dirname "$build_dir")/../cache/native-store}"
native_store_root="$(realpath -m "$native_store_root")"
native_items=(libcuteafd_native.so PROGRAMS.json V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json exl3 fp8)
wip_native_key() {
  {
    printf 'schema 1\nrole %s\narch %s %s\n' "$role" "$cuda_arch" "$(uname -m)"
    # The dev image (wip.sh passes its ID) fixes CUDA, CuTe-DSL, Triton and torch.
    printf 'image %s\n' "${CUTEAFD_WIP_DEV_IMAGE_ID:?the native cache needs the dev image ID}"
    # Every scope switch that reaches CMake or an exporter.
    env | grep -E '^CUTEAFD_WIP_[A-Z0-9_]+=' |
      grep -vE '^CUTEAFD_WIP_(EXPORT_LOCKS|EXPORT_LOCK_FILES|HOST_EXPORT_TURN|SEED_WAIT_S|EXPORT_WAIT_S|NATIVE_STORE|NATIVE_CACHE|NATIVE_REQUIRE_HIT|DEV_IMAGE_ID)=' |
      LC_ALL=C sort
    printf 'audio %s exl3 %s nvfp4 %s coordinator_aot %s xgrammar %s\n' "$audio_aot" "$exl3_aot" "$nvfp4_aot" "$coordinator_aot" "$xgrammar"
    # The CuTe DSL host target the exporters pin (#127; python/tools/lib/_pinned_sparkinfer.py):
    # an override, or "native", changes every exported object's host code.
    printf 'cute-host-target %s\n' "${CUTEAFD_CUTE_HOST_TARGET-<pinned default>}"
    # Pinned submodule trees by their verified lock digests (sparkinfer, xgrammar,
    # transformers), then every native, exporter and build-helper file.
    cat "$source_dir/third_party/"*.lock.json
    (cd "$source_dir" && find native python scripts/build -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 |
      LC_ALL=C sort -z | xargs -0 sha256sum)
  } | sha256sum | awk '{print $1}'
}
if [[ "$phase" == export ]]; then
  :
elif [[ "$phase" != native ]]; then
  rust_started=$SECONDS
  cargo build \
    --locked \
    --quiet \
    --manifest-path "$source_dir/rust/Cargo.toml" \
    -p cuteafd-daemon \
    --release
  printf '%s' "$wip_rust_fingerprint" >"$rust_phase_marker"
  wip_phase_mark rust $((SECONDS - rust_started)) "kache ${CUTEAFD_KACHE_MODE:-disabled}$([[ -d "${CUTEAFD_KACHE_WARNING_DIR:-/nonexistent}" ]] && echo ' fallback')"
else
  [[ "$(cat "$rust_phase_marker" 2>/dev/null || true)" == "$wip_rust_fingerprint" && -x "$CARGO_TARGET_DIR/release/cuteafd" ]] ||
    { echo 'native phase needs a successful rust phase for this source and cache mode' >&2; exit 2; }
fi

native_key=""
native_entry=""
native_hit=0
if [[ "${CUTEAFD_WIP_NATIVE_CACHE:-on}" == on && "${CUTEAFD_BUILD_CACHES:-on}" != off && -n "${CUTEAFD_WIP_DEV_IMAGE_ID:-}" ]]; then
  native_key_started=$SECONDS
  native_key="$(wip_native_key)"
  native_entry="$native_store_root/$role-$cuda_arch/$native_key"
  if [[ -f "$native_entry/COMPLETE" ]] && (cd "$native_entry/output" && sha256sum --quiet --strict -c ../NATIVE_SHA256SUMS); then
    native_hit=1
  fi
  [[ "$phase" == native || "$phase" == export ]] ||
    wip_phase_mark native-key $((SECONDS - native_key_started)) "$native_key $([[ $native_hit == 1 ]] && echo hit || echo miss)"
fi
# wip.sh reads this after the rust phase: a hit needs no GPU turn for the native phase.
# The key is written even on a miss, so wip.sh can import a host-wide entry.
rm -f "$output_dir/.native-cache-hit" "$output_dir/.native-cache-key"
[[ -z "$native_key" ]] || printf '%s %s\n' "$role-$cuda_arch" "$native_key" >"$output_dir/.native-cache-key"
((native_hit == 0)) || printf '%s\n' "$native_key" >"$output_dir/.native-cache-hit"
if [[ "$phase" == native && "${CUTEAFD_WIP_NATIVE_REQUIRE_HIT:-0}" == 1 ]] && ((native_hit == 0)); then
  echo 'native cache entry disappeared; the native phase needs its GPU turn' >&2
  exit 3
fi
[[ "$phase" != rust ]] || exit 0

native_started=$SECONDS
# The output holds exactly this build's native files: no package from an earlier
# build's scopes survives into the slot.
for item in "${native_items[@]}"; do rm -rf "${output_dir:?}/$item"; done
if ((native_hit)); then
  echo "WIP native outputs: reusing $native_entry (no CMake, no export)"
  # Only native files are restored: the daemon and its provenance come from Cargo.
  cp -a "$native_entry/output/." "$output_dir/"
  touch "$native_entry/COMPLETE"
  # The build tree is untouched: keep its native fingerprint, record the new Rust one.
  printf '%s %s' "$wip_rust_fingerprint" "$wip_previous_native" >"$wip_fingerprint_marker"
else
# Cargo is complete before GPU admission. Current wip.sh runs this native phase
# under a held fair turn on the host (CUTEAFD_WIP_HOST_EXPORT_TURN) and passes
# CUTEAFD_WIP_EXPORT_LOCKS=off; the in-container flock below only serves older
# containers that still mount a lock file. Ordinary WIP builds take no locks.
export_lock_fds=()
if [[ "${CUTEAFD_WIP_EXPORT_LOCKS:-off}" == on ]]; then
  for export_lock in ${CUTEAFD_WIP_EXPORT_LOCK_FILES:?missing export lock files}; do
    echo "WIP AOT/export waiting for $export_lock (timeout 1800s)"
    exec {export_lock_fd}>"$export_lock"
    flock -w 1800 "$export_lock_fd" || { echo "timed out waiting for $export_lock" >&2; exit 2; }
    export_lock_fds+=("$export_lock_fd")
  done
fi
cmake \
  "${compiler_cache_cmake_args[@]}" \
  -S "$source_dir/native" \
  -B "$build_dir/native" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTEAFD_ENABLE_CUDA=ON \
  -DCUTEAFD_ENABLE_VISION_ATTENTION_AOT="${CUTEAFD_WIP_VISION_ATTENTION_AOT:-ON}" \
  -DCUTEAFD_ENABLE_AUDIO_AOT="$audio_aot" \
  -DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON \
  -DCUTEAFD_SPARK_TP_ROLES="$spark_tp_roles" \
  -DCUTEAFD_EXPERT_FAMILIES="$expert_families" \
  -DCUTEAFD_GENERIC_SPARK_COUNTS="${CUTEAFD_WIP_GENERIC_SPARK_COUNTS:-}" \
  -DCUTEAFD_FP8_MOE_BF16_FAMILIES="$bf16_families" \
  -DCUTEAFD_ENABLE_V41_NVFP4_AOT="$nvfp4_aot" \
  -DCUTEAFD_ENABLE_EXL3_PACKAGES="$exl3_aot" \
  -DCUTEAFD_V41_EXL3_BITS="${CUTEAFD_WIP_EXL3_BITS:-2;3}" \
  -DCUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_FP8_AOT="$coordinator_aot" \
  -DCUTEAFD_DSV4_MAX_CONTEXT="${CUTEAFD_WIP_DSV4_MAX_CONTEXT:-1048576}" \
  -DCUTEAFD_ENABLE_DSV4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_DSV4_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_GLM_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_GLM_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_CONTEXT_SPLIT_AOT="$( [[ "$role" == coordinator && ( "${CUTEAFD_WIP_GLM_AOT:-OFF}" == ON || "${CUTEAFD_WIP_GLMF_AOT:-OFF}" == ON ) ]] && echo "${CUTEAFD_WIP_CONTEXT_SPLIT_AOT:-ON}" || echo OFF)" \
  -DCUTEAFD_ENABLE_MIMO_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_MIMO_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_MIMO_GEOMETRIES="$(g="${CUTEAFD_WIP_MIMO_GEOMETRIES:-mimo}"; echo "${g//,/;}")" \
  -DCUTEAFD_ENABLE_GLMF_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_GLMF_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_GLMF_WIDE_DECODE_ROWS="${CUTEAFD_WIP_GLMF_WIDE_DECODE_ROWS:-128}" \
  -DCUTEAFD_ENABLE_QWEN4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_QWEN4_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_V41_ATTENTION_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_HC_LAGGED_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_NARROW_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_RDMA=ON \
  -DCUTEAFD_SPARKINFER_SOURCE_DIR="$source_dir/third_party/sparkinfer" \
  -DCUTEAFD_SPARKINFER_LOCK_FILE="$source_dir/third_party/sparkinfer.lock.json" \
  -DCUTEAFD_ENABLE_NCCL=OFF \
  -DCUTEAFD_ENABLE_XGRAMMAR="$xgrammar" \
  -DCUTEAFD_XGRAMMAR_SOURCE_DIR="$source_dir/third_party/xgrammar" \
  -DCUTEAFD_XGRAMMAR_LOCK_FILE="$source_dir/third_party/xgrammar.lock.json" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCUTEAFD_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_dir/native"
for export_lock_fd in "${export_lock_fds[@]}"; do
  flock -u "$export_lock_fd"
  exec {export_lock_fd}>&-
done
printf '%s' "$wip_current_fingerprint" >"$wip_fingerprint_marker"

install -m 0755 "$build_dir/native/libcuteafd_native.so" "$output_dir/libcuteafd_native.so"
# The coordinator program manifest (DeepSeek V4, GLM, GLM Flash, MiMo, Qwen; an empty table
# when none was built), as the release images carry it at /opt/cuteafd/share/PROGRAMS.json:
# run-family.sh --wip serves those families from it.
if [[ -s "$build_dir/native/dsv4_programs/dsv4_programs.json" ]]; then
  install -m 0644 "$build_dir/native/dsv4_programs/dsv4_programs.json" "$output_dir/PROGRAMS.json"
else
  printf '%s\n' '{"schema":1,"programs":[]}' >"$output_dir/PROGRAMS.json"
fi
# The EXL3 package is only built and installed when the opt-in is ON. An
# official-only WIP build (CUTEAFD_WIP_EXL3_AOT=OFF) has no exl3/ directory and
# the native launch path never references one.
if [[ "$exl3_aot" == ON ]]; then
  wip_exl3_bits="${CUTEAFD_WIP_EXL3_BITS:-2;3}"
  wip_exl3_tag="k${wip_exl3_bits//[;]/}"
  wip_exl3_tag="${wip_exl3_tag//,/}"
  python3 "$source_dir/python/tools/aot/package_exl3_aot.py" install \
    --package "$build_dir/native/exl3-$wip_exl3_tag" --output "$output_dir/exl3/exl3-$wip_exl3_tag"
  python3 "$source_dir/python/tools/aot/package_exl3_aot.py" verify \
    --package "$output_dir/exl3/exl3-$wip_exl3_tag" --role "$role"
  # Other expert geometries (FAMILY:exl3-kTIERS entries) ship as exl3-FAMILY-kTIERS.
  IFS=';' read -ra wip_family_list <<<"$expert_families"
  for wip_family in "${wip_family_list[@]}"; do
    [[ "$wip_family" == *:exl3-k* ]] || continue
    wip_package="exl3-${wip_family%%:*}-${wip_family#*:exl3-}"
    python3 "$source_dir/python/tools/aot/package_exl3_aot.py" install \
      --package "$build_dir/native/$wip_package" --output "$output_dir/exl3/$wip_package"
    python3 "$source_dir/python/tools/aot/package_exl3_aot.py" verify \
      --package "$output_dir/exl3/$wip_package" --role "$role"
  done
fi
# Exact FP8 expert packages (FAMILY:fp8 entries) ship as fp8/fp8-FAMILY.
IFS=';' read -ra wip_fp8_list <<<"$expert_families"
fp8_revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$source_dir/third_party/sparkinfer.lock.json")"
for wip_family in "${wip_fp8_list[@]}"; do
  case "$wip_family" in
    *:fp8) wip_package="fp8-${wip_family%%:*}" ;;
    # ModelOpt NVFP4 (W4A16 / W4A4 large-row steps) share the fp8_moe programs.
    *:nvfp4|*:nvfp4a4) wip_package="fp8-${wip_family%%:*}-${wip_family#*:}" ;;
    *) continue ;;
  esac
  mkdir -p "$output_dir/fp8"
  # A FAMILY:nvfp4 entry also builds the default W4A4 sibling (fp8_moe.cmake).
  wip_packages=("$wip_package")
  # Stage only requested siblings; discard an older slot's opt-in after opt-out.
  if [[ "$role" == expert && "$wip_family" == *:fp8 ]]; then
    rm -rf "$output_dir/fp8/${wip_package}-bf16"
    [[ ";$bf16_families;" != *";${wip_family%%:*};"* ]] ||
      wip_packages+=("${wip_package}-bf16")
  fi
  [[ "$wip_family" == *:nvfp4 ]] && wip_packages+=("${wip_package}a4")
  # Local Qwen NVFP4 keeps its MTP experts in FP8: CMake builds the TP1 FP8 sibling on the
  # coordinator (fp8_moe.cmake); stage it as release images do.
  [[ "$role" == coordinator && "$wip_family" == qwen4:nvfp4* ]] && wip_packages+=("fp8-qwen4")
  for wip_package in "${wip_packages[@]}"; do
    rm -rf "$output_dir/fp8/$wip_package"
    cp -a "$build_dir/native/fp8/$wip_package" "$output_dir/fp8/$wip_package"
    fp8_input_args=()
    [[ "$wip_package" != *-bf16 ]] || fp8_input_args=(--input bf16)
    python3 "$source_dir/python/tools/aot/package_fp8_moe_aot.py" verify \
      --package "$output_dir/fp8/$wip_package" --sparkinfer-revision "$fp8_revision" --role "$role" "${fp8_input_args[@]}"
  done
done
install -m 0644 "$build_dir/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"
# Always emit the built-role manifest (empty for the legacy default). Roles are
# derived from the AOT export manifests CMake actually produced and bound to the
# built library hash, so a stale/partial export cannot advertise a role.
nvfp4_manifest_args=()
[[ "$nvfp4_aot" != ON ]] || nvfp4_manifest_args+=(--nvfp4)
python3 "$source_dir/scripts/build/write-v41-expert-tp-manifest.py" \
  "${nvfp4_manifest_args[@]}" \
  --role "$role" \
  --requested "$spark_tp_roles" \
  --native-build-dir "$build_dir/native" \
  --native-library "$build_dir/native/libcuteafd_native.so" \
  --output "$output_dir/V41_EXPERT_TP_AOT.json"
if [[ "$coordinator_aot" == ON ]]; then
  install -m 0644 "$build_dir/native/v41_fp8/v41_fp8.json" "$output_dir/V41_FP8_AOT.json"
else
  printf '%s\n' '{"schema":1,"role":"expert","enabled":false}' >"$output_dir/V41_FP8_AOT.json"
fi
fi
if [[ "${phase:-all}" != export ]]; then
install -m 0755 "$CARGO_TARGET_DIR/release/cuteafd" "$output_dir/cuteafd"
fi
if [[ "${phase:-all}" == export ]]; then
  :
elif [[ -n "${CUTEAFD_KACHE:-}${CUTEAFD_KACHE_REQUESTED:-}" ]]; then
  python3 "$(dirname "${BASH_SOURCE[0]}")/write-compiler-provenance.py" \
    "$source_dir" "$output_dir/cuteafd" "$output_dir/COMPILER_PROVENANCE.json"
elif [[ -f "$output_dir/COMPILER_PROVENANCE.json" ]]; then
  rm -f "$output_dir/COMPILER_PROVENANCE.json"
fi
# Generic-family tables must come from this build, not an earlier role/opt-in.
# A native-cache hit restored the table with the rest of the native outputs.
if ((${native_hit:-0})); then
  :
elif [[ "$role" == coordinator && "${CUTEAFD_WIP_DSV4_AOT:-OFF};${CUTEAFD_WIP_GLM_AOT:-OFF};${CUTEAFD_WIP_MIMO_AOT:-OFF};${CUTEAFD_WIP_GLMF_AOT:-OFF};${CUTEAFD_WIP_QWEN4_AOT:-OFF}" == *ON* ]]; then
  install -m 0644 "$build_dir/native/dsv4_programs/dsv4_programs.json" "$output_dir/PROGRAMS.json"
else
  printf '%s\n' '{"schema":1,"programs":[]}' >"$output_dir/PROGRAMS.json"
fi
# Publish this build's native outputs for the next build with the same key.
if [[ -n "${native_key:-}" ]] && ((${native_hit:-0} == 0)); then
  native_stage="$native_store_root/$role-$cuda_arch/.stage-$native_key-$$"
  rm -rf "$native_stage"
  mkdir -p "$native_stage/output"
  for item in "${native_items[@]}"; do
    [[ ! -e "$output_dir/$item" ]] || cp -a "$output_dir/$item" "$native_stage/output/"
  done
  (cd "$native_stage/output" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >../NATIVE_SHA256SUMS)
  printf '%s\n' "$native_key" >"$native_stage/COMPLETE"
  if [[ -e "$native_entry" ]]; then
    rm -rf "$native_stage"  # another build published the same key first
  else
    mv -T "$native_stage" "$native_entry" 2>/dev/null || rm -rf "$native_stage"
  fi
  # Keep the most recently used entries per role and architecture.
  find "$native_store_root/$role-$cuda_arch" -mindepth 2 -maxdepth 2 -name COMPLETE -printf '%T@ %h\n' |
    sort -rn | tail -n +"$((${CUTEAFD_WIP_NATIVE_KEEP:-16} + 1))" | cut -d' ' -f2- |
    while IFS= read -r stale; do rm -rf "$stale"; done
fi
[[ -z "${native_started:-}" ]] ||
  wip_phase_mark native $((SECONDS - native_started)) "$([[ ${native_hit:-0} == 1 ]] && echo reused || echo built)"

[[ "${phase:-all}" != export ]] || exit 0
(
  cd "$output_dir"
  sha256sum cuteafd libcuteafd_native.so V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json PROGRAMS.json >ARTIFACT_SHA256SUMS
  sha256sum -c ARTIFACT_SHA256SUMS
)
