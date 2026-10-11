#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-wip-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH BUILD_DIR OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 5 ]] || usage
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
cuteafd_build_cache_cargo_offline "$source_dir/rust/Cargo.toml"
if [[ "${CUTEAFD_WIP_EXPORT_LOCKS:-off}" == on ]] && ! cuteafd_compiler_cache_check_cmake_compilers "$build_dir/native"; then
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
cargo build \
  --locked \
  --quiet \
  --manifest-path "$source_dir/rust/Cargo.toml" \
  -p cuteafd-daemon \
  --release

# Cargo is complete before GPU admission. Locks are host files individually
# bind-mounted by wip.sh; ordinary WIP builds do not acquire hardware locks.
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

install -m 0755 "$CARGO_TARGET_DIR/release/cuteafd" "$output_dir/cuteafd"
if [[ -n "${CUTEAFD_KACHE:-}${CUTEAFD_KACHE_REQUESTED:-}" ]]; then
  python3 "$(dirname "${BASH_SOURCE[0]}")/write-compiler-provenance.py" \
    "$source_dir" "$output_dir/cuteafd" "$output_dir/COMPILER_PROVENANCE.json"
elif [[ -f "$output_dir/COMPILER_PROVENANCE.json" ]]; then
  rm -f "$output_dir/COMPILER_PROVENANCE.json"
fi
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
python3 "$source_dir/scripts/build/write-v41-expert-tp-manifest.py" \
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
# Generic-family tables must come from this build, not an earlier role/opt-in.
if [[ "$role" == coordinator && "${CUTEAFD_WIP_DSV4_AOT:-OFF};${CUTEAFD_WIP_GLM_AOT:-OFF};${CUTEAFD_WIP_MIMO_AOT:-OFF};${CUTEAFD_WIP_GLMF_AOT:-OFF};${CUTEAFD_WIP_QWEN4_AOT:-OFF}" == *ON* ]]; then
  install -m 0644 "$build_dir/native/dsv4_programs/dsv4_programs.json" "$output_dir/PROGRAMS.json"
else
  printf '%s\n' '{"schema":1,"programs":[]}' >"$output_dir/PROGRAMS.json"
fi
(
  cd "$output_dir"
  sha256sum cuteafd libcuteafd_native.so V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json PROGRAMS.json >ARTIFACT_SHA256SUMS
  sha256sum -c ARTIFACT_SHA256SUMS
)
