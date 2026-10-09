#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-release-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
output_dir="$(realpath -m "$4")"
# Where the writable build root is created. Unset keeps the historical in-container
# /tmp scratch; build.sh relocates it to a unique per-task path on the host, mounted
# at the identical path in the container, so a release build never depends on how
# large a container's default /tmp happens to be. mktemp is given an absolute
# template, so honoring this variable is what makes the relocation real: a mere
# TMPDIR export would be ignored by the template.
build_root_parent="${CUTEAFD_RELEASE_BUILD_ROOT:-/tmp}"
# The build containers run as the invoking user rather than root (build.sh passes
# --user): that UID has no passwd entry and no home the image created, so the
# writable home and cache roots the caller named are created before Cargo, CMake
# or TorchInductor touches them (Cargo does create its own home leaf, but not the
# human home above it).
for writable_root in "${HOME:-}" "${CARGO_HOME:-}" "${TORCHINDUCTOR_CACHE_DIR:-}"; do
  [[ -n "$writable_root" ]] || continue
  mkdir -p "$writable_root"
done
# Reject unsafe output/cache filesystems before staging or invoking Cargo.
# SOURCE_DIR is a read-only input: the release container mounts it `/source:ro`
# and this script stages a writable copy into the build root below, so probing
# the source here would only trip the guard's fail-closed read-only rule. The
# cargo target is guarded because it is where the daemon is written, and the build
# root parent because the default target lives beneath it. Each filesystem is
# probed once: with no target override the parent already covers it, and the parent
# is added separately only when a relocated target points elsewhere. With the hook
# unset this is the historical output, /tmp, cargo-home probe.
build_root_probe=()
if [[ -n "${CARGO_TARGET_DIR:-}" && "$CARGO_TARGET_DIR" != "$build_root_parent" ]]; then
  build_root_probe=("$build_root_parent")
fi
python3 "$(dirname "$0")/assert-build-filesystem.py" \
  "$output_dir" "${CARGO_TARGET_DIR:-$build_root_parent}" "${CARGO_HOME:-$HOME/.cargo}" \
  ${build_root_probe[@]+"${build_root_probe[@]}"}

# Report the real cause here instead of an opaque mktemp failure: a requested build
# root the container cannot write means the bind mount is missing, or it exists on
# the host as a user the container does not run as.
if ! test -w "$build_root_parent"; then
  echo "release build root is not writable inside this container: $build_root_parent" >&2
  echo "it must exist on the host and be bind-mounted at the identical path (see build.sh --help)" >&2
  exit 2
fi

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
exl3_paired_tp4=OFF
exl3_residency=""
# Opt-in replicated-group Spark expert roles. Empty is the default and keeps
# the historical Spark TP4 shard (and every release default) byte-identical.
# Release images carry the audio tower: AUDIO=auto serves it for qualified checkpoints.
audio_aot="${CUTEAFD_RELEASE_AUDIO_AOT:-ON}"
case "$audio_aot" in ON|OFF) ;; *) echo "CUTEAFD_RELEASE_AUDIO_AOT must be ON or OFF, got: $audio_aot" >&2; exit 2 ;; esac
spark_tp_roles="${CUTEAFD_RELEASE_SPARK_TP_ROLES:-}"
spark_tp_role_list=()
if [[ -n "$spark_tp_roles" ]]; then
  IFS=';' read -ra spark_tp_role_list <<<"$spark_tp_roles"
  for spark_tp_role in "${spark_tp_role_list[@]}"; do
    case "$spark_tp_role" in
      tp2|tp3|tp6) ;;
      *) echo "CUTEAFD_RELEASE_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got: $spark_tp_role" >&2; exit 2 ;;
    esac
  done
  [[ "$role" == expert ]] ||
    { echo "CUTEAFD_RELEASE_SPARK_TP_ROLES is only valid for the expert role" >&2; exit 2; }
fi

# Extra routed-expert kernel families (FAMILY:ROLE list, e.g. dsv4f:spark),
# validated by native/cmake/shared/expert_families.cmake. Empty keeps the V4.1 image.
# Both builds take the same list; CMake keeps the entries for its architecture.
expert_families="${CUTEAFD_RELEASE_EXPERT_FAMILIES:-}"
# Optional Spark siblings retain unquantized BF16 expert inputs. Empty keeps
# the existing artifact set; requesting one also requires its main FP8 family.
bf16_families="${CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES:-}"
IFS=';' read -ra bf16_family_list <<<"$bf16_families"
for bf16_family in "${bf16_family_list[@]}"; do
  case "$bf16_family" in
    mimo|mimop|mimof|glm|glmf|qwen4) ;;
    *) echo "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES: unknown family $bf16_family" >&2; exit 2 ;;
  esac
  [[ ";$expert_families;" == *";$bf16_family:fp8;"* ]] ||
    { echo "CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$bf16_family needs $bf16_family:fp8 in CUTEAFD_RELEASE_EXPERT_FAMILIES" >&2; exit 2; }
done
# v7 ships both EXL3 decoder families by default: the uniform K=2 raw
# publication family (2,3) and the staged K3.25 family (3,4). Paired TP4
# builds remain single-family and stay on the v5 (3,4) family.
exl3_bit_families="${CUTEAFD_RELEASE_EXL3_BIT_FAMILIES:-2,3;3,4}"
case "${CUTEAFD_RELEASE_EXL3_PAIRED_TP4:-off}" in
  on)
    [[ "$role" == expert ]] || { echo "Paired EXL3 package requires expert role" >&2; exit 2; }
    exl3_paired_tp4=ON
    exl3_residency=80=2
    exl3_bit_families="${CUTEAFD_RELEASE_EXL3_BIT_FAMILIES:-3,4}"
    ;;
  off) ;;
  *) echo "CUTEAFD_RELEASE_EXL3_PAIRED_TP4 must be on or off" >&2; exit 2 ;;
esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a CUTEAFD source tree: $source_dir" >&2
  exit 2
}
[[ -f "$source_dir/THIRD_PARTY_NOTICES.md" ]] || {
  echo "SOURCE_DIR is missing THIRD_PARTY_NOTICES.md" >&2
  exit 2
}
python3 "$source_dir/scripts/build/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"
if [[ "$xgrammar" == ON ]]; then
  python3 "$source_dir/scripts/build/verify-xgrammar-source.py" \
    --source "$source_dir/third_party/xgrammar" \
    --lock "$source_dir/third_party/xgrammar.lock.json"
fi

build_root="$(mktemp -d "$build_root_parent/cuteafd-release-build.XXXXXX")"
trap 'rm -rf "$build_root"' EXIT
mkdir -p "$build_root/source"
# The cargo target and the install source must agree. Default to the writable
# staged copy; an explicit external override is honored for both so a caller
# that relocates the target cannot leave the install pointing at the old path.
cargo_target_dir="${CARGO_TARGET_DIR:-$build_root/source/rust/target}"
tar \
  -C "$source_dir" \
  --exclude=.git \
  --exclude='*/.git' \
  --exclude='.venv*' \
  --exclude='*/.venv*' \
  --exclude=.mypy_cache \
  --exclude='*/.mypy_cache' \
  --exclude=.pytest_cache \
  --exclude='*/.pytest_cache' \
  --exclude=.ruff_cache \
  --exclude='*/.ruff_cache' \
  --exclude=__pycache__ \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude=.cuteafd-cache \
  --exclude=.cuteafd-release \
  --exclude=.cuteafd-release-image \
  --exclude=.cuteafd-wip \
  --exclude=dist \
  --exclude=rust/target \
  --exclude='native/build*' \
  -cf - . |
  tar -C "$build_root/source" -xf -

python3 "$build_root/source/scripts/build/verify-sparkinfer-source.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --require-no-python-cache
if [[ "$xgrammar" == ON ]]; then
  python3 "$build_root/source/scripts/build/verify-xgrammar-source.py" \
    --source "$build_root/source/third_party/xgrammar" \
    --lock "$build_root/source/third_party/xgrammar.lock.json"
fi

if [[ "${CUTEAFD_BUILD_CACHES:-on}" == off ]]; then
  cold_cache="$(mktemp -d "$build_root/cold-cache.XXXXXXXX")"
  export CARGO_HOME="$cold_cache/cargo"
  export CUTEAFD_KACHE= CUTEAFD_SCCACHE_CUDA=0
  export TORCH_EXTENSIONS_DIR="$cold_cache/torch-extensions" XDG_CACHE_HOME="$cold_cache/xdg"
  export B12X_ROCE_CACHE_DIR="$cold_cache/roce" TRITON_CACHE_DIR="$cold_cache/triton"
  export TORCHINDUCTOR_CACHE_DIR="$cold_cache/torchinductor"
fi
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$build_root/cache/torchinductor}"
mkdir -p "${CARGO_HOME:-$HOME/.cargo}" "$TORCHINDUCTOR_CACHE_DIR"
source "$(dirname "${BASH_SOURCE[0]}")/build-caches.sh"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$build_root/cache/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$build_root/cache/xdg}"
export B12X_ROCE_CACHE_DIR="${B12X_ROCE_CACHE_DIR:-$build_root/cache/roce}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$build_root/cache/triton}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$B12X_ROCE_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONPATH="$build_root/source/third_party/sparkinfer:$build_root/source/python/reference/cuteafd_reference:$build_root/source/python/reference${PYTHONPATH:+:$PYTHONPATH}"
export PYO3_PYTHON=python3
compiler_cache_cmake_args=()
source "$(dirname "${BASH_SOURCE[0]}")/compiler-cache.sh"
cuteafd_compiler_cache_setup "$build_root"
cuteafd_compiler_cache_check_cmake_compilers "$build_root/native"
mapfile -t compiler_cache_cmake_args < <(cuteafd_compiler_cache_cmake_args "$build_root/native")
cargo_target_dir+="$( [[ "${CUTEAFD_KACHE_MODE:-disabled}" != enabled ]] || printf -- '-kache' )"
export CARGO_TARGET_DIR="$cargo_target_dir"
cuteafd_build_cache_cargo_offline "$build_root/source/rust/Cargo.toml"
cargo build \
  --locked \
  --manifest-path "$build_root/source/rust/Cargo.toml" \
  -p cuteafd-daemon \
  --release

# Release images serve the native V4.1 path. The DS4 Flash/Pro AOT bridge is
# retained for development commands, but must not add legacy generated kernels
# or ABI coupling to the release artifact.
cmake \
  "${compiler_cache_cmake_args[@]}" \
  -S "$build_root/source/native" \
  -B "$build_root/native" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTEAFD_ENABLE_CUDA=ON \
  -DCUTEAFD_ENABLE_VISION_ATTENTION_AOT="${CUTEAFD_RELEASE_VISION_ATTENTION_AOT:-ON}" \
  -DCUTEAFD_ENABLE_AUDIO_AOT="$audio_aot" \
  -DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON \
  -DCUTEAFD_SPARK_TP_ROLES="$spark_tp_roles" \
  -DCUTEAFD_EXPERT_FAMILIES="$expert_families" \
  -DCUTEAFD_FP8_MOE_BF16_FAMILIES="$bf16_families" \
  -DCUTEAFD_ENABLE_V41_NVFP4_AOT="${CUTEAFD_RELEASE_NVFP4_AOT:-ON}" \
  -DCUTEAFD_ENABLE_EXL3_PACKAGES=ON \
  -DCUTEAFD_V41_EXL3_BIT_FAMILIES="$exl3_bit_families" \
  -DCUTEAFD_V41_EXL3_PAIRED_TP4="$exl3_paired_tp4" \
  -DCUTEAFD_V41_EXL3_RESIDENCY="$exl3_residency" \
  -DCUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_FP8_AOT="$coordinator_aot" \
  -DCUTEAFD_DSV4_MAX_CONTEXT="${CUTEAFD_RELEASE_DSV4_MAX_CONTEXT:-1048576}" \
  -DCUTEAFD_ENABLE_DSV4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_RELEASE_DSV4_AOT:-ON}" || echo OFF)" \
  -DCUTEAFD_ENABLE_GLM_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_RELEASE_GLM_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_MIMO_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_RELEASE_MIMO_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_MIMO_GEOMETRIES="$(g="${CUTEAFD_RELEASE_MIMO_GEOMETRIES:-mimo,mimo2,mimop,mimop2}"; echo "${g//,/;}")" \
  -DCUTEAFD_ENABLE_GLMF_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_RELEASE_GLMF_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_QWEN4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_RELEASE_QWEN4_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_V41_ATTENTION_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_HC_LAGGED_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_NARROW_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_RDMA=ON \
  -DCUTEAFD_SPARKINFER_SOURCE_DIR="$build_root/source/third_party/sparkinfer" \
  -DCUTEAFD_SPARKINFER_LOCK_FILE="$build_root/source/third_party/sparkinfer.lock.json" \
  -DCUTEAFD_ENABLE_NCCL=OFF \
  -DCUTEAFD_ENABLE_XGRAMMAR="$xgrammar" \
  -DCUTEAFD_XGRAMMAR_SOURCE_DIR="$build_root/source/third_party/xgrammar" \
  -DCUTEAFD_XGRAMMAR_LOCK_FILE="$build_root/source/third_party/xgrammar.lock.json" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCUTEAFD_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_root/native"

install -d "$output_dir"
install -m 0755 "$cargo_target_dir/release/cuteafd" "$output_dir/cuteafd"
if [[ -n "${CUTEAFD_KACHE:-}${CUTEAFD_KACHE_REQUESTED:-}" ]]; then
  python3 "$(dirname "${BASH_SOURCE[0]}")/write-compiler-provenance.py" \
    "$source_dir" "$output_dir/cuteafd" "$output_dir/COMPILER_PROVENANCE.json"
fi
install -m 0755 "$build_root/native/libcuteafd_native.so" "$output_dir/libcuteafd_native.so"
exl3_family_tags=()
IFS=';' read -ra exl3_family_list <<<"$exl3_bit_families"
for exl3_family in "${exl3_family_list[@]}"; do
  exl3_family_tags+=("k${exl3_family//,/}")
done
# Families nest under exl3/ so images keep one well-known EXL3 root; the
# daemon resolves exl3/exl3-kXX by checkpoint tiers and treats a direct
# layout child of exl3/ as the legacy single-family package.
for exl3_tag in "${exl3_family_tags[@]}"; do
  python3 "$build_root/source/python/tools/aot/package_exl3_aot.py" install \
    --package "$build_root/native/exl3-$exl3_tag" --output "$output_dir/exl3/exl3-$exl3_tag"
  python3 "$build_root/source/python/tools/aot/package_exl3_aot.py" verify \
    --package "$output_dir/exl3/exl3-$exl3_tag" --role "$role"
done
# Other expert geometries (FAMILY:exl3-kTIERS entries) ship as exl3-FAMILY-kTIERS.
IFS=';' read -ra release_family_list <<<"$expert_families"
for release_family in "${release_family_list[@]}"; do
  [[ "$release_family" == *:exl3-k* ]] || continue
  release_package="exl3-${release_family%%:*}-${release_family#*:exl3-}"
  python3 "$build_root/source/python/tools/aot/package_exl3_aot.py" install \
    --package "$build_root/native/$release_package" --output "$output_dir/exl3/$release_package"
  python3 "$build_root/source/python/tools/aot/package_exl3_aot.py" verify \
    --package "$output_dir/exl3/$release_package" --role "$role"
done
# Exact FP8 expert packages (FAMILY:fp8 entries) ship as fp8/fp8-FAMILY; the
# directory always exists so the release image can COPY it.
# CMake automatically builds the TP1 FP8 MTP sibling for local Qwen NVFP4.
# Stage it on the coordinator without requesting an unsupported Spark layout.
if [[ "$role" == coordinator ]]; then
  for release_family in "${release_family_list[@]}"; do
    case "$release_family" in
      qwen4:nvfp4|qwen4:nvfp4a4)
        release_family_list+=(qwen4:fp8)
        break
        ;;
    esac
  done
fi
mkdir -p "$output_dir/fp8"
fp8_revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$build_root/source/third_party/sparkinfer.lock.json")"
for release_family in "${release_family_list[@]}"; do
  case "$release_family" in
    *:fp8) release_package="fp8-${release_family%%:*}" ;;
    # ModelOpt NVFP4 (W4A16 / W4A4 large-row steps) share the fp8_moe programs.
    *:nvfp4|*:nvfp4a4) release_package="fp8-${release_family%%:*}-${release_family#*:}" ;;
    *) continue ;;
  esac
  # GLM 5.3 Flash's NVFP4 dense MLP runs on the coordinator only (fp8_moe.cmake
  # skips it in the Spark build).
  [[ "$role" == expert && "$release_family" == glmfdense:* ]] && continue
  mkdir -p "$output_dir/fp8"
  # A FAMILY:nvfp4 entry also builds the default W4A4 sibling (fp8_moe.cmake).
  release_packages=("$release_package")
  # Stage only requested siblings; discard an older slot's opt-in after opt-out.
  if [[ "$role" == expert && "$release_family" == *:fp8 ]]; then
    rm -rf "$output_dir/fp8/${release_package}-bf16"
    [[ ";$bf16_families;" != *";${release_family%%:*};"* ]] ||
      release_packages+=("${release_package}-bf16")
  fi
  [[ "$release_family" == *:nvfp4 ]] && release_packages+=("${release_package}a4")
  for release_package in "${release_packages[@]}"; do
    rm -rf "$output_dir/fp8/$release_package"
    cp -a "$build_root/native/fp8/$release_package" "$output_dir/fp8/$release_package"
    fp8_input_args=()
    [[ "$release_package" != *-bf16 ]] || fp8_input_args=(--input bf16)
    python3 "$build_root/source/python/tools/aot/package_fp8_moe_aot.py" verify \
      --package "$output_dir/fp8/$release_package" --sparkinfer-revision "$fp8_revision" --role "$role" "${fp8_input_args[@]}"
  done
done
install -m 0644 "$build_root/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"
# Always write the built-role manifest, including the empty-role default, so
# the release Dockerfile can COPY it unconditionally. Every listed role is
# derived from the AOT export manifest CMake actually produced.
python3 "$build_root/source/scripts/build/write-v41-expert-tp-manifest.py" \
  --role "$role" \
  --requested "$spark_tp_roles" \
  --native-build-dir "$build_root/native" \
  --native-library "$build_root/native/libcuteafd_native.so" \
  --output "$output_dir/V41_EXPERT_TP_AOT.json"
if [[ "$coordinator_aot" == ON ]]; then
  # Automatic RTX placement requires the full local-expert ABI in release images.
  python3 - "$output_dir/libcuteafd_native.so" <<'PY_CHECK'
import ctypes
import sys
library = ctypes.CDLL(sys.argv[1])
getattr(library, "cuteafd_v41_local_expert_info")
getattr(library, "cuteafd_v41_tp2_expert_info")
PY_CHECK
  install -m 0644 "$build_root/native/v41_fp8/v41_fp8.json" "$output_dir/V41_FP8_AOT.json"
else
  printf '%s\n' '{"schema":1,"role":"expert","enabled":false}' >"$output_dir/V41_FP8_AOT.json"
fi
# DeepSeek V4 coordinator program manifest (an empty table when not built).
if [[ -s "$build_root/native/dsv4_programs/dsv4_programs.json" ]]; then
  install -m 0644 "$build_root/native/dsv4_programs/dsv4_programs.json" "$output_dir/PROGRAMS.json"
else
  printf '%s\n' '{"schema":1,"programs":[]}' >"$output_dir/PROGRAMS.json"
fi
install -m 0644 \
  "$build_root/source/THIRD_PARTY_NOTICES.md" \
  "$output_dir/THIRD_PARTY_NOTICES.md"
install -m 0644 \
  "$build_root/source/third_party/sparkinfer/LICENSE" \
  "$output_dir/SPARKINFER_LICENSE"
install -m 0644 \
  "$build_root/source/third_party/xgrammar/LICENSE" \
  "$output_dir/XGRAMMAR_LICENSE"
install -m 0644 \
  "$build_root/source/third_party/xgrammar.lock.json" \
  "$output_dir/XGRAMMAR_PROVENANCE.json"
python3 "$build_root/source/scripts/build/sparkinfer-release-provenance.py" \
  --source "$build_root/source/third_party/sparkinfer" \
  --lock "$build_root/source/third_party/sparkinfer.lock.json" \
  --license "$output_dir/SPARKINFER_LICENSE" \
  --notices "$output_dir/THIRD_PARTY_NOTICES.md" \
  --write "$output_dir/SPARKINFER_PROVENANCE.json"
(
  cd "$output_dir"
  sha256sum \
    THIRD_PARTY_NOTICES.md \
    SPARKINFER_PROVENANCE.json \
    SPARKINFER_LICENSE >SPARKINFER_SHA256SUMS
  sha256sum -c SPARKINFER_SHA256SUMS
  sha256sum \
    THIRD_PARTY_NOTICES.md \
    XGRAMMAR_PROVENANCE.json \
    XGRAMMAR_LICENSE >XGRAMMAR_SHA256SUMS
  sha256sum -c XGRAMMAR_SHA256SUMS
)
test -x "$output_dir/cuteafd"
test -s "$output_dir/libcuteafd_native.so"
test -s "$output_dir/V41_EXPERT_TP_AOT.json"
test -s "$output_dir/THIRD_PARTY_NOTICES.md"
test -s "$output_dir/SPARKINFER_PROVENANCE.json"
test -s "$output_dir/SPARKINFER_LICENSE"
test -s "$output_dir/SPARKINFER_SHA256SUMS"
test -s "$output_dir/XGRAMMAR_PROVENANCE.json"
test -s "$output_dir/XGRAMMAR_LICENSE"
test -s "$output_dir/XGRAMMAR_SHA256SUMS"
