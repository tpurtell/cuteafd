#!/usr/bin/env bash
# Run `cuteafd programs-check` on a freshly built artifact set.
#
# Usage: programs-check.sh CUTEAFD_BINARY NATIVE_LIB MANIFEST ROLE FAMILIES
#
# The check only reads the built program table and its manifest; it never touches a
# GPU. But loading libcuteafd_native.so pulls in the CUDA driver library
# (libcuda.so.1), and AOT exports run in CPU-only containers that have no driver.
# When no real libcuda.so.1 is installed, resolve it to the CUDA toolkit's stub
# (SONAME libcuda.so.1) for this one command; the stub satisfies the loader and
# would fail any real driver call, which programs-check does not make.
set -euo pipefail
binary="$1" native_lib="$2" manifest="$3" role="$4" families="$5"
case "$role" in coordinator|spark) ;; *) echo "programs-check.sh: role must be coordinator or spark, got: $role" >&2; exit 2 ;; esac

cuda_driver_present() {
  ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1 '
}

stub_dir=
if ! cuda_driver_present; then
  for candidate in /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so; do
    if [[ -f "$candidate" ]]; then
      stub_dir="$(mktemp -d)"
      trap 'rm -rf -- "$stub_dir"' EXIT
      ln -s "$candidate" "$stub_dir/libcuda.so.1"
      break
    fi
  done
  [[ -n "$stub_dir" ]] || { echo "programs-check.sh: no CUDA driver and no toolkit libcuda stub found" >&2; exit 2; }
fi

if [[ -n "$stub_dir" ]]; then
  export LD_LIBRARY_PATH="$stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$binary" programs-check --native-lib "$native_lib" --manifest "$manifest" \
  --exlr8-role "$role" --exlr8-family "$families"
