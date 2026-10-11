#!/usr/bin/env bash
# Source and call cuteafd_compiler_cache_setup BUILD_DIR, or use as a compiler wrapper.

cuteafd_compiler_cache_warn() {
  if [[ "${cuteafd_cache_warned:-0}" == 0 ]]; then
    printf 'warning: cuteafd compiler cache disabled: %s; using plain compilers\n' "$*" >&2
    cuteafd_cache_warned=1
  fi
}

cuteafd_compiler_cache_setup() {
  cuteafd_sccache_cuda_setup "$1"
  cuteafd_kache_setup "$1"
}

cuteafd_sccache_cuda_setup() {
  [[ "${CUTEAFD_SCCACHE_CUDA:-0}" == 1 ]] || return 0
  [[ -z "${CMAKE_CUDA_COMPILER_LAUNCHER:-}" ]] || return 0
  local wrapper cache
  wrapper="$(command -v sccache 2>/dev/null)" || {
    cuteafd_compiler_cache_warn 'sccache not found'; return 0;
  }
  cache="${SCCACHE_DIR:-$1/cache/sccache}/$(uname -m)"
  if ! python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" >/dev/null 2>&1 ||
     ! mkdir -p "$cache" || ! timeout 5 "$wrapper" --version >/dev/null 2>&1; then
    cuteafd_compiler_cache_warn 'sccache cache/tool unavailable'; return 0
  fi
  export SCCACHE_DIR="$cache" CUTEAFD_SCCACHE_CUDA_ACTIVE="$wrapper"
  # WIP containers share the host network, so the default 127.0.0.1:4226
  # server would be another container's daemon, whose mount namespace lacks
  # this build's files ("No such file or directory" in CMake's CUDA probe).
  # Give each build its own port. Host-network containers share one hostname
  # and mount the cache and /wip/build at fixed paths, so key the port on the
  # container's root filesystem, which differs per container (and per host).
  if [[ -z "${SCCACHE_SERVER_PORT:-}" ]]; then
    local key
    key="$cache:$(realpath -m "$1"):$(awk '$5 == "/" {print; exit}' /proc/self/mountinfo 2>/dev/null):$(hostname)"
    export SCCACHE_SERVER_PORT=$((20000 + $(printf '%s' "$key" | cksum | cut -d' ' -f1) % 20000))
  fi
  export CMAKE_CUDA_COMPILER_LAUNCHER="$wrapper"
}

cuteafd_kache_setup() {
  [[ -n "${CUTEAFD_KACHE:-}" ]] || return 0
  export CUTEAFD_KACHE_MODE=disabled
  local build_dir="$1" wrapper cache config remote="${CUTEAFD_KACHE_REMOTE:-}"
  wrapper="$(command -v -- "$( [[ "$CUTEAFD_KACHE" == 1 ]] && printf kache || printf '%s' "$CUTEAFD_KACHE" )" 2>/dev/null)" || {
    cuteafd_compiler_cache_warn "kache not found ($CUTEAFD_KACHE)"; return 0;
  }
  [[ -x "$wrapper" ]] && timeout 5 "$wrapper" --version >/dev/null 2>&1 || {
    cuteafd_compiler_cache_warn "kache is not executable on $(uname -m)"; return 0;
  }
  # Do not silently replace an agent's existing wrapper or native toolchain.
  if [[ -n "${RUSTC_WRAPPER:-}${RUSTC_WORKSPACE_WRAPPER:-}${CMAKE_C_COMPILER_LAUNCHER:-}${CMAKE_CXX_COMPILER_LAUNCHER:-}" ||
        "${CC:-cc}" == *[[:space:]]* || "${CXX:-c++}" == *[[:space:]]* ]]; then
    cuteafd_compiler_cache_warn 'an existing compiler wrapper is configured'; return 0
  fi
  cache="${CUTEAFD_KACHE_CACHE_DIR:-$HOME/.cache/cuteafd/builds/compiler-cache/kache}/$(uname -m)"
  # Local index/runtime must stay on a build-safe local filesystem, never the remote.
  if ! cuteafd_compiler_cache_local_fs "$cache" >/dev/null 2>&1 ||
     ! python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" > /dev/null 2>&1 ||
     ! mkdir -p "$cache" "$build_dir/compiler-cache"; then
    cuteafd_compiler_cache_warn "local cache unavailable ($cache)"; return 0
  fi
  if [[ -n "$remote" ]]; then
    # Run the probe out of process: an unavailable hard NFS mount must not hang a build.
    if ! timeout -k 1 3 python3 - "$remote" <<'PY'
import os, sys, tempfile
from pathlib import Path
path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=".cuteafd-kache-probe-", dir=path) as directory:
    old = Path(directory) / "stage"
    old.write_bytes(b"cache probe")
    new = Path(directory) / "committed"
    os.replace(old, new)
    assert new.read_bytes() == b"cache probe"
PY
    then
      cuteafd_compiler_cache_warn "remote unavailable ($remote)"; return 0
    fi
  fi
  config="$build_dir/compiler-cache/config.toml"
  if ! python3 - "$config" "$remote" "$(uname -m)" <<'PY'
import json, sys
from pathlib import Path
text = '''[cache]
local_max_size = "30GiB"
auto_clean_orphaned_targets = false
auto_share_target_files = false
auto_clean_unused_units_days = 0
seed_new_targets = false
adaptive_incremental = false
build_script_hermetic = false
'''
if sys.argv[2]:
    text += '\n[cache.remote]\ntype = "filesystem"\npath = ' + json.dumps(sys.argv[2])
    text += '\nprefix = ' + json.dumps('artifacts/' + sys.argv[3]) + '\n'
Path(sys.argv[1]).write_text(text)
PY
  then
    cuteafd_compiler_cache_warn 'cannot write cache configuration'; return 0
  fi
  # cc recognizes this basename; cmake-rs otherwise drops the cc wrapper.
  local launcher
  launcher="$(realpath "${BASH_SOURCE[0]}")"
  if [[ "$launcher" == *[[:space:]]* || "$wrapper" == *[[:space:]]* ]]; then
    cuteafd_compiler_cache_warn 'wrapper paths contain whitespace'; return 0
  fi
  # Some AOT exporters exec CC/CXX as a single path, not a shell command.
  # Keep shims build-local so concurrent toolchains cannot overwrite each other.
  local shim_dir
  if ! shim_dir="$(python3 - "$build_dir/compiler-cache/bin" "$launcher" "${CC:-cc}" "${CXX:-c++}" <<'PY'
import hashlib, os, shlex, shutil, sys
from pathlib import Path
compilers = [shutil.which(name) for name in sys.argv[3:]]
if not all(compilers):
    raise SystemExit('cannot resolve C/C++ compilers')
compilers = [str(Path(compiler).resolve()) for compiler in compilers]
# A new toolchain needs a new shim path so CMake's compiler-change check sees it.
identity = hashlib.sha256('\n'.join(compilers).encode()).hexdigest()[:16]
root = Path(sys.argv[1]).resolve() / identity
root.mkdir(parents=True, exist_ok=True)
for name, compiler in zip(('cc', 'c++'), compilers):
    path = root / name
    text = '#!/bin/sh\nexec ' + shlex.quote(sys.argv[2]) + ' ' + shlex.quote(str(Path(compiler).resolve())) + ' "$@"\n'
    if not path.exists() or path.read_text() != text:
        stage = path.with_suffix('.tmp')
        stage.write_text(text)
        stage.chmod(0o755)
        os.replace(stage, path)
print(root)
PY
)"; then
    cuteafd_compiler_cache_warn 'cannot create compiler shims'; return 0
  fi
  export CUTEAFD_KACHE_ACTIVE="$wrapper" KACHE_CONFIG="$config" KACHE_HOST_CONFIG=
  export KACHE_CACHE_DIR="$cache" KACHE_BUILD_SCRIPT_CACHE=0 KACHE_OUT_DIR_ALIAS=0
  export KACHE_VERIFY_RESTORES=always
  export CUTEAFD_KACHE_WARNING_DIR="$build_dir/compiler-cache/warned"
  rmdir "$CUTEAFD_KACHE_WARNING_DIR" 2>/dev/null || true
  export RUSTC_WRAPPER="$launcher" CC_KNOWN_WRAPPER_CUSTOM=compiler-cache
  # kache restores and plain-retry fallbacks leave rustc's incremental
  # dep-graph/query-cache partial ("missing incremental ... paths"), which
  # forces a fresh target. kache already caches whole crates, so incremental
  # adds nothing under it.
  export CARGO_INCREMENTAL=0
  export CC="$shim_dir/cc" CXX="$shim_dir/c++" CUTEAFD_KACHE_SHIM_DIR="$shim_dir"
  export CMAKE_C_COMPILER_LAUNCHER="$launcher" CMAKE_CXX_COMPILER_LAUNCHER="$launcher"
  export CUTEAFD_KACHE_MODE=enabled
}

cuteafd_compiler_cache_local_fs() {
  timeout -k 1 5 python3 - "$1" <<'PY'
import json, subprocess, sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
while not path.exists():
    path = path.parent
mounts = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(path),
                                           '--output', 'FSTYPE'], text=True))['filesystems']
raise SystemExit(0 if len(mounts) == 1 and mounts[0]['fstype'] in
                 {'ext4', 'xfs', 'btrfs', 'zfs', 'tmpfs', 'overlay'} else 1)
PY
}

cuteafd_compiler_cache_check_cmake_compilers() {
  local native_dir="$1"
  [[ -f "$native_dir/CMakeCache.txt" ]] || return 0
  # CMake ignores new CC/CXX values after its first configure. Refuse before
  # building with stale shims (including the former two-word wrapper setup).
  python3 - "$native_dir" "${CC:-cc}" "${CXX:-c++}" <<'PY'
import shlex, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
values = {}
for line in (root / 'CMakeCache.txt').read_text().splitlines():
    if line.startswith('CMAKE_') and ':' in line and '=' in line:
        key, value = line.split('=', 1)
        values[key.split(':', 1)[0]] = value
for language, current in zip(('C', 'CXX'), sys.argv[2:]):
    cached = values.get(f'CMAKE_{language}_COMPILER')
    if not cached:
        continue
    # Preserve support for callers' plain, multi-word compiler commands.
    resolved = shutil.which(current) or shutil.which(shlex.split(current)[0])
    if resolved and Path(cached).resolve() == Path(resolved).resolve():
        continue
    print(f'error: CMake {language} compiler changed: {cached} -> {current}; '
          f'rerun with fresh configure (remove {root / "CMakeCache.txt"} and '
          f'{root / "CMakeFiles"} first)', file=sys.stderr)
    raise SystemExit(1)
PY
}

cuteafd_compiler_cache_cmake_args() {
  local native_dir="$1" language key launcher cached
  for language in C CXX CUDA; do
    key="CMAKE_${language}_COMPILER_LAUNCHER"
    launcher="${!key:-}"
    if [[ -n "$launcher" ]]; then
      # Never override an unmanaged launcher just because another language opted in.
      printf '%s\n' "-D$key=$launcher"
    elif [[ -f "$native_dir/CMakeCache.txt" ]]; then
      cached="$(grep -E "^$key:[^=]*=" "$native_dir/CMakeCache.txt" | cut -d= -f2- || true)"
      if [[ "$cached" == *compiler-cache.sh* || "$cached" == */sccache || "$cached" == sccache ]]; then
        printf '%s\n' "-D$key="
      fi
    fi
  done
}

# Docker argv rendering is deliberately a no-op unless explicitly configured.
# Call on the Docker host; ARM machines need their own native kache executable.
cuteafd_compiler_cache_docker_args() {
  cuteafd_sccache_cuda_docker_args
  cuteafd_kache_docker_args
}

cuteafd_sccache_cuda_docker_args() {
  [[ "${CUTEAFD_SCCACHE_CUDA:-0}" == 1 ]] || return 0
  local cache="${CUTEAFD_SCCACHE_CACHE_DIR:-$HOME/.cache/cuteafd/builds/compiler-cache/sccache}"
  [[ "$cache" != *','* ]] || { cuteafd_compiler_cache_warn 'cache path contains comma'; return 0; }
  python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" >/dev/null 2>&1 && mkdir -p "$cache" || {
    cuteafd_compiler_cache_warn 'sccache host cache unavailable'; return 0;
  }
  printf '%s\n' --mount "type=bind,src=$(realpath "$cache"),dst=/opt/cuteafd-sccache-cache" \
    -e CUTEAFD_SCCACHE_CUDA=1 -e SCCACHE_DIR=/opt/cuteafd-sccache-cache
}

cuteafd_kache_docker_args() {
  [[ -n "${CUTEAFD_KACHE:-}" ]] || return 0
  # Even an unavailable opt-in records that this build actually ran plain.
  printf '%s\n' -e CUTEAFD_KACHE_REQUESTED=1
  local wrapper cache
  if [[ "$CUTEAFD_KACHE" == 1 ]]; then
    wrapper=/opt/cuteafd-kache
  else
    wrapper="$(command -v -- "$CUTEAFD_KACHE" 2>/dev/null)" || {
      cuteafd_compiler_cache_warn "kache not found ($CUTEAFD_KACHE)"; return 0;
    }
  fi
  if [[ "$CUTEAFD_KACHE" != 1 ]] && ! [[ -f "$wrapper" && -x "$wrapper" ]]; then
    cuteafd_compiler_cache_warn "kache is not an executable file ($wrapper)"; return 0
  fi
  [[ "$CUTEAFD_KACHE" == 1 ]] || wrapper="$(realpath "$wrapper")"
  cache="${CUTEAFD_KACHE_CACHE_DIR:-$HOME/.cache/cuteafd/builds/compiler-cache/kache}"
  cache="$(realpath -m "$cache")"
  if [[ "$wrapper$cache${CUTEAFD_KACHE_REMOTE:-}" == *','* ]]; then
    cuteafd_compiler_cache_warn 'Docker mount paths contain a comma'; return 0
  fi
  if ! cuteafd_compiler_cache_local_fs "$cache" >/dev/null 2>&1 ||
     ! python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" >/dev/null 2>&1 || ! mkdir -p "$cache"; then
    cuteafd_compiler_cache_warn "local cache unavailable ($cache)"; return 0
  fi
  if [[ -n "${CUTEAFD_KACHE_REMOTE:-}" ]] && ! timeout -k 1 3 test -d "$CUTEAFD_KACHE_REMOTE"; then
    cuteafd_compiler_cache_warn "remote unavailable ($CUTEAFD_KACHE_REMOTE)"; return 0
  fi
  [[ "$CUTEAFD_KACHE" == 1 ]] || printf '%s\n' --mount "type=bind,src=$wrapper,dst=/opt/cuteafd-kache,readonly"
  printf '%s\n' --mount "type=bind,src=$cache,dst=/opt/cuteafd-kache-cache" \
    -e CUTEAFD_KACHE=/opt/cuteafd-kache -e CUTEAFD_KACHE_CACHE_DIR=/opt/cuteafd-kache-cache
  if [[ -n "${CUTEAFD_KACHE_REMOTE:-}" ]]; then
    printf '%s\n' --mount "type=bind,src=$CUTEAFD_KACHE_REMOTE,dst=/opt/cuteafd-kache-remote" \
      -e CUTEAFD_KACHE_REMOTE=/opt/cuteafd-kache-remote
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  # CMake also uses us as a launcher. Let the shim wrap its real compiler once;
  # otherwise kache sees a shell script as the compiler, not cc/c++.
  if [[ -n "${CUTEAFD_KACHE_SHIM_DIR:-}" ]] &&
     [[ "${1:-}" == "$CUTEAFD_KACHE_SHIM_DIR/cc" || "${1:-}" == "$CUTEAFD_KACHE_SHIM_DIR/c++" ]]; then
    exec "$@"
  fi
  # One failing cache invocation disables it for the remainder of this build.
  if [[ -d "${CUTEAFD_KACHE_WARNING_DIR:-/nonexistent}" ]]; then
    exec "$@"
  fi
  # Preserve genuine compiler errors, but never let an optional cache break a build.
  if [[ -n "${CUTEAFD_KACHE_ACTIVE:-}" ]] &&
     timeout -k 5 "${CUTEAFD_KACHE_TIMEOUT_SECONDS:-300}" "$CUTEAFD_KACHE_ACTIVE" "$@"; then
    exit 0
  fi
  if [[ -n "${CUTEAFD_KACHE_ACTIVE:-}" ]] && mkdir "${CUTEAFD_KACHE_WARNING_DIR:-/nonexistent}" 2>/dev/null; then
    printf 'warning: cuteafd kache invocation failed or timed out; retrying with plain compiler\n' >&2
  fi
  exec "$@"
fi
