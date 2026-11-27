#!/usr/bin/env bash
# Resolve the PeerSwarm binary and repo-contained cache root for CuteAFD scripts.

peerswarm_repo_root() {
  if [[ -n "${PEERSWARM_ROOT:-}" && -f "$PEERSWARM_ROOT/peerswarm.toml" ]]; then
    printf '%s\n' "$PEERSWARM_ROOT"
    return 0
  fi
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/peerswarm.toml" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P
}

peerswarm_bin() {
  if [[ -n "${PEERSWARM_BIN:-}" && -x "$PEERSWARM_BIN" ]]; then
    printf '%s\n' "$PEERSWARM_BIN"
    return 0
  fi
  local root
  root="$(peerswarm_repo_root)"
  if [[ -x "$root/peerswarm/target/release/peerswarm" ]]; then
    printf '%s\n' "$root/peerswarm/target/release/peerswarm"
    return 0
  fi
  if command -v peerswarm >/dev/null 2>&1; then
    command -v peerswarm
    return 0
  fi
  echo "peerswarm binary not found; build it with: cargo build --release --manifest-path $root/peerswarm/Cargo.toml" >&2
  return 1
}

cuteafd_cache_root() {
  if [[ -n "${CUTEAFD_CACHE_ROOT:-}" ]]; then
    printf '%s\n' "$CUTEAFD_CACHE_ROOT"
    return 0
  fi
  if [[ -n "${PEERSWARM_CACHES:-}" ]]; then
    printf '%s\n' "$PEERSWARM_CACHES"
    return 0
  fi
  local root
  root="$(peerswarm_repo_root)"
  if [[ -f "$root/peerswarm.toml" ]]; then
    printf '%s\n' "$root/peerswarm.state/caches"
    return 0
  fi
  printf '%s\n' "$HOME/.cache/cuteafd"
}
