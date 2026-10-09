#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
[[ "${1:-}" == rotate && $# == 1 ]] || { printf 'usage: %s rotate\n' "$0" >&2; exit 2; }
release_prepare_secret "$HOME/.cache/cuteafd/console" secret >/dev/null
CONSOLE_SECRET_FILE="$(release_prepare_secret "$HOME/.cache/cuteafd/console" secret rotate)"
printf 'Console secret rotated; existing cookies expire within 10 seconds.\n'
