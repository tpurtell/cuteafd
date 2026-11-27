#!/usr/bin/env bash
# Copy verified live sources for the release compiler without host Git metadata.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: stage-release-source.sh SOURCE_DIR STAGING_DIR" >&2; exit 2; }
source_dir="$(realpath "$1")"
staging_dir="$(realpath -m "$2")"
[[ "$staging_dir" != "$source_dir" &&
   ("$staging_dir" != "$source_dir/"* || "$staging_dir" == "$source_dir/peerswarm.state/"*) &&
   "$source_dir" != "$staging_dir/"* ]] || {
  echo "release staging directory must be outside the source tree" >&2; exit 2;
}
python3 "$(dirname "$0")/assert-build-filesystem.py" "$staging_dir"
mkdir -p "$staging_dir"
rsync -a --delete --delete-excluded \
  --exclude .git --exclude '.venv*/' --exclude .mypy_cache/ \
  --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude __pycache__/ \
  --exclude '*.pyc' --exclude '*.pyo' --exclude .cuteafd-cache/ \
  --exclude .cuteafd-release/ --exclude .cuteafd-release-image/ \
  --exclude .cuteafd-wip --exclude peerswarm.state/ --exclude peerswarm/target/ \
  --exclude dist/ --exclude rust/target/ \
  --exclude 'native/build*/' \
  "$source_dir/" "$staging_dir/"
