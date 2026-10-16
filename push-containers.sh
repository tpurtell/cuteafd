#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./push-containers.sh [--config FILE] [--no-latest | --latest] TAG

Tags and pushes the current coordinator and Spark inference images to GHCR.
The supplied release tag is published for both images. Stable tags also update
latest; release-candidate tags (rc, case-insensitive) never update latest.
--latest is refused for release candidates. --no-latest disables latest for any tag. The
coordinator image is local; the Spark image is published from SPARK_0_HOST.
The Spark image must advertise the V41 expert roles it carries (./build.sh bakes
the universal tp2;tp3;tp6 set by default); a role-less legacy build is rejected.

--config FILE selects the configuration that names the local image pair to
publish (default: ./cuteafd.config, which after the v10 runtime promotion names
the v10 pair). The explicit v10 BUILD target is retained and is now identical
to the runtime default:
  ./push-containers.sh --config cuteafd.build-v10.config v10
The tag argument is unchanged and is still what both images are published as;
the two GHCR repositories are fixed.

Examples:
  ./push-containers.sh v10
  ./push-containers.sh --config cuteafd.build-v10.config v10

Every remote step shares one SSH option set with ./build.sh and ./run.sh:
  CUTEAFD_RELEASE_SSH_CONFIG       ssh config file to use (default empty: stock
                                  OpenSSH resolution; BatchMode is always forced so
                                  a publish can never wait on a prompt).
                                  /dev/null discards a broken system include but
                                  also drops your ~/.ssh/config host aliases, so
                                  prefer a file containing 'Include ~/.ssh/config'.
EOF
}

config="$repo_root/cuteafd.config"
tag=
latest=auto
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-latest|--latest)
      requested=on
      [[ "$1" != --no-latest ]] || requested=off
      [[ "$latest" == auto || "$latest" == "$requested" ]] ||
        release_die "--latest and --no-latest conflict"
      latest="$requested"
      shift
      ;;
    --config)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--config requires a configuration file"
      config="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      release_die "unknown push argument: $1"
      ;;
    *)
      [[ -z "$tag" ]] || {
        usage >&2
        exit 2
      }
      tag="$1"
      shift
      ;;
  esac
done
[[ -n "$tag" ]] || {
  usage >&2
  exit 2
}

[[ "$tag" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] ||
  release_die "invalid Docker tag: $tag"
[[ "$tag" != latest ]] ||
  release_die "provide a version tag, not latest"
if [[ "${tag,,}" == *rc* ]]; then
  [[ "$latest" != on ]] || release_die "release-candidate tags cannot update latest"
  latest=off
elif [[ "$latest" == auto ]]; then
  latest=on
fi

release_load_config "$config"
release_need docker
release_need ssh

# One SSH option set for the whole release pipeline: scripts/lib/release-common.sh owns
# it. Resolved here, after the tag and configuration validation, so a mistyped
# CUTEAFD_RELEASE_SSH_CONFIG is reported as itself before the daemon is queried and
# before any Spark is contacted. The Spark image was placed on $SPARK_0_HOST through
# that same transport, so publishing it from that host has to use it as well.
release_configure_ssh_transport

coordinator_repository="ghcr.io/tpurtell/cuteafd-coordinator"
spark_repository="ghcr.io/tpurtell/cuteafd-spark-expert"
spark_host="$SPARK_0_HOST"
expected_source="https://github.com/tpurtell/cuteafd"

# The published pair is universal: ./build.sh bakes the TP2/TP3/TP6 Spark expert
# shards by default on top of the always-built TP4 shard, and the release launcher
# refuses an explicit SPARK_TP/SPARK_EP topology without the matching role.
# push-universal-role-guard:start
push_require_universal_roles() {
  local advertised="$1" image="$2" role
  for role in tp2 tp3 tp6; do
    [[ ";$advertised;" == *";$role;"* ]] ||
      release_die "$image does not advertise Spark expert role '$role' (advertised: '${advertised:-<none>}'); refusing to publish a legacy or subset build as the universal release pair. Rebuild it with ./build.sh, whose default is tp2;tp3;tp6."
  done
}
# push-universal-role-guard:end

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE"

release_ssh -o ConnectTimeout=10 "$spark_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
image="$1"
docker info >/dev/null
docker image inspect "$image" >/dev/null
REMOTE

coordinator_revision="$(
  docker image inspect \
    -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
spark_revision="$(
  release_ssh "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$1"
REMOTE
)"
coordinator_source="$(
  docker image inspect \
    -f '{{index .Config.Labels "org.opencontainers.image.source"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
spark_source="$(
  release_ssh "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{index .Config.Labels "org.opencontainers.image.source"}}' "$1"
REMOTE
)"
[[ -n "$coordinator_revision" && "$coordinator_revision" != "<no value>" ]] ||
  release_die "coordinator image has no engine revision label"
[[ -n "$spark_revision" && "$spark_revision" != "<no value>" ]] ||
  release_die "$spark_host Spark image has no engine revision label"
[[ "$coordinator_revision" == "$spark_revision" ]] ||
  release_die "image revision mismatch: coordinator=$coordinator_revision spark=$spark_revision"
[[ "$coordinator_source" == "$expected_source" ]] ||
  release_die "coordinator image is not linked to the release repository: $coordinator_source"
[[ "$spark_source" == "$expected_source" ]] ||
  release_die "$spark_host image is not linked to the release repository: $spark_source"

spark_roles="$(
  release_ssh "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
docker image inspect \
  -f '{{or (index .Config.Labels "io.cuteafd.spark_tp_roles") (index .Config.Labels "io.cuteafd.v41.spark_tp_roles")}}' "$1"
REMOTE
)"
[[ "$spark_roles" != "<no value>" ]] || spark_roles=
push_require_universal_roles "$spark_roles" "$SPARK_EXPERT_DOCKER_INFERENCE"

echo "Publishing CUTEAFD containers"
echo "  revision:    $coordinator_revision"
echo "  expert roles: $spark_roles"
echo "  coordinator: $coordinator_repository:$tag"
echo "  spark:       $spark_repository:$tag (from $spark_host)"

docker tag "$COORDINATOR_DOCKER_INFERENCE" "$coordinator_repository:$tag"
release_ssh "$spark_host" bash -s -- \
  "$SPARK_EXPERT_DOCKER_INFERENCE" "$spark_repository:$tag" <<'REMOTE'
set -euo pipefail
docker tag "$1" "$2"
REMOTE

docker push "$coordinator_repository:$tag"
release_ssh "$spark_host" docker push "$spark_repository:$tag"

if [[ "$latest" == on ]]; then
  docker tag "$COORDINATOR_DOCKER_INFERENCE" "$coordinator_repository:latest"
  release_ssh "$spark_host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" "$spark_repository:latest" <<'REMOTE'
set -euo pipefail
docker tag "$1" "$2"
REMOTE

  docker push "$coordinator_repository:latest"
  release_ssh "$spark_host" docker push "$spark_repository:latest"
  echo "Published both $tag and latest:"
else
  echo "Published $tag only (latest unchanged):"
fi
echo "  docker pull $coordinator_repository:$tag"
echo "  docker pull $spark_repository:$tag"
