#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/lib/build-supervision.sh"
usage() {
  printf '%s\n' 'Usage: build-dev-images.sh [--config FILE] [--spark-hosts HOST,...]' \
    '  [--seed-hosts HOST,...] [--export-hosts HOST,...] [--role coordinator|expert|both]' \
    '  [--distribute-only] [--publish] [--dry-run]' \
    'Build toolchain-only dev images natively, independent of SparkInfer pins.' \
    '--publish uses only ghcr.io/tpurtell/cuteafd-dev:tc-HASH{-amd64,-arm64} and latest.' \
    'Publishing never retags local development defaults. Use rhea or a LAN-only seed as the Spark seed.' \
    '--seed-hosts: build-only Spark seeds that also receive the Spark image' \
    '  (default: the LAN-only seeds, kakapo,cassowary; empty for none).' \
    '--export-hosts: remote coordinator export hosts that receive the coordinator image' \
    '  (default: the config'"'"'s CUTEAFD_EXPORT_HOST, reached by ssh hostname; empty for none).' \
    '  Its previous image stays under :prev-ID; a running container keeps its image.' \
    '--distribute-only: build nothing and retag nothing here or on the seed; copy this toolchain'"'"'s' \
    '  existing images to receivers whose copy differs (layer list). A host that already matches is skipped.' \
    'A copy to or from a host without RoCE goes zstd-compressed over AES-GCM ssh on its LAN address.' \
    'Both architectures build concurrently; CUTEAFD_DEV_IMAGE_SEQUENTIAL=1 restores coordinator-first order.' \
    'Hold build.lock and the Spark seed host lock around actual builds.'
}
config="$repo_root/cuteafd.config"; hosts_csv=; role=both; dry_run=0; publish=0; distribute_only=0
seed_hosts_csv="$(IFS=,; echo "${release_lan_seed_hosts[*]}")"; export_hosts_csv=__config__
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?}"; shift 2 ;;
    --spark-hosts) hosts_csv="${2:?}"; shift 2 ;;
    --seed-hosts) seed_hosts_csv="${2-}"; shift 2 ;;
    --export-hosts) export_hosts_csv="${2-}"; shift 2 ;;
    --distribute-only) distribute_only=1; shift ;;
    --role) role="${2:?}"; shift 2 ;;
    --publish) publish=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) release_die "unknown argument: $1" ;;
  esac
done
case "$role" in coordinator|expert|both) ;; *) release_die 'invalid --role' ;; esac
(( ! publish )) || [[ "$role" == both ]] || release_die 'publishing requires both architectures'
(( ! publish || ! distribute_only )) || release_die '--distribute-only never publishes'
sequential="${CUTEAFD_DEV_IMAGE_SEQUENTIAL:-0}"
[[ "$sequential" == 0 || "$sequential" == 1 ]] || release_die 'CUTEAFD_DEV_IMAGE_SEQUENTIAL must be 0 or 1'
release_load_config "$config"
for tool in docker ssh rsync python3; do release_need "$tool"; done
if ((publish)); then
  [[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] ||
    release_die 'refusing to publish from a dirty tree'
fi
hash="$(python3 "$repo_root/scripts/build/dev-toolchain.py")"
revision="$(git -C "$repo_root" rev-parse HEAD)"
base_amd64="$(python3 "$repo_root/scripts/build/dev-toolchain.py" amd64)"
base_arm64="$(python3 "$repo_root/scripts/build/dev-toolchain.py" arm64)"
repo_of() { local image="$1"; [[ "${image##*/}" != *:* ]] && printf '%s' "$image" || printf '%s' "${image%:*}"; }
coordinator_tag="$(repo_of "$COORDINATOR_DOCKER_DEV"):tc-$hash"
spark_tag="$(repo_of "$SPARK_EXPERT_DOCKER_DEV"):tc-$hash"
package=ghcr.io/tpurtell/cuteafd-dev
if ((publish)); then
  coordinator_tag="$package:tc-$hash-amd64"; spark_tag="$package:tc-$hash-arm64"
fi
if [[ -n "$hosts_csv" ]]; then IFS=, read -ra spark_hosts <<<"$hosts_csv"
else
  spark_hosts=()
  for i in 0 1 2 3 4 5; do name="SPARK_${i}_HOST"; [[ -z "${!name:-}" ]] || spark_hosts+=("${!name}"); done
fi
seed_only_hosts=()
[[ -z "$seed_hosts_csv" ]] || IFS=, read -ra seed_only_hosts <<<"$seed_hosts_csv"
# Remote coordinator export hosts (scripts/build/export-remote.sh) need the exact
# coordinator dev image: they admit an export only when its layer list matches.
[[ "$export_hosts_csv" != __config__ ]] || export_hosts_csv="${CUTEAFD_EXPORT_HOST:-}"
export_hosts=()
[[ -z "$export_hosts_csv" ]] || IFS=, read -ra export_hosts <<<"$export_hosts_csv"
# export_ssh HOST ARGS...: connect by name, as scripts/build/export-remote.sh does.
export_ssh() {
  local host="$1"
  shift
  release_ssh -o ConnectTimeout=10 "$host" "$@"
}
if [[ "$role" != coordinator ]]; then
  ((${#spark_hosts[@]})) || release_die 'no Spark build host'
  seed_host="${spark_hosts[0]}"
  # Build-only seeds receive the image too (each once, never the seed itself).
  for host in ${seed_only_hosts[@]+"${seed_only_hosts[@]}"}; do
    [[ " ${spark_hosts[*]} " == *" $host "* ]] || spark_hosts+=("$host")
  done
fi
printf 'Toolchain: tc-%s\n  amd64: %s\n  arm64: %s\n' "$hash" "$coordinator_tag" "$spark_tag"
((publish)) || [[ "$role" == expert ]] || ((${#export_hosts[@]} == 0)) ||
  printf 'Export hosts receive the coordinator image: %s\n' "${export_hosts[*]}"
((!distribute_only)) || printf 'Distribute only: no build, no retag on this host or the seed.\n'
((!publish)) || printf '  index: %s:tc-%s\n  latest: %s:latest\n' "$package" "$hash" "$package"
case "$distribute_only:$role:$sequential" in
  1:*) printf 'Plan: distribute existing %s image(s) only.\n' "$role" ;;
  0:both:0) printf 'Plan: coordinator (amd64) and expert (arm64) legs build concurrently; checks/retag/distribution/publish wait for both.\n' ;;
  0:both:1) printf 'Plan: sequential: coordinator then expert (CUTEAFD_DEV_IMAGE_SEQUENTIAL=1); shared steps wait for both.\n' ;;
  *) printf 'Plan: %s leg only.\n' "$role" ;;
esac
[[ "$role" == coordinator ]] || printf 'Spark seed: %s; receives the image: %s\n' "$seed_host" "${spark_hosts[*]:1}"
if ((dry_run)); then printf 'Dry run: nothing built or pushed.\n'; exit 0; fi
# layers_of RUNNER IMAGE: the image's content identity (its RootFS layer list). An
# image ID is not portable: docker save | load can give the same layers another ID.
# Empty when the image is missing; otherwise the digest export-remote.sh compares.
layers_of() {
  "${@:1:$#-1}" "l=\$(docker image inspect -f '{{json .RootFS.Layers}}' '${!#}' 2>/dev/null) && printf '%s\\n' \"\$l\" | sha256sum | cut -c1-64" 2>/dev/null || true
}
# copy_coordinator_image HOST: this host's coordinator image to an export host,
# skipped when its layers already match; then retagged there like a builder.
copy_coordinator_image() {
  local host="$1" want have
  want="$(layers_of bash -c "$coordinator_tag")"
  [[ -n "$want" ]] || release_die "no local $coordinator_tag to distribute"
  have="$(layers_of export_ssh "$host" "$coordinator_tag")"
  if [[ -n "$want" && "$want" == "$have" ]]; then
    printf 'Export host %s already has %s\n' "$host" "$coordinator_tag"
  else
    printf 'Copying %s to export host %s compressed\n' "$coordinator_tag" "$host"
    docker image save "$coordinator_tag" | zstd -T8 -3 -q |
      export_ssh "$host" "${release_bulk_ssh_opts[@]}" "zstd -d -q | docker image load" >/dev/null
  fi
  export_ssh "$host" bash -s -- "$COORDINATOR_DOCKER_DEV" "$coordinator_tag" "$(repo_of "$COORDINATOR_DOCKER_DEV")" <<<"$retag_body"
  [[ "$(layers_of export_ssh "$host" "$COORDINATOR_DOCKER_DEV")" == "$want" ]] ||
    release_die "export host $host did not receive $coordinator_tag"
}
# Preserve old si-* images for campaigns, and old tc-* images before moving a local default.
# A same-toolchain rebuild (or a refreshed receiver: docker load moves the tc- tag)
# keeps the old image as :prev-ID, never under the tc- tag the new image now owns.
# Builders and receivers share this; a running container keeps its image.
retag_body='set -euo pipefail
image="$1"; versioned="$2"; repo="$3"
new="$(docker image inspect -f "{{.Id}}" "$versioned" 2>/dev/null || true)"
old="$(docker image inspect -f "{{.Id}}" "$image" 2>/dev/null || true)"
if [[ -n "$old" && "$old" != "$new" ]]; then
  si="$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.sparkinfer.revision\"}}" "$old")"
  tc="$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.toolchain.hash\"}}" "$old")"
  [[ -z "$si" || "$si" == "<no value>" ]] || docker tag "$old" "$repo:si-${si:0:7}"
  if [[ -z "$tc" || "$tc" == "<no value>" || "$repo:tc-$tc" == "$versioned" ]]; then
    docker tag "$old" "$repo:prev-${old:7:12}"
  else
    docker tag "$old" "$repo:tc-$tc"
  fi
fi
docker tag "$versioned" "$image"'
check_body='set -euo pipefail
[[ "$(docker image inspect -f "{{index .Config.Labels \"io.cuteafd.toolchain.hash\"}}" "$1")" == "$2" ]] || { printf "toolchain label mismatch\n" >&2; exit 1; }
[[ "$(docker image inspect -f "{{.Architecture}}" "$1")" == "$3" ]] || { printf "architecture mismatch\n" >&2; exit 1; }'
release_need mkfifo
log_root="${CUTEAFD_DEV_IMAGE_LOG_ROOT:-$HOME/.cache/cuteafd/builds/devimg}"
release_validate_path_setting CUTEAFD_DEV_IMAGE_LOG_ROOT "$log_root"
python3 "$repo_root/scripts/build/assert-build-filesystem.py" "$log_root"
mkdir -p "$log_root"
leg_log_dir="$(mktemp -d "$log_root/build-legs.XXXXXXXX")"
run_id="${leg_log_dir##*/}"

build_coordinator_dev() (
  [[ "$(uname -m)" == x86_64 ]] || release_die 'coordinator build must be native amd64'
  DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$base_amd64" --build-arg CUTEAFD_ENGINE_COMMIT="$revision" \
    --build-arg CUTEAFD_TOOLCHAIN_HASH="$hash" -f "$repo_root/docker/Dockerfile.dev" -t "$coordinator_tag" "$repo_root"
  bash -c "$check_body" _ "$coordinator_tag" "$hash" amd64
)
build_expert_dev() (
  [[ "$(ssh -o BatchMode=yes "$seed_host" uname -m)" == aarch64 ]] || release_die 'Spark build must be native arm64'
  remote_dir="${CUTEAFD_DEV_IMAGE_REMOTE_DIR:-$(ssh -o BatchMode=yes "$seed_host" 'printf %s "$HOME"')/.cache/cuteafd/builds/devimg-$hash}"
  release_validate_path_setting CUTEAFD_DEV_IMAGE_REMOTE_DIR "$remote_dir"
  ssh -o BatchMode=yes "$seed_host" python3 - "$remote_dir" <"$repo_root/scripts/build/assert-build-filesystem.py"
  ssh -o BatchMode=yes "$seed_host" "mkdir -p '$remote_dir'"
  (cd "$repo_root" && rsync -aR --delete --exclude .git .dockerignore docker/ \
    scripts/build/install-dev-cache-tools.sh scripts/build/install-nsys.sh "$seed_host:$remote_dir/")
  ssh -o BatchMode=yes "$seed_host" setsid --wait bash -s -- \
    "$run_id" "$remote_dir" "$base_arm64" "$revision" "$hash" "$spark_tag" <<'REMOTE'
set -euo pipefail
# Independent of the staging directory, so cancellation can arrive during rsync.
process_dir="$HOME/.cache/cuteafd/builds/devimg-processes"
process_file="$process_dir/$1.pid"
cancel_file="$process_dir/$1.cancel"
mkdir -p "$process_dir"
[[ ! -e "$cancel_file" ]] || exit 143
printf '%s\n' "$$" >"$process_file"
cleanup_dev_phase() {
  trap '' HUP INT TERM
  kill -TERM -- "-$$" 2>/dev/null || true
  rm -f "$process_file"
}
trap cleanup_dev_phase EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
# Close the race between the first marker check and PID publication.
[[ ! -e "$cancel_file" ]] || exit 143
cd "$2"
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE="$3" --build-arg CUTEAFD_ENGINE_COMMIT="$4" \
  --build-arg CUTEAFD_TOOLCHAIN_HASH="$5" -f docker/Dockerfile.dev -t "$6" .
REMOTE
  ssh -o BatchMode=yes "$seed_host" bash -s -- "$spark_tag" "$hash" arm64 <<<"$check_body"
)
cancel_coordinator_dev() { :; } # The supervised group owns the local Docker client.
cancel_expert_dev() {
  # Killing the SSH client alone does not reliably stop a remote Docker build.
  timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=5 "$seed_host" bash -s -- "$run_id" <<'CANCEL'
set -euo pipefail
process_dir="$HOME/.cache/cuteafd/builds/devimg-processes"
mkdir -p "$process_dir"
: >"$process_dir/$1.cancel"
pid=""
[[ ! -f "$process_dir/$1.pid" ]] || read -r pid <"$process_dir/$1.pid" || true
if [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]] &&
   grep -zFq -- "$1" "/proc/$pid/cmdline"; then
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((i=0; i<5; i++)); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || true
fi
rm -f "$process_dir/$1.pid"
CANCEL
}
# copy_spark_image SRC DST: the Spark image from SRC to DST, relayed through this
# host. Between RoCE hosts it streams as is; when either end has no RDMA (a
# 10 GbE build seed) it goes zstd-compressed over AES-GCM on that end's LAN address.
copy_spark_image() {
  local src="$1" dst="$2" src_addr= dst_addr=
  if release_host_has_rdma "$src" && release_host_has_rdma "$dst"; then
    ssh -o BatchMode=yes "$src" "docker image save '$spark_tag'" | ssh -o BatchMode=yes "$dst" 'docker image load'
    return
  fi
  release_host_has_rdma "$src" || src_addr="$(python3 "$repo_root/scripts/build/seed-transfer.py" lan-addr "$src")"
  release_host_has_rdma "$dst" || dst_addr="$(python3 "$repo_root/scripts/build/seed-transfer.py" lan-addr "$dst")"
  printf 'Copying %s %s -> %s compressed\n' "$spark_tag" "$src" "$dst"
  release_bulk_ssh "$src" "$src_addr" "docker image save '$spark_tag' | zstd -T8 -3 -q" |
    release_bulk_ssh "$dst" "$dst_addr" 'zstd -d -q | docker image load'
}
# distribute_spark_image HOST: the seed's Spark image to HOST unless its layers
# already match, then the local default retagged there.
distribute_spark_image() {
  local host="$1" want
  want="$(layers_of ssh -o BatchMode=yes "$seed_host" "$spark_tag")"
  if [[ -n "$want" && "$(layers_of ssh -o BatchMode=yes "$host" "$spark_tag")" == "$want" ]]; then
    printf '%s already has %s\n' "$host" "$spark_tag"
  else
    copy_spark_image "$seed_host" "$host"
  fi
  ssh -o BatchMode=yes "$host" bash -s -- "$SPARK_EXPERT_DOCKER_DEV" "$spark_tag" "$(repo_of "$SPARK_EXPERT_DOCKER_DEV")" <<<"$retag_body"
}
if ((distribute_only)); then
  # Nothing is built and nothing changes on this host or the seed: receivers only.
  if [[ "$role" != coordinator ]]; then
    for host in "${spark_hosts[@]:1}"; do distribute_spark_image "$host"; done
  fi
  if [[ "$role" != expert ]]; then
    for host in ${export_hosts[@]+"${export_hosts[@]}"}; do copy_coordinator_image "$host"; done
  fi
  printf 'Done. Receivers hold this toolchain'"'"'s images; running containers keep theirs.\n'
  exit 0
fi
case "$role" in
  both) build_supervise 'dev image build' "$leg_log_dir" "$sequential" \
    coordinator build_coordinator_dev cancel_coordinator_dev amd64.log \
    expert build_expert_dev cancel_expert_dev arm64.log ;;
  coordinator) build_supervise 'dev image build' "$leg_log_dir" 0 \
    coordinator build_coordinator_dev cancel_coordinator_dev amd64.log ;;
  expert) build_supervise 'dev image build' "$leg_log_dir" 0 \
    expert build_expert_dev cancel_expert_dev arm64.log ;;
esac
# Shared mutations happen only after every selected leg's build and check pass.
if ((!publish)); then
  if [[ "$role" != expert ]]; then
    bash -c "$retag_body" _ "$COORDINATOR_DOCKER_DEV" "$coordinator_tag" "$(repo_of "$COORDINATOR_DOCKER_DEV")"
    # Remote coordinator exports admit only a layer-identical image.
    for host in ${export_hosts[@]+"${export_hosts[@]}"}; do copy_coordinator_image "$host"; done
  fi
  if [[ "$role" != coordinator ]]; then
    ssh -o BatchMode=yes "$seed_host" bash -s -- "$SPARK_EXPERT_DOCKER_DEV" "$spark_tag" "$(repo_of "$SPARK_EXPERT_DOCKER_DEV")" <<<"$retag_body"
    for host in "${spark_hosts[@]:1}"; do distribute_spark_image "$host"; done
  fi
fi
if ((publish)); then
  # Validate both before pushing either; publish only this explicitly approved package.
  bash -c "$check_body" _ "$coordinator_tag" "$hash" amd64
  ssh -o BatchMode=yes "$seed_host" bash -s -- "$spark_tag" "$hash" arm64 <<<"$check_body"
  docker push "$coordinator_tag"
  ssh -o BatchMode=yes "$seed_host" docker push "$spark_tag"
  docker buildx imagetools create -t "$package:tc-$hash" -t "$package:latest" "$coordinator_tag" "$spark_tag"
  for tag in "$coordinator_tag" "$spark_tag" "$package:tc-$hash"; do docker buildx imagetools inspect "$tag"; done
  # Use a temporary empty Docker config so the visibility probe cannot reuse login credentials.
  anon="$(mktemp -d)"; trap 'rm -rf "$anon"' EXIT
  if DOCKER_CONFIG="$anon" docker pull "$package:tc-$hash" >/dev/null; then
    printf 'Anonymous manifest access: public\n'
  else
    printf 'Anonymous access failed; if private, TJ must flip package visibility in GitHub UI.\n' >&2
  fi
fi
printf 'Done. Recreate WIP containers only when changing toolchains.\n'
