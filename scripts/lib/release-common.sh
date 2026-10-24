#!/usr/bin/env bash

RELEASE_COORDINATOR_CONTAINER_NAME=cuteafd-coordinator
RELEASE_SPARK_CONTAINER_PREFIX=cuteafd-spark-expert

release_die() {
  echo "cuteafd release: $*" >&2
  exit 2
}

# CUDA and NVML can enumerate cards differently. UUID is the only join key;
# querying the driver here creates no context or device allocation.
release_cuda_total_bytes() {
  python3 - "$1" <<'PYCAPACITY'
import ctypes, sys, uuid
try:
    wanted = uuid.UUID(sys.argv[1].removeprefix('GPU-')).bytes
    cuda = ctypes.CDLL('libcuda.so.1')
    def checked(name, *args):
        function = getattr(cuda, name)
        function.restype = ctypes.c_int
        code = function(*args)
        if code:
            raise ValueError(f'{name} failed with CUDA error {code}')
    checked('cuInit', ctypes.c_uint(0))
    count = ctypes.c_int()
    checked('cuDeviceGetCount', ctypes.byref(count))
    for ordinal in range(count.value):
        device = ctypes.c_int()
        checked('cuDeviceGet', ctypes.byref(device), ctypes.c_int(ordinal))
        identity = (ctypes.c_ubyte * 16)()
        checked('cuDeviceGetUuid_v2', ctypes.byref(identity), device)
        if bytes(identity) != wanted:
            continue
        total = ctypes.c_size_t()
        checked('cuDeviceTotalMem_v2', ctypes.byref(total), device)
        if total.value == 0:
            raise ValueError('CUDA reported zero device capacity')
        print(total.value)
        break
    else:
        raise ValueError(f'no CUDA device matches UUID {sys.argv[1]}')
except (OSError, AttributeError, ValueError) as error:
    sys.exit(f'coordinator CUDA capacity: {error}')
PYCAPACITY
}

# Resolve total capacity in bytes, never free memory. Rust owns the default rule.
release_coordinator_profile() {
  local registry="${repo_root:?}/rust/crates/cuteafd-core/src/serving_capacity.rs"
  python3 - "$registry" "$1" "${2:-}" <<'PYPROFILE'
import pathlib, re, sys
from decimal import Decimal
try:
    source = pathlib.Path(sys.argv[1]).read_text()
    def literal(name):
        declarations = re.findall(r'^pub const ' + name + r':.*$', source, re.M)
        if len(declarations) != 1:
            raise ValueError(f'missing or duplicate capacity registry constant {name}')
        match = re.fullmatch(r'pub const ' + name + r': u64 = ([0-9]+);', declarations[0])
        if match is None:
            raise ValueError(f'nonliteral capacity registry constant {name}')
        return int(match.group(1))
    threshold, small, large = (literal(name) for name in (
        'SMALL_CARD_CONCURRENCY_THRESHOLD_BYTES', 'SMALL_CARD_DEFAULT_CONCURRENCY',
        'LARGE_CARD_DEFAULT_CONCURRENCY'))
    if not (0 < threshold < 2**64 and 0 < small <= large <= 0xffffffff):
        raise ValueError('invalid capacity registry geometry')
    physical = Decimal(sys.argv[2].strip())
    budget = Decimal(sys.argv[3]) * 2**30 if sys.argv[3] else physical
    if not all(value.is_finite() and 1 <= value < 2**64 for value in (physical, budget)):
        raise ValueError('GPU capacity must be finite, positive and representable in bytes')
    logical = min(physical, budget)
    print(logical / 2**30, small if int(logical) <= threshold else large)
except (OSError, ValueError, ArithmeticError) as error:
    sys.exit(f'coordinator capacity registry: {error}')
PYPROFILE
}

# The worker and coordinator must admit the same live rows, including decoder
# replay. Fail closed if the Rust registry stops being literal declarations.
release_v41_expert_capacity() {
  local registry="${repo_root:?}/rust/crates/cuteafd-core/src/coordinator_programs.rs"
  python3 - "$registry" "$@" <<'PYROWS'
import pathlib, re, sys
source = pathlib.Path(sys.argv[1]).read_text()
def literal(name, shape):
    declarations = re.findall(r'^pub const ' + name + r':.*$', source, re.M)
    if len(declarations) != 1:
        raise ValueError(f'missing or duplicate row registry constant {name}')
    match = re.fullmatch(r'pub const ' + name + shape, declarations[0])
    if match is None:
        raise ValueError(f'nonliteral row registry constant {name}')
    return match.group(1)
try:
    names = ('PREFILL_MIN_ROWS', 'PREFILL_MAX_ROWS', 'LIVE_MIN_ROWS', 'DECODER_REPLAY_ROWS')
    low, high, floor, replay = (int(literal('V41_' + name, r': u32 = ([0-9]+);')) for name in names)
    capacities = [int(x) for x in literal('V41_SERVING_AOT_ROWS',
        r': \[u32; 3\] = \[([0-9]+, [0-9]+, [0-9]+)\];').split(', ')]
    if not (0 < low <= floor <= high and 0 < replay <= high
            and capacities == sorted(set(capacities)) and capacities[-1] >= high
            and all(0 < x <= 0xffffffff for x in [low, high, floor, replay, *capacities])):
        raise ValueError('invalid row registry geometry')
    if not sys.argv[2:]:
        raise ValueError('missing prefill chunk')
    for text in sys.argv[2:]:
        if not re.fullmatch(r'[0-9]+', text) or not low <= int(text) <= high:
            raise ValueError(f'prefill chunk must be in {low}..{high}: {text}')
        required = max(int(text), floor, replay)
        print(next(rows for rows in capacities if rows >= required))
except (ValueError, StopIteration) as error:
    sys.exit(f'V4.1 row registry: {error}')
PYROWS
}

release_validate_wip_instance() {
  [[ -z "${WIP_INSTANCE:-}" || "$WIP_INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] ||
    release_die "WIP_INSTANCE must be [A-Za-z0-9_.-], starting with a letter or digit (max 41 characters)"
}

release_validate_wip_root() {
  [[ -n "${WIP_ROOT:-}" ]] || return 0
  release_validate_path_setting WIP_ROOT "$WIP_ROOT"
  local parent="$HOME/.cache/cuteafd/builds"
  [[ "$WIP_ROOT" == "$parent/"* ]] || release_die "WIP_ROOT must be under $parent/"
  [[ "$(realpath -m "$WIP_ROOT")" == "$(realpath -m "$parent")/"* ]] ||
    release_die "WIP_ROOT resolves outside $parent/"
}

release_wip_container() {
  release_validate_wip_instance
  case "$1" in coordinator|spark-expert) ;; *) release_die "invalid WIP container role: $1" ;; esac
  printf 'cuteafd-%s-wip%s' "$1" "${WIP_INSTANCE:+-$WIP_INSTANCE}"
}

# The host slot index finds the producing instance even when the launch config
# omits it. Old slots without an index retain the legacy unnamed container.
release_wip_slot_instance() {
  local slot="$1" metadata="$HOME/.cache/cuteafd/wip-slots/$1.json" recorded
  [[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || release_die "invalid WIP slot: $slot"
  release_validate_wip_instance
  if [[ -n "${WIP_INSTANCE:-}" ]]; then
    metadata="$HOME/.cache/cuteafd/wip-slots/$WIP_INSTANCE/$slot.json"
  fi
  if [[ -f "$metadata" ]]; then
    recorded="$(python3 - "$metadata" "$slot" <<'PYMETA'
import json, sys
m = json.load(open(sys.argv[1]))
assert m['slot'] == sys.argv[2] and isinstance(m['wip_instance'], str)
print(m['wip_instance'])
PYMETA
)" || release_die "invalid WIP slot metadata: $metadata"
    [[ -z "${WIP_INSTANCE:-}" || "$WIP_INSTANCE" == "$recorded" ]] ||
      release_die "WIP slot $slot belongs to instance $recorded, not $WIP_INSTANCE"
    WIP_INSTANCE="$recorded"
    release_validate_wip_instance
  fi
  WIP_LAYOUT_SLOT="${WIP_INSTANCE:+$WIP_INSTANCE/}$slot"
}

# One staged run layout per serving instance: a launch re-stages its layout
# (rm + mv), which must never swap files under another live server of the same
# slot. Launchers call this after release_wip_slot_instance.
release_wip_serving_layout() {
  local serving="${1:-}"
  [[ -z "$serving" || "$serving" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || release_die "invalid serving instance: $serving"
  WIP_LAYOUT_SLOT="$WIP_LAYOUT_SLOT/serve-${serving:-default}"
}

release_record_wip_slot() {
  release_validate_wip_instance
  python3 - "$HOME/.cache/cuteafd/wip-slots" "$1" "${WIP_INSTANCE:-}" <<'PYMETA'
import json, os, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(mode='w', dir=root, delete=False) as f:
    json.dump({'slot': sys.argv[2], 'wip_instance': sys.argv[3]}, f)
    name = f.name
os.replace(name, root / (sys.argv[2] + '.json'))
if sys.argv[3]:
    instance_root = root / sys.argv[3]; instance_root.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=instance_root, delete=False) as f:
        json.dump({'slot': sys.argv[2], 'wip_instance': sys.argv[3]}, f)
        name = f.name
    os.replace(name, instance_root / (sys.argv[2] + '.json'))
PYMETA
}

release_need() {
  command -v "$1" >/dev/null 2>&1 || release_die "required command not found: $1"
}

# --------------------------------------------------------------------------- #
# One SSH option set for the scripted calls to a Spark made by the production
# release paths.
#
# A wrong-owner or world-writable drop-in under /etc/ssh/ssh_config.d/ makes
# OpenSSH abort with "Bad owner or permissions" before any host is contacted, and
# an operator cannot fix that by wrapping only their own interactive ssh: the
# internal calls are made by these scripts. CUTEAFD_RELEASE_SSH_CONFIG therefore
# resolves one option set that every site shares. Stock resolution is the default
# because a build or serving host's ~/.ssh/config legitimately carries the host
# aliases and identity files that reach the Sparks; BatchMode is always forced so a
# prompt can never stall an unattended build or a readiness poll.
#
# The contract covers the release pipeline end to end, so that one setting cannot
# be honored on the way in and ignored on the way out:
#   ./build.sh                        build, sync and the remote expert legs
#   ./run.sh                          preflight, launch, readiness, EXIT teardown
#   ./stop.sh, release_stop_*         container teardown
#   ./push-containers.sh              publishing the Spark image from SPARK_0_HOST
# The standalone NOT-LAUNCH-READY harnesses - scripts/launch/run-tp-ep-native-candidate.sh
# and wip.sh - are deliberately outside this contract. They keep
# their own ssh forms (per-host bind addresses, argv rendered into a command string)
# and are not release-pipeline-verified; they join when integrated, not before.
#
# These are declarations only: nothing here changes behavior until a script calls
# release_configure_ssh_transport or release_ssh, so a script that keeps its own
# ssh calls is untouched by sourcing this file.
# --------------------------------------------------------------------------- #

# A setting that reaches a remote shell command string, a Docker bind source or an
# rsync `host:path` spec is accepted only as a canonical absolute path over this
# small conservative alphabet: a whitelist rather than a metacharacter blacklist, so
# a quote, backslash, tilde, colon or brace is refused outright instead of being
# trusted to survive one more layer of quoting. Colon matters twice - it also
# splits an rsync remote spec, so allowing it would let a value name another host.
# Dot segments are rejected separately, because the alphabet allows dots and
# `/a/../b` is a traversal. Values are refused, never rewritten: the path a build
# reports must be the path it mounts.
release_canonical_path='^/[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*$'

release_path_has_no_dot_segment() {
  local segment
  local -a parts
  IFS='/' read -ra parts <<<"$1"
  for segment in ${parts[@]+"${parts[@]}"}; do
    [[ "$segment" != "." && "$segment" != ".." ]] || return 1
  done
  return 0
}

# Validate one setting. NAME is the variable as the operator wrote it. An empty
# value is always acceptable: each caller decides what its own default is.
release_validate_path_setting() {
  local name="$1" value="$2"
  [[ -n "$value" ]] || return 0
  if [[ ! "$value" =~ $release_canonical_path ]] || ! release_path_has_no_dot_segment "$value"; then
    release_die "$name must be a canonical absolute path over letters, digits, dot, underscore, plus and minus - no spaces, dot segments, trailing slashes, . or .. segments or shell metacharacters - got: $value"
  fi
}

# True when CHILD is `path` itself or below it, compared a component at a time.
release_path_within() {
  local child="$1" parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}

# Declared empty up front so a `set -u` caller can read them while building its own
# rsync or docker arguments; they only carry real values after
# release_configure_ssh_transport has run. Sourcing this file still changes no
# behavior: nothing is resolved and nothing is exported here.
release_ssh_config=''
release_rsh=''
release_ssh_opts=()
_release_ssh_transport_configured=''

# Resolve the transport once per process. Idempotent, because a script may call it
# at start-up to fail early on a bad value and still be reached through a helper
# that guards itself. The first resolution wins on purpose: switching option sets
# halfway through a build or a readiness poll would send later calls somewhere else.
release_configure_ssh_transport() {
  [[ -z "$_release_ssh_transport_configured" ]] || return 0
  release_ssh_config="${CUTEAFD_RELEASE_SSH_CONFIG-}"
  release_validate_path_setting CUTEAFD_RELEASE_SSH_CONFIG "$release_ssh_config"
  release_ssh_opts=(-o BatchMode=yes)
  if [[ -n "$release_ssh_config" ]]; then
    release_ssh_opts+=(-F "$release_ssh_config")
    release_rsh="ssh -o BatchMode=yes -F $release_ssh_config"
  else
    release_rsh='ssh -o BatchMode=yes'
  fi
  # rsync and rdmasync read this natively, so child scripts inherit the transport
  # without being rewritten. Note that "-F" inside an rsh string is an ssh option:
  # as a bare rsync argument it would mean --filter=dir-merge instead.
  export RSYNC_RSH="$release_rsh"
  _release_ssh_transport_configured=1
}

release_ssh() {
  release_configure_ssh_transport
  # The guarded expansion is deliberate: on bash 4.2 an empty "${array[@]}" aborts
  # under `set -u`, and this function is reached from EXIT traps and readiness polls
  # where dying silently would hide a real failure.
  ssh ${release_ssh_opts[@]+"${release_ssh_opts[@]}"} "$@"
}

# Bulk transfers that touch a host without RoCE (a build seed on 10 GbE only)
# use AES-GCM (hardware-accelerated on x86-64 and GB10), no ssh compression
# (zstd compresses the payload where it pays) and, when ADDR is given, the
# host's directly attached LAN address instead of a slower name route; the
# host key is still checked under HOST's own name.
release_bulk_ssh_opts=(-c aes128-gcm@openssh.com -o Compression=no)
# release_gpu_device_args UUID: the docker run arguments that expose one GPU.
# The one place a container's GPU binding is spelled; it is CDI (see cdi-gpus.sh).
source "$(dirname "${BASH_SOURCE[0]}")/cdi-gpus.sh"
release_gpu_device_args() {
  cuteafd_cdi_gpu_args "$1" || return 1
  printf '%s\n' "${CUTEAFD_CDI_ARGS[@]}"
}
# LAN-only build seeds: GB10 hosts that compile, export and seal Spark slots but
# never serve. They sit off the RoCE fabric (10 GbE), so bulk transfers to them go
# over their LAN address and their sealed slots reach the Sparks through sparknest.
# Keep in sync with scripts/build/build-resource.py LAN_SEEDS, scripts/hw/hwlock.py
# ORDER and scripts/bench/wip-cards.py SEED_HOSTS (a script test checks).
release_lan_seed_hosts=(kakapo cassowary)
release_is_lan_seed() {
  [[ " ${release_lan_seed_hosts[*]} " == *" $1 "* ]]
}
# release_host_has_rdma HOST: HOST is on the RoCE fabric: at least one RDMA port
# is ACTIVE. A LAN-only build seed never counts, even with its adapters present
# (cassowary's are down) or a private seed-to-seed link up.
release_host_has_rdma() {
  ! release_is_lan_seed "$1" || return 1
  release_ssh "$1" 'grep -qsx "4: ACTIVE" /sys/class/infiniband/*/ports/*/state'
}
# release_bulk_ssh HOST ADDR ARGS...: one bulk ssh leg (ADDR may be empty).
release_bulk_ssh() {
  local host="$1" addr="$2"
  shift 2
  local -a route=()
  [[ -z "$addr" || "$addr" == "$host" ]] || route=(-o "HostName=$addr" -o "HostKeyAlias=$host")
  release_ssh "${release_bulk_ssh_opts[@]}" ${route[@]+"${route[@]}"} "$host" "$@"
}

# Inference images verify package payload hashes when built. Compare their
# immutable manifests before changing services so all peers select one layout.
release_exl3_package_identity() {
  local revision="$1" manifest layout digest
  manifest="$(cat)"
  layout="$(jq -er --arg revision "$revision" '
    if .schema == "cuteafd.exl3-package.v1" and .role == "spark"
      and .sparkinfer_revision == $revision
    then "disjoint"
    else error("invalid Spark EXL3 package identity") end
  ' <<<"$manifest")" || release_die "invalid Spark EXL3 package identity"
  digest="$(printf '%s' "$manifest" | sha256sum | awk '{print $1}')"
  printf '%s:%s\n' "$layout" "$digest"
}

# Compact Spark EXL3 package admission. Every physical rank r needs its
# tp<degree>-rank<r>/m<capacity> package for every worker capacity up to the
# launch capacity, with the exact shard width 2304/degree (1152 at TP2, 768 at
# TP3). This runs in the per-host preflight loop, so a missing package fails
# before any service change. TP2 is the default degree.
release_validate_exl3_compact_variants() {
  local capacity="$1" family="$2" tp="${3:-2}"
  [[ "$tp" == 2 || "$tp" == 3 || "$tp" == 6 ]] || release_die "EXL3 variant admission supports TP2, TP3 or TP6, got TP$tp"
  local width=$((2304 / tp))
  [[ "$family" =~ ^k([23])([34])$ && ( "$family" == k23 || "$family" == k34 ) ]] || release_die "cannot resolve compact EXL3 package bit tiers"
  local low="${BASH_REMATCH[1]}" high="${BASH_REMATCH[2]}"
  jq -e --argjson capacity "$capacity" --argjson low "$low" --argjson high "$high" --argjson tp "$tp" --argjson width "$width" '
    . as $manifest | .compute == [12,1] and
    all([1,16,80,256,1024,4096][] | select(. <= $capacity); . as $m |
      all(range(0; $tp); . as $rank |
        [$manifest.variants[]? |
          select(.directory == ("tp" + ($tp|tostring) + "-rank" + ($rank|tostring) + "/m" + ($m|tostring))) |
          select(.capacity == $m and .intermediate == $width and .experts == 384
                 and .top_k == 6 and .output_dtype == "bf16"
                 and .bits == [$low,$high])] | length == 1))
  ' >/dev/null || release_die "Spark EXL3 package lacks required TP$tp rank/capacity/shape/bit variants"
}

# Historical name: the TP2-degree call keeps working for older harnesses.
release_validate_exl3_tp2_variants() {
  release_validate_exl3_compact_variants "$1" "$2" 2
}

release_model_list_matches() {
  local model_id="$1"
  local full_model_id="${model_id}-full"
  jq -e \
    --arg model_id "$model_id" \
    --arg full_model_id "$full_model_id" \
    '.object == "list"
      and (.data | type == "array")
      and ([.data[]? | select(.id == $model_id)] | length == 1)
      and ([.data[]? | select(.id == $full_model_id)] | length == 1)' \
    >/dev/null 2>&1
}

# Provision a reusable owned 0600 secret without putting its value in argv.
release_prepare_secret() {
  python3 - "$1" "$2" "${3:-reuse}" <<'PYSECRET'
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
root = Path(sys.argv[1])
root.mkdir(mode=0o700, parents=True, exist_ok=True)
if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o022:
    raise SystemExit('secret directory must be owned by this user and not writable by others')
if sys.argv[2] == 'secret' and stat.S_IMODE(root.stat().st_mode) != 0o700:
    raise SystemExit('console secret directory must have mode 0700')
path = root / sys.argv[2]
if path.exists() or path.is_symlink():
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit('secret must be an owned regular file with mode 0600')
        key = stream.read().rstrip('\r\n')
        if not key or any(not 33 <= ord(c) <= 126 for c in key):
            raise SystemExit('secret file is invalid')
        if sys.argv[2] == 'secret' and (len(key) != 64 or any(c not in '0123456789abcdef' for c in key)):
            raise SystemExit('console secret must be 32 random bytes encoded as hex')
fd, temporary = tempfile.mkstemp(prefix='.' + sys.argv[2] + '-', dir=root)
try:
    with os.fdopen(fd, 'w') as stream:
        stream.write(secrets.token_hex(32) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if sys.argv[3] == 'rotate':
        # Preserve the inode: a running container's read-only bind mount sees rotation.
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
        with os.fdopen(fd, 'w') as stream:
            stream.write(Path(temporary).read_text())
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
    else:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
finally:
    os.unlink(temporary)
print(path)
PYSECRET
}
release_console_supported() {
  local image="$1" wip_binary="${2:-}"
  if [[ -n "$wip_binary" ]]; then
    docker run --rm --network none --mount "type=bind,src=$wip_binary,dst=/opt/cuteafd-console-probe,readonly" \
      --entrypoint /opt/cuteafd-console-probe "$image" serve-native --help 2>/dev/null | grep -q -- '--console-secret-file'
  else
    [[ "$(docker image inspect --format '{{index .Config.Labels "org.cuteafd.console-gate"}}' "$image" 2>/dev/null)" == 1 ]]
  fi
}
release_prepare_console() {
  CONSOLE_SECRET_FILE="$(release_prepare_secret "$HOME/.cache/cuteafd/console" secret)" || release_die "could not provision console secret"
  USAGE_DIR="$HOME/.cache/cuteafd/instances/${1:-default}/usage"
  mkdir -p "$USAGE_DIR"
  chmod 700 "$USAGE_DIR"
}
release_print_console_link() {
  # The only intentional disclosure is an interactive terminal, never a
  # redirected log (bench harnesses and agents capture launcher stdout).
  local base="${CONSOLE_URL:-$1}"
  if [[ -t 1 || "${CUTEAFD_PRINT_CONSOLE_LINK:-}" == 1 ]]; then
    printf 'console unlock: %s/console/unlock?token=%s\n' "${base%/}" "$(tr -d '\r\n' <"$CONSOLE_SECRET_FILE")"
  else
    printf 'console unlock: %s/console/unlock?token=<contents of %s>\n' "${base%/}" "$CONSOLE_SECRET_FILE"
  fi
}

# The shell line that prints this host's console unlock link, for pages and logs
# (never the secret itself): run it on the host that launched the server.
release_console_link_command() {
  local base="${CONSOLE_URL:-$1}" file="${CONSOLE_SECRET_FILE/#$HOME/\~}"
  printf 'echo "%s/console/unlock?token=$(cat %s)"' "${base%/}" "$file"
}

# API keys are opt-in (API_KEY_FILE); the benchmark runs on the console cookie.
release_prepare_api_key() {
  local enabled="${1:-off}" instance="${2:-default}"
  case "$enabled" in
    on|off) ;;
    *) release_die "ENABLE_BENCH must be on or off" ;;
  esac
  if [[ -n "${API_KEY_FILE:-}" ]]; then
    [[ -f "$API_KEY_FILE" && -r "$API_KEY_FILE" ]] || release_die "API_KEY_FILE must name a readable file"
  fi
}

release_api_curl() {
  if [[ -n "${API_KEY_FILE:-}" ]]; then
    # Keep the key out of argv/process listings. Health probes do not need it.
    printf 'Authorization: Bearer %s\n' "$(tr -d '\r\n' <"$API_KEY_FILE")" |
      curl --header @- "$@"
  else
    curl "$@"
  fi
}

release_api_advertises_model() {
  local url="$1"
  local model_id="$2"
  release_api_curl -fsS "$url/v1/models" 2>/dev/null |
    release_model_list_matches "$model_id"
}

release_native_model_list_matches() {
  local model_id="$1"
  jq -e \
    --arg model_id "$model_id" \
    '.object == "list"
      and (.data | type == "array")
      and ([.data[]? | select(.id == $model_id)] | length == 1)' \
    >/dev/null 2>&1
}

release_api_advertises_native_model() {
  local url="$1"
  local model_id="$2"
  release_api_curl -fsS "$url/v1/models" 2>/dev/null |
    release_native_model_list_matches "$model_id"
}

release_validate_model_list_file() {
  local path="$1"
  local model_id="$2"
  release_model_list_matches "$model_id" <"$path" ||
    release_die "API model list does not advertise exact configured identities: $model_id and ${model_id}-full"
}

release_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

# Evaluate on the host that starts the container: Spark and coordinator UIDs
# can differ. Keep caches writable and checkpoint mounts outside root's home.
# Self-contained so remote launchers can send this function to the worker host.
release_serving_docker_args() {
  local user group primary_group
  local -a groups=()
  user="$(id -un)"
  primary_group="$(id -g)"
  read -ra groups <<<"$(id -G)"
  RELEASE_SERVING_ARGS=(--user "$(id -u):$(id -g)"
    -e "USER=$user" -e "LOGNAME=$user" -e HOME=/tmp/cuteafd-home
    -e HF_HOME=/hf -e CUTEAFD_BENCH_DIR=/run/cuteafd/bench
    -e TORCH_EXTENSIONS_DIR=/tmp/cuteafd-home/torch-extensions
    -e XDG_CACHE_HOME=/tmp/cuteafd-home/.cache
    -e TRITON_CACHE_DIR=/tmp/cuteafd-home/triton
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/cuteafd-home/torchinductor)
  # ADR-052's dedicated directory survives socket replacement on daemon restart.
  # Never create it here: absent storage stays unknown without root-owned state.
  if [[ -d /srv/sparknest/run ]]; then
    RELEASE_SERVING_ARGS+=(--mount type=bind,src=/srv/sparknest/run,dst=/srv/sparknest/run,readonly)
  fi
  # Docker's numeric --user otherwise drops groups used by host access checks.
  for group in "${groups[@]}"; do
    [[ "$group" == "$primary_group" ]] || RELEASE_SERVING_ARGS+=(--group-add "$group")
  done
}

# Shared host stores mounted into the coordinator must be ours: a missing bind
# source is recreated by Docker as root, which a non-root server cannot write.
release_require_owned_state() {
  local path owner
  for path in "$@"; do
    [[ -e "$path" || -L "$path" ]] || continue
    owner="$(stat -c %u "$path")"
    [[ "$owner" == "$(id -u)" && ! -L "$path" ]] || release_die "$path is owned by uid $owner, not $(id -u) (Docker recreates a deleted bind-mount source as root); fix with: sudo chown -R $(id -u):$(id -g) $path"
  done
}

# Fail before serving when legacy root-owned outputs cannot be updated. Check
# directory descendants, or only a --file target and its parent (not siblings),
# without changing modes/owners. Shipped to worker shells; no repo dependencies.
release_require_writable_state() {
  python3 - "$@" <<'PYSTATE'
import os
from pathlib import Path
import sys
values = sys.argv[1:]
file_only = bool(values and values[0] == '--file')
if file_only:
    values = values[1:]
for value in values:
    root = Path(value)
    paths = [root, root.parent] if file_only else [root]
    if root.is_dir() and not file_only:
        def denied(error):
            raise SystemExit(f'serving state is not writable by uid {os.getuid()}: {error.filename}; migrate legacy ownership before launching')
        for directory, dirs, files in os.walk(root, onerror=denied):
            paths.extend(Path(directory) / name for name in dirs + files)
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        mode = os.W_OK | (os.X_OK if path.is_dir() else 0)
        if not os.access(path, mode):
            raise SystemExit(f'serving state is not writable by uid {os.getuid()}: {path}; migrate legacy ownership before launching')
PYSTATE
}

# PROFILE / PROFILE_DIR / PROFILE_NCU_*: the switches are documented once, at
# the PROFILE comment in scripts/launch/run-family.sh.
# release_profile_docker_args sets RELEASE_PROFILE_ARGS (docker environment and
# mount arguments, for the coordinator and every Spark worker) in the caller's
# shell, so a bad value stops the launch. Arguments: PROFILE PROFILE_DIR HOST
# [PROFILE_SPARK].
release_profile_switches='census transport-timing expert-timing expert-routes draft-timing nsys ncu'
# nsys sessions are per container; the window is opened and closed with
# `docker exec <coordinator> nsys start|stop --session=$RELEASE_NSYS_SESSION`.
RELEASE_NSYS_SESSION=cuteafd-profile
release_profile_docker_args() {
  local profile="$1" dir="$2" host="$3" spark="${4-}" switch
  local -a switches=()
  RELEASE_PROFILE_ARGS=()
  if [[ -z "$profile" && -z "$spark" ]]; then
    [[ -z "$dir" ]] || release_die "PROFILE_DIR needs PROFILE or PROFILE_SPARK"
    return 0
  fi
  if [[ -z "$profile" ]]; then
    # PROFILE_SPARK alone: every container still gets the host name and the
    # shared PROFILE_DIR mount (release_profile_spark_wrapper checks the rest).
    [[ "$dir" =~ ^/[A-Za-z0-9._/-]+$ ]] || release_die "PROFILE_SPARK=$spark needs an absolute PROFILE_DIR (the report goes there)"
    [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || release_die "invalid profile host: $host"
    RELEASE_PROFILE_ARGS+=(-e "CUTEAFD_PROFILE_HOST=$host" -v "$dir:$dir" -e "CUTEAFD_PROFILE_DIR=$dir")
    return 0
  fi
  [[ "$profile" =~ ^[a-z-]+(,[a-z-]+)*$ ]] ||
    release_die "PROFILE must be a comma list of: ${release_profile_switches// /, }"
  IFS=, read -ra switches <<<"$profile"
  for switch in "${switches[@]}"; do
    case "$switch" in
      census) RELEASE_PROFILE_ARGS+=(-e CUTEAFD_GRAPH_CENSUS=1 -e 'RUST_LOG=info,cuteafd::graph_capture=debug'
        -e CUTEAFD_STARTUP_MEMORY_PROBE=1) ;;
      transport-timing) RELEASE_PROFILE_ARGS+=(-e CUTEAFD_PROTOCOL_V2_TCP_TIMING=1) ;;
      expert-timing) RELEASE_PROFILE_ARGS+=(-e CUTEAFD_EXPERT_TIMING=1) ;;
      expert-routes) [[ -n "$dir" ]] || release_die "PROFILE=expert-routes needs PROFILE_DIR (the route log goes there)"
        RELEASE_PROFILE_ARGS+=(-e CUTEAFD_EXPERT_ROUTES=1) ;;
      draft-timing) RELEASE_PROFILE_ARGS+=(-e CUTEAFD_DRAFT_TIMING=1) ;;
      # Process wrappers (release_profile_wrapper); Spark workers take PROFILE_SPARK.
      nsys|ncu) [[ -n "$dir" ]] || release_die "PROFILE=$switch needs PROFILE_DIR (the report goes there)" ;;
      *) release_die "unknown PROFILE switch $switch (known: ${release_profile_switches// /, })" ;;
    esac
  done
  [[ ",$profile," != *,nsys,* || ",$profile," != *,ncu,* ]] || release_die "PROFILE takes nsys or ncu, not both"
  [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || release_die "invalid profile host: $host"
  RELEASE_PROFILE_ARGS+=(-e "CUTEAFD_PROFILE_HOST=$host")
  if [[ -n "$dir" ]]; then
    [[ "$dir" =~ ^/[A-Za-z0-9._/-]+$ ]] || release_die "PROFILE_DIR must be an absolute directory"
    RELEASE_PROFILE_ARGS+=(-v "$dir:$dir" -e "CUTEAFD_PROFILE_DIR=$dir")
  fi
}

# With PROFILE=expert-routes, coordinator only: PROFILE_EXPERT_INPUT_SAMPLE=N
# saves the MoE input rows of the first N serving layer-steps (default 64, 0 =
# none); PROFILE_EXPERT_ROUTE_WEIGHTS=on also logs the gate weights of each
# narrow wave (off by default). Sets RELEASE_PROFILE_SAMPLE_ARGS (coordinator
# docker environment). Arguments: PROFILE SAMPLE WEIGHTS.
release_profile_sample_args() {
  local profile="$1" sample="$2" weights="${3-}"
  RELEASE_PROFILE_SAMPLE_ARGS=()
  if [[ -n "$sample" ]]; then
    [[ ",$profile," == *,expert-routes,* ]] || release_die "PROFILE_EXPERT_INPUT_SAMPLE needs PROFILE=expert-routes"
    [[ "$sample" =~ ^(0|[1-9][0-9]{0,5})$ ]] || release_die "PROFILE_EXPERT_INPUT_SAMPLE must be 0..999999 layer-steps"
    RELEASE_PROFILE_SAMPLE_ARGS+=(-e "CUTEAFD_EXPERT_INPUT_SAMPLE=$sample")
  fi
  case "$weights" in
    ""|off) ;;
    on) [[ ",$profile," == *,expert-routes,* ]] || release_die "PROFILE_EXPERT_ROUTE_WEIGHTS needs PROFILE=expert-routes"
      RELEASE_PROFILE_SAMPLE_ARGS+=(-e CUTEAFD_EXPERT_ROUTES_WEIGHTS=1) ;;
    *) release_die "PROFILE_EXPERT_ROUTE_WEIGHTS must be on or off" ;;
  esac
}

# Print the profiler to exec. The container calls this with the tool name only,
# so the search is the image's own files. Tests pass a fixture root as $2.
# ncu: /usr/local/bin/ncu, else the newest /opt/nvidia/nsight-compute/*/ncu.
# nsys: /usr/local/bin/nsys, else /usr/local/cuda/bin/nsys, else the newest
# /opt/nvidia/nsight-systems/*/nsys. Version names are dotted numbers, so
# 2026.1.10 beats 2026.1.9.
release_profile_resolve_tool() {
  local tool="$1" root="${2-}" candidate version best="" best_version=""
  local restore_nullglob=0
  local -a stable=() found=()
  case "$tool" in
    ncu) stable=("$root/usr/local/bin/ncu") ;;
    nsys) stable=("$root/usr/local/bin/nsys" "$root/usr/local/cuda/bin/nsys") ;;
    *) printf 'unknown profiler %s\n' "$tool" >&2; return 127 ;;
  esac
  for candidate in "${stable[@]}"; do
    if [[ -f "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  shopt -q nullglob && restore_nullglob=1
  shopt -s nullglob
  if [[ "$tool" == ncu ]]; then
    found=("$root"/opt/nvidia/nsight-compute/*/ncu)
  else
    found=("$root"/opt/nvidia/nsight-systems/*/nsys)
  fi
  ((restore_nullglob)) || shopt -u nullglob
  for candidate in "${found[@]}"; do
    [[ -f "$candidate" && -x "$candidate" ]] || continue
    version="$(basename "$(dirname "$candidate")")"
    [[ "$version" =~ ^[0-9]+([.][0-9]+)+$ ]] || continue
    if [[ -z "$best" ]]; then
      best="$candidate"
      best_version="$version"
      continue
    fi
    [[ "$version" == "$best_version" ]] && continue
    if [[ "$(printf '%s\n%s\n' "$best_version" "$version" | LC_ALL=C sort -V | tail -n 1)" == "$version" ]]; then
      best="$candidate"
      best_version="$version"
    fi
  done
  if [[ -n "$best" ]]; then
    printf '%s\n' "$best"
    return 0
  fi
  if [[ "$tool" == ncu ]]; then
    printf 'ncu not found: looked for /usr/local/bin/ncu and /opt/nvidia/nsight-compute/*/ncu\n' >&2
  else
    printf 'nsys not found: looked for /usr/local/bin/nsys, /usr/local/cuda/bin/nsys, and /opt/nvidia/nsight-systems/*/nsys\n' >&2
  fi
  return 127
}

# TOOL, then the profiler's arguments, then the served command. exec replaces
# this shell so the profiler receives docker stop's SIGINT.
release_profile_exec() {
  local tool="$1" bin
  shift
  bin="$(release_profile_resolve_tool "$tool")" || exit 127
  exec "$bin" "$@"
}

# Container command: resolve TOOL at start, then exec it with the remaining
# arguments. The caller appends the served command after this argv.
release_profile_container_wrapper() {
  local tool="$1" script
  shift
  script="$(declare -f release_profile_resolve_tool; declare -f release_profile_exec)"
  script+=$'\nrelease_profile_exec "$@"'
  RELEASE_PROFILE_WRAPPER=(/bin/bash -c "$script" release-profile-resolve "$tool" "$@")
}

# Sets RELEASE_PROFILE_WRAPPER: the argv that goes between the image and the
# served command (`docker run ... IMAGE "${RELEASE_PROFILE_WRAPPER[@]}" cuteafd
# ...`), and RELEASE_PROFILE_WRAPPER_DOCKER_ARGS (docker arguments before the
# image: the profiler replaces the wrapper shell before serving, so `docker stop`
# sends it SIGINT and waits up to 300 s for the report; the cuteafd.graceful-stop
# label makes smoke teardown, release_stop_local_container and
# release_stop_remote_containers use that stop before removing), both empty
# unless PROFILE names nsys or ncu. This is the coordinator's wrapper: a Spark
# role fails closed here (Spark workers take PROFILE_SPARK through
# release_profile_spark_wrapper, and the environment switches from
# release_profile_docker_args). Call after that helper, which has validated
# PROFILE and PROFILE_DIR.
# The profiler binary is not chosen here. The host install and the image install
# differ (the dev image has ncu at /opt/nvidia/nsight-compute/<version>/ncu, which
# may or may not be linked from /usr/local/bin/ncu). The wrapper command resolves
# it when the container starts.
# PROFILE_NSYS_GPU_METRICS adds nsys GPU metrics sampling (clocks, power, DRAM
# and SM activity): `all` samples every GPU the container sees (use this), or an
# index list in nsys's own device order, which is not CUDA's or nvidia-smi's
# (`nsys profile --gpu-metrics-devices=help` lists it).
# PROFILE_NCU_SKIP=N skips the first N launches that match PROFILE_NCU_KERNEL
# (`ncu --launch-skip`), so the replayed ones are past readiness.
# Arguments: PROFILE PROFILE_DIR HOST ROLE NCU_KERNEL NCU_COUNT NCU_SET NSYS_GPU_METRICS NCU_SKIP.
release_profile_wrapper() {
  local profile="$1" dir="$2" host="$3" role="$4" kernel="${5:-}" count="${6:-}" set="${7:-}" metrics="${8:-}" skip="${9:-}"
  RELEASE_PROFILE_WRAPPER=() RELEASE_PROFILE_WRAPPER_DOCKER_ARGS=()
  local wrapped=
  [[ ",$profile," == *,nsys,* ]] && wrapped=nsys
  [[ ",$profile," == *,ncu,* ]] && wrapped=ncu
  if [[ -z "$wrapped" ]]; then
    [[ -z "$kernel$count$set$skip" ]] || release_die "PROFILE_NCU_KERNEL/COUNT/SET/SKIP need PROFILE=ncu"
    [[ -z "$metrics" ]] || release_die "PROFILE_NSYS_GPU_METRICS needs PROFILE=nsys"
    return 0
  fi
  [[ "$role" == coordinator ]] ||
    release_die "PROFILE=$wrapped wraps the coordinator only; a Spark worker takes PROFILE_SPARK (role $role)"
  [[ "$dir" =~ ^/[A-Za-z0-9._/-]+$ ]] || release_die "PROFILE=$wrapped needs an absolute PROFILE_DIR"
  [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || release_die "invalid profile host: $host"
  _release_profile_tool_wrapper PROFILE "$wrapped" "$dir/$wrapped-$host" "$kernel" "$count" "$set" "$skip" "$metrics"
}

# Builds RELEASE_PROFILE_WRAPPER(_DOCKER_ARGS) for one container. KEY is the
# setting family the messages name (PROFILE or PROFILE_SPARK); OUTPUT is the
# report path without its suffix. Arguments: KEY TOOL OUTPUT NCU_KERNEL NCU_COUNT
# NCU_SET NCU_SKIP NSYS_GPU_METRICS.
_release_profile_tool_wrapper() {
  local key="$1" tool="$2" output="$3" kernel="$4" count="$5" set="$6" skip="$7" metrics="$8"
  RELEASE_PROFILE_WRAPPER_DOCKER_ARGS=(--stop-signal SIGINT --stop-timeout 300 --label cuteafd.graceful-stop=1)
  if [[ "$tool" == nsys ]]; then
    [[ -z "$kernel$count$set$skip" ]] || release_die "${key}_NCU_KERNEL/COUNT/SET/SKIP need $key=ncu"
    # Collection starts paused (nsys start/stop through docker exec open and
    # close the window); stopping the session or the container writes the report.
    release_profile_container_wrapper nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none --start-later=true "--session-new=$RELEASE_NSYS_SESSION" \
      "--output=$output" --force-overwrite=true
    if [[ -n "$metrics" ]]; then
      [[ "$metrics" == all || "$metrics" =~ ^[0-9]+(,[0-9]+)*$ ]] ||
        release_die "${key}_NSYS_GPU_METRICS must be all or an nsys GPU index list (e.g. 0 or 0,1)"
      RELEASE_PROFILE_WRAPPER+=("--gpu-metrics-devices=$metrics")
    fi
    return 0
  fi
  [[ -z "$metrics" ]] || release_die "${key}_NSYS_GPU_METRICS needs $key=nsys"
  [[ -n "$kernel" ]] || release_die "$key=ncu needs ${key}_NCU_KERNEL (a kernel-name regex); it replays every matched launch"
  [[ "$kernel" != *[[:space:]]* ]] || release_die "${key}_NCU_KERNEL must not contain whitespace"
  count="${count:-20}" set="${set:-full}" skip="${skip:-0}"
  [[ "$count" =~ ^[1-9][0-9]{0,3}$ ]] || release_die "${key}_NCU_COUNT must be 1..9999"
  [[ "$set" =~ ^[a-z_]+$ ]] || release_die "${key}_NCU_SET must be an ncu section set name (e.g. full, basic)"
  [[ "$skip" =~ ^(0|[1-9][0-9]{0,8})$ ]] || release_die "${key}_NCU_SKIP must be 0..999999999 matched launches"
  local -a skip_args=()
  [[ "$skip" == 0 ]] || skip_args=(--launch-skip "$skip")
  release_profile_container_wrapper ncu --kernel-name "regex:$kernel" ${skip_args[@]+"${skip_args[@]}"} \
    --launch-count "$count" --set "$set" \
    --graph-profiling node --target-processes all --export "$output" --force-overwrite
}

# PROFILE_SPARK=nsys|ncu wraps the expert worker of each Spark rank listed in
# PROFILE_SPARK_RANKS (comma list, default 0: every profiled rank perturbs the
# exchange) the way PROFILE wraps the coordinator, independently of it: same
# in-container binary resolution, host user, no capabilities, graceful-stop
# label. Reports are PROFILE_DIR/<tool>-HOST-rankR.<nsys-rep|ncu-rep> on the
# Spark. nsys starts paused under session cuteafd-profile in the worker container
# (`ssh HOST docker exec WORKER nsys start|stop --session=cuteafd-profile`);
# PROFILE_SPARK_NSYS_GPU_METRICS samples the GB10 (all, or nsys's index list).
# ncu replays PROFILE_SPARK_NCU_COUNT (default 20) launches matching the required
# regex PROFILE_SPARK_NCU_KERNEL after skipping PROFILE_SPARK_NCU_SKIP matching
# launches (default 0), collecting PROFILE_SPARK_NCU_SET (default full); its
# report is written when the worker stops (`docker stop`, never `rm -f`).
# Sets RELEASE_PROFILE_WRAPPER(_DOCKER_ARGS) for RANK, empty when RANK is not
# selected; an empty RANK only validates. Every value is checked on every call,
# so the first call stops a bad launch before any worker starts.
# Arguments: PROFILE_SPARK RANKS PROFILE_DIR HOST RANK WORLD NCU_KERNEL NCU_COUNT
# NCU_SET NCU_SKIP NSYS_GPU_METRICS.
release_profile_spark_wrapper() {
  local tool="$1" ranks="$2" dir="$3" host="$4" rank="$5" world="$6"
  local kernel="${7:-}" count="${8:-}" set="${9:-}" skip="${10:-}" metrics="${11:-}" selected listed
  local -a listed_ranks=()
  local -A seen=()
  RELEASE_PROFILE_WRAPPER=() RELEASE_PROFILE_WRAPPER_DOCKER_ARGS=()
  if [[ -z "$tool" ]]; then
    [[ -z "$ranks$kernel$count$set$skip$metrics" ]] ||
      release_die "PROFILE_SPARK_RANKS/NCU_*/NSYS_GPU_METRICS need PROFILE_SPARK"
    return 0
  fi
  [[ "$tool" == nsys || "$tool" == ncu ]] || release_die "PROFILE_SPARK must be nsys or ncu"
  [[ "$dir" =~ ^/[A-Za-z0-9._/-]+$ ]] || release_die "PROFILE_SPARK=$tool needs an absolute PROFILE_DIR (the report goes there)"
  [[ "$world" =~ ^[0-9]+$ ]] && ((world > 0)) || release_die "PROFILE_SPARK=$tool needs Spark expert workers (SPARK_COUNT=$world)"
  ranks="${ranks:-0}"
  [[ "$ranks" =~ ^[0-9]{1,2}(,[0-9]{1,2})*$ ]] || release_die "PROFILE_SPARK_RANKS must be a comma list of Spark ranks (e.g. 0 or 0,3)"
  IFS=, read -ra listed_ranks <<<"$ranks"
  selected=0
  for listed in "${listed_ranks[@]}"; do
    [[ "$listed" == 0 || "$listed" != 0* ]] || release_die "PROFILE_SPARK_RANKS must be a comma list of Spark ranks (e.g. 0 or 0,3)"
    ((10#$listed < world)) || release_die "PROFILE_SPARK_RANKS names rank $listed; this launch has ranks 0..$((world - 1))"
    [[ -z "${seen[$listed]:-}" ]] || release_die "PROFILE_SPARK_RANKS lists rank $listed twice"
    seen[$listed]=1
    [[ "$listed" != "$rank" ]] || selected=1
  done
  # Validate the tool's own settings even when this rank is not profiled.
  _release_profile_tool_wrapper PROFILE_SPARK "$tool" "$dir/$tool-x" "$kernel" "$count" "$set" "$skip" "$metrics"
  RELEASE_PROFILE_WRAPPER=() RELEASE_PROFILE_WRAPPER_DOCKER_ARGS=()
  ((selected)) || return 0
  [[ "$host" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || release_die "invalid profile host: $host"
  _release_profile_tool_wrapper PROFILE_SPARK "$tool" "$dir/$tool-$host-rank$rank" "$kernel" "$count" "$set" "$skip" "$metrics"
}

# Shared key grammar for release lifecycle scripts and the family launcher.
release_known_key() {
  case "$1" in
    FAMILY) return 0 ;; # Optional build family when checkpoint metadata is not local.
    EXLR8_MANIFEST) return 0 ;; # Published EXLR8 manifest name; builds derive EXLR8 programs from it.
    SPECULATOR|SPECULATOR_MODEL_ID|SPECULATOR_MODEL_REVISION|SPECULATOR_DEPTH|SPECULATOR_DRAFTS|SPECULATOR_FP8|SPECULATION_TRACE) return 0 ;;
    DRAFT_SURVIVAL_LENGTHS|DRAFT_POLICY|DRAFT_ROW_COST) return 0 ;;
    MAPPED_SCALE_PRELOAD_GIB|FULL_PREFILL_LOGITS|PREFIX_CACHE_MARK_MIB|PREFIX_POINT_BOUNDARIES|API_KEY_FILE|ENABLE_BENCH|CONSOLE_ASSETS|USAGE|GATEWAY) return 0 ;;
    GLM5_FLASH_KDA_FP8|GLM5_FLASH_FP8_HEAD|GLM5_FLASH_FP8_PREFILL|GLM5_FLASH_KDA_SPLIT|GLM5_FLASH_PREFILL_LANES|GLM5_FLASH_PREFILL_LANE_ROWS|GLM5_FLASH_HEADROOM_GIB|GLM5_FLASH_GRAPH_BUDGET_MIB|GLM5_FLASH_EXL3_WORKER_PATH|GLM5_FLASH_EXL3_ROUTE_DUMP|GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS|GLM5_FLASH_INDEX_CACHE|GLMF_KDA_FP8|GLMF_FP8_HEAD|GLMF_FP8_PREFILL) return 0 ;;
    MIMO_PREFIX_DRAFT|MIMO_HOST_CACHE|MIMO_PREFILL_CHUNK_S|MIMO_SNAPSHOT_WAIT) return 0 ;;
    MIMO_WEIGHT_POLICY|MIMO_FP8_HEAD|MIMO_FP8_O_PROJ|QWEN_FP8_DECODE|QWEN_FP8_HEAD|QWEN_STARTUP_GRAPHS|POOL_TOKENS|PREFIX_PARTIAL|KV_CACHE|SWA_KV_CACHE|DECODE_GRAPHS|EXPERT_INPUT|DECODE_SHARE|L2_PREFETCH|FP8_SCALES|DRAFT_CONTEXT_SLOTS|DRAFT_SEQUENCES|DRAFT_KV_CACHE|SERVED_MODEL_ID|COORDINATOR_GPUS|COORDINATOR_SPLIT|COORDINATOR_SPLIT_GPU|INSTANCE|WIP_INSTANCE|WIP_ROOT|FP8_EXPERT_PREFILL|SPARK_INTAKE|CONSOLE_TEXT|THERMAL|EXPERT_BACKEND) return 0 ;;
    VISION|VISION_REPLICAS|AUDIO|EMBEDDING|MEDIA_CACHE_BYTES|CHAT_TEMPLATE_FROM|TP2_ATTENTION|TP2_QUERY_PROJECTION|TP2_OUTPUT_PROJECTION|TP2_DSPARK_EXPERTS) return 0 ;;
    GLM5_FLASH_DRAFT_HEAD) return 0 ;;
    GLM5_FLASH_DRAFT_LINEAR) return 0 ;;
    GLM5_FLASH_TARGET_HEAD) return 0 ;;
    TARGET_HEAD) return 0 ;;
    GLM5_FLASH_EXL3_SCHEDULE) return 0 ;;
    RDMA_BOND_BALANCE|HOST_ISA) return 0 ;;
    GLM5_FLASH_KDA_STATE) return 0 ;;
    GLM5_FLASH_PREFIX_MARKS) return 0 ;;
    GLM5_FLASH_REPLAY_RECORDS) return 0 ;;
    GLM5_FLASH_DECODE_ROWS) return 0 ;;
    GLM5_FLASH_PREFILL_BATCH) return 0 ;;
    GLM5_FLASH_ROUTE_RING_CHECK|GLM5_FLASH_REMOTE_DRAFTER) return 0 ;;
    PROBE_DUMP_ROOT|ATTENTION_PLACEMENT|ATTENTION_PLAN_ROWS) return 0 ;;
    DECODE_LANES|PREFILL_ROWS|PREFILL_WAVES|PREFILL_SPAN_ROWS) return 0 ;; # Shared prefill units: rows each, units in flight, rows per continuous span (PLAN "Prefill contract").
    PROFILE|PROFILE_DIR|PROFILE_NCU_KERNEL|PROFILE_NCU_COUNT|PROFILE_NCU_SET|PROFILE_NCU_SKIP|PROFILE_NSYS_GPU_METRICS|PROFILE_EXPERT_INPUT_SAMPLE|PROFILE_EXPERT_ROUTE_WEIGHTS) return 0 ;;
    PROFILE_SPARK|PROFILE_SPARK_RANKS|PROFILE_SPARK_NCU_KERNEL|PROFILE_SPARK_NCU_COUNT|PROFILE_SPARK_NCU_SET|PROFILE_SPARK_NCU_SKIP|PROFILE_SPARK_NSYS_GPU_METRICS) return 0 ;;
    HTTP_QUEUE_DEPTH|HTTP_QUEUE_WAIT_MS|MODEL_ID|MODEL_VARIANT|MODEL_REVISION|EXPERT_FORMAT|DSPARK_DRAFT_POLICY|DSPARK_DRAFT_LIMIT|V41_FP8_HEAD|RTX_GPUS|RTX_EXPERT_LAYERS|RTX_EXPERT_PEER|COORDINATOR_GPU|COORDINATOR_GPU_UUID|COORDINATOR_GPU_PCI_BUS_ID|COORDINATOR_GPU_HEADROOM_GIB|KV_POOL_TOKENS|KV_POOL_SIZE|HOST_CACHE_BYTES|MEMORY_RESERVATION|MAX_CONTEXT_TOKENS|MAX_OUTPUT_TOKENS|CONCURRENCY|PREFIX_CACHE_ENTRIES|PREFILL_BATCH_TOKENS|SPARK_DEVICE_BUDGET_BYTES|SPARK_REDUCTION_MIN_ROWS|SPARKINFER_EXL3|SPARK_COUNT|SPARK_TP|SPARK_EP|ADDR|EXPERT_PORT|SPARK_[0-5]_HOST|SPARK_[0-5]_LANE_A|SPARK_[0-5]_LANE_B|CUTEAFD_EXPORT_HOST|WIP_BUILD_SEED|COORDINATOR_DOCKER_DEV|COORDINATOR_DOCKER_INFERENCE|SPARK_EXPERT_DOCKER_DEV|SPARK_EXPERT_DOCKER_INFERENCE|COORDINATOR_GPU_BUDGET_GIB|SPARK_HOSTS)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Resolve an explicit pool-host order into rank/host/rail rows. Nothing changes
# unless SPARK_HOSTS is set; legacy SPARK_<rank>_* configurations remain verbatim.
release_spark_host_rows() {
  local csv="$1" count="$2" mode="${3:-launch}" host index rank=0 seen=,
  local key configured_host lane_a lane_b
  local -a hosts
  [[ "$csv" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*(,[A-Za-z0-9][A-Za-z0-9._-]*)*$ ]] ||
    release_die "SPARK_HOSTS must be a comma-separated Spark pool host list"
  IFS=, read -r -a hosts <<<"$csv"
  if [[ "$mode" == launch ]]; then
    [[ "$count" =~ ^[0-6]$ && "${#hosts[@]}" == "$count" ]] ||
      release_die "SPARK_HOSTS must name exactly SPARK_COUNT=$count hosts"
  fi
  for host in "${hosts[@]}"; do
    [[ "$seen" != *",$host,"* ]] || release_die "SPARK_HOSTS contains duplicate host $host"
    seen+="$host,"
    case "$host" in
      ostrich) index=1 ;; dodo) index=2 ;; emu) index=3 ;;
      kiwi) index=4 ;; rhea) index=5 ;; moa) index=6 ;;
      *)
        lane_a= lane_b=
        for index in {0..5}; do
          key="SPARK_${index}_HOST"
          configured_host="${!key:-}"
          if [[ "$configured_host" == "$host" ]]; then
            key="SPARK_${index}_LANE_A"; lane_a="${!key:-}"
            key="SPARK_${index}_LANE_B"; lane_b="${!key:-}"
            break
          fi
        done
        [[ -n "$lane_a" ]] ||
          release_die "SPARK_HOSTS names unknown Spark pool host $host; configure SPARK_<rank>_HOST and LANE_A for this custom host"
        printf '%s %s %s %s\n' "$rank" "$host" "$lane_a" "$lane_b"
        rank=$((rank + 1))
        continue ;;
    esac
    printf '%s %s 10.55.0.%s 10.55.1.%s\n' "$rank" "$host" "$index" "$index"
    rank=$((rank + 1))
  done
}

# Optional logical per-GPU ceiling. Validate identically in both launchers
# before any worker/service changes; an empty key keeps physical admission.
release_validate_coordinator_gpu_budget() {
  local value="$1"
  [[ -z "$value" ]] && return 0
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] &&
    python3 -c 'import math,sys; n=float(sys.argv[1])*2**30; sys.exit(not (math.isfinite(n) and 1 <= n < 2**64))' "$value" ||
    release_die "COORDINATOR_GPU_BUDGET_GIB must be finite, positive GiB representable in bytes"
}

# Load and validate a configuration file.
#
# mode defaults to "launch" and keeps the full strict launch contract,
# including serving topology, GPU layout, model-format and reduction-rail
# requirements. stop.sh passes "stop": cleanup only needs the Spark host keys
# and the port/address values, so parsing, known-key, quoting and every
# value-safety check stay in force while launch-only topology/readiness
# requirements are skipped. A launch-incomplete but syntactically valid file
# can therefore clean every host it names.
# The family of the checkpoint a config names (scripts/lib/checkpoint-family.py
# on its config.json). A config without MODEL_ID, or a snapshot not present
# here, is DeepSeek V4.1, the release default; its own validation reports a
# missing snapshot.
release_config_family() {
  local config="$1" model="" revision="" line key value hub dir
  [[ -f "$config" ]] || { printf 'deepseek_v41\n'; return 0; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(release_trim "${line%%#*}")"
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in
      MODEL_ID) model="$value" ;;
      MODEL_REVISION) revision="$value" ;;
    esac
  done < "$config"
  [[ -n "$model" ]] || { printf 'deepseek_v41\n'; return 0; }
  hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  dir="$hub/models--${model//\//--}"
  [[ -n "$revision" || ! -f "$dir/refs/main" ]] || revision="$(<"$dir/refs/main")"
  local described=""
  if [[ -n "$revision" && -f "$dir/snapshots/$revision/config.json" ]]; then
    described="$(python3 "$(dirname "${BASH_SOURCE[0]}")/checkpoint-family.py" \
      "$dir/snapshots/$revision/config.json" 2>/dev/null)" || described=""
  fi
  # Unrecognized configs stay on the DeepSeek V4.1 path, whose checks name the problem.
  [[ -n "$described" ]] || described=deepseek_v41
  printf '%s\n' "${described%% *}"
}

# Resolve launcher-only GLM Flash defaults; explicit old values keep the old argv.
release_glmf_auto() {
  local key="$1" value="$2" preferred="$3" fallback="$4" eligible="$5" reason="$6"
  if [[ "$value" == auto ]]; then
    value="$fallback"
    [[ "$eligible" != 1 ]] || value="$preferred"
    printf '%s=auto -> %s (%s)\n' "$key" "$value" "$reason" >&2
  fi
  printf '%s' "$value"
}

# Inspect the selected image's program metadata without exposing GPUs. A missing
# manifest or incomplete set must keep the old 64-row path, not break auto.
release_glmf_wide_decode_available() {
  local snapshot="$1" index="$2" state="$3" kda="$4" image="$5"
  shift 5
  local geometry
  geometry="$(python3 - "$snapshot/config.json" <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
    c = c.get("text_config", c)
    print(c["moe_intermediate_size"], c["intermediate_size"])
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
PY
)" || return 1
  docker run --rm --network none "$@" --entrypoint python3 "$image" -c '
import json, sys
try:
    programs = {p["name"] for p in json.load(open("/opt/cuteafd/share/PROGRAMS.json"))["programs"]}
    moe, dense = sys.argv[4].split()
    names = ["mhc_post_pre", "mla_producer", "o", "sparse_mla_decode",
             "index_producer", "index_topk_decode", "ffn_i"+moe, "ffn_i"+dense]
    names += ["kda" if sys.argv[2] == "f32" else "kda_s16"]
    if sys.argv[3] != "off": names += ["kda_w8"]
    compact = sys.argv[1] == "compact"
    if compact: names += ["index_producer_c"]
    names += ["kda_commit" + ("_c" if compact else "") + ("_s16" if sys.argv[2] != "f32" else "")]
    sys.exit(0 if all("glmf_"+n+"_m128" in programs for n in names) else 1)
except (OSError, ValueError, KeyError, TypeError):
    sys.exit(1)
' "$index" "$state" "$kda" "$geometry" >/dev/null 2>&1
}

# Header-only eligibility; the Rust planner validates the complete tower contract.
release_resolve_audio_mode() {
  local mode="$1" snapshot="$2"
  [[ "$mode" == auto ]] || { printf '%s' "$mode"; return; }
  python3 - "$snapshot" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    c = json.loads((p / "config.json").read_text())
    qualified = (c.get("model_type") == "mimo_v2" and c.get("hidden_size") in (4096, 6144)
                 and bool(c.get("audio_config")) and (p / "audio_tokenizer/config.json").is_file()
                 and (p / "audio_tokenizer/model.safetensors").is_file())
except (OSError, ValueError):
    qualified = False
print("auto" if qualified else "off")
PY
}

# Resolve fixed onboarding from the checkpoint's sparse layers.
release_onboard() {
  local value="$1" model="$2" revision="${3:-}" hub snapshot
  hub="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${model//\//--}"
  [[ -n "$revision" || ! -f "$hub/refs/main" ]] || revision="$(<"$hub/refs/main")"
  snapshot="$hub/snapshots/$revision/config.json"
  local -a args=(--onboard "$value")
  [[ ! -f "$snapshot" ]] || args+=("$snapshot")
  python3 "$(dirname "${BASH_SOURCE[0]}")/checkpoint-family.py" "${args[@]}"
}

release_load_config() {
  local config="$1"
  local mode="${2:-launch}"
  case "$mode" in
    launch|stop) ;;
    *) release_die "unknown release_load_config mode: $mode (expected launch or stop)" ;;
  esac
  [[ -f "$config" ]] || release_die "configuration file not found: $config"

  local default_model_id=deepseek-ai/DeepSeek-V4.1-Flash-1017
  MODEL_ID="$default_model_id"
  MODEL_VARIANT=flash
  VISION=auto
  AUDIO=auto
  MODEL_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
  SERVED_MODEL_ID=
  EXPERT_FORMAT=native
  SPECULATOR=dspark
  TP2_ATTENTION=off
  TP2_QUERY_PROJECTION=off
  TP2_OUTPUT_PROJECTION=off
  TP2_DSPARK_EXPERTS=off
  DSPARK_DRAFT_POLICY=adaptive
  DSPARK_DRAFT_LIMIT=
  V41_FP8_HEAD=
  RTX_EXPERT_LAYERS=auto
  RTX_GPUS=auto
  COORDINATOR_GPU=0
  COORDINATOR_GPU_UUID=
  COORDINATOR_GPU_PCI_BUS_ID=
  COORDINATOR_GPU_HEADROOM_GIB=8
  COORDINATOR_GPU_BUDGET_GIB=
  POOL_TOKENS=
  EMBEDDING=gpu
  KV_POOL_TOKENS=
  KV_POOL_SIZE=
  HOST_CACHE_BYTES=auto
  API_KEY_FILE="${API_KEY_FILE:-}"
  ENABLE_BENCH=off
  MEMORY_RESERVATION=
  MAX_CONTEXT_TOKENS=1048576
  MAX_OUTPUT_TOKENS=393216
  HTTP_QUEUE_DEPTH=
  HTTP_QUEUE_WAIT_MS=25000
  CONCURRENCY=
  RELEASE_CONCURRENCY_EXPLICIT=0
  PREFIX_CACHE_ENTRIES=20
  PREFILL_BATCH_TOKENS=2048
  RELEASE_PREFILL_EXPLICIT=0
  SPARK_DEVICE_BUDGET_BYTES=107374182400
  SPARK_REDUCTION_MIN_ROWS=16
  SPARKINFER_EXL3=disable
  if [[ -z "${_release_count_env_saved:-}" ]]; then
    _release_count_env="${SPARK_COUNT-}" _release_hosts_env="${SPARK_HOSTS-}"
    _release_count_env_saved=1
  fi
  local count_env="$_release_count_env" hosts_env="$_release_hosts_env"
  SPARK_COUNT=
  RELEASE_SPARK_COUNT_EXPLICIT=0
  SPARK_HOSTS=
  # Optional explicit replicated expert-group topology. Absent means the legacy
  # geometry (TP = SPARK_COUNT, EP = 1). See docs/tp-ep-configuration.md.
  SPARK_TP=
  SPARK_EP=
  PROFILE=
  PROFILE_DIR=
  PROFILE_NCU_KERNEL=
  PROFILE_NCU_COUNT=
  PROFILE_NCU_SET=
  PROFILE_NCU_SKIP=
  PROFILE_NSYS_GPU_METRICS=
  PROFILE_SPARK=
  PROFILE_SPARK_RANKS=
  PROFILE_SPARK_NCU_KERNEL=
  PROFILE_SPARK_NCU_COUNT=
  PROFILE_SPARK_NCU_SET=
  PROFILE_SPARK_NCU_SKIP=
  PROFILE_SPARK_NSYS_GPU_METRICS=
  PROFILE_EXPERT_INPUT_SAMPLE=
  PROFILE_EXPERT_ROUTE_WEIGHTS=
  ADDR=0.0.0.0:8000
  EXPERT_PORT=19441
  CUTEAFD_EXPORT_HOST=
  WIP_BUILD_SEED=kakapo
  COORDINATOR_DOCKER_DEV=cuteafd-coordinator-dev
  COORDINATOR_DOCKER_INFERENCE=cuteafd-coordinator
  SPARK_EXPERT_DOCKER_DEV=cuteafd-spark-expert-dev
  SPARK_EXPERT_DOCKER_INFERENCE=cuteafd-spark-expert
  # Reset on every load: the coordinator container name is derived from the
  # constant at the top of this file plus INSTANCE at the end of the function,
  # so a second load in one process must not suffix an already-suffixed name.
  RELEASE_COORDINATOR_CONTAINER_NAME=cuteafd-coordinator
  INSTANCE=
  local wip_env="${WIP_INSTANCE-}" wip_root_env="${WIP_ROOT-}"
  WIP_ROOT=
  WIP_INSTANCE=
  for release_i in {0..5}; do
    printf -v "SPARK_${release_i}_HOST" '%s' ""
    printf -v "SPARK_${release_i}_LANE_A" '%s' ""
    printf -v "SPARK_${release_i}_LANE_B" '%s' ""
  done

  local raw line key value model_id_explicit=0 model_revision_explicit=0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(release_trim "${raw%%#*}")"
    [[ -n "$line" ]] || continue
    [[ "$line" == *=* ]] || release_die "invalid configuration line: $raw"
    key="$(release_trim "${line%%=*}")"
    value="$(release_trim "${line#*=}")"
    release_known_key "$key" || release_die "unknown configuration key: $key"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == *[[:space:]]* ]]; then
      release_die "unquoted whitespace is not allowed for $key"
    fi
    printf -v "$key" '%s' "$value"
    [[ "$key" != CONCURRENCY ]] || RELEASE_CONCURRENCY_EXPLICIT=1
    [[ "$key" != PREFILL_BATCH_TOKENS || -z "$value" ]] || RELEASE_PREFILL_EXPLICIT=1
    if [[ ( "$key" == SPARK_COUNT || "$key" == SPARK_HOSTS ) && -n "$value" ]]; then
      RELEASE_SPARK_COUNT_EXPLICIT=1
    fi
    [[ "$key" != MODEL_ID ]] || model_id_explicit=1
    [[ "$key" != MODEL_REVISION ]] || model_revision_explicit=1
  done <"$config"

  WIP_INSTANCE="${wip_env:-$WIP_INSTANCE}"
  WIP_ROOT="${wip_root_env:-$WIP_ROOT}"
  release_validate_wip_root
  release_validate_wip_instance

  [[ -z "$count_env" ]] || SPARK_COUNT="$count_env"
  [[ -z "$hosts_env" ]] || SPARK_HOSTS="$hosts_env"
  [[ -z "$SPARK_COUNT$SPARK_HOSTS" ]] || RELEASE_SPARK_COUNT_EXPLICIT=1
  RELEASE_AVAILABLE_SPARKS=0
  for release_i in {0..5}; do
    local host_key="SPARK_${release_i}_HOST"
    [[ -n "${!host_key:-}" ]] || break
    RELEASE_AVAILABLE_SPARKS=$((RELEASE_AVAILABLE_SPARKS + 1))
  done
  if [[ -z "$SPARK_COUNT" ]]; then
    if [[ -n "$SPARK_HOSTS" ]]; then
      local -a count_hosts
      IFS=, read -r -a count_hosts <<<"$SPARK_HOSTS"
      SPARK_COUNT="${#count_hosts[@]}"
    else
      SPARK_COUNT="$RELEASE_AVAILABLE_SPARKS"
    fi
  fi
  if [[ -n "$SPARK_HOSTS" ]]; then
    local host_rows rank host lane_a lane_b
    host_rows="$(release_spark_host_rows "$SPARK_HOSTS" "$SPARK_COUNT" "$mode")" || return 2
    # Stop preserves legacy host names too: cleanup is the union of both lists.
    if [[ "$mode" != stop ]]; then
      while read -r rank host lane_a lane_b; do
        printf -v "SPARK_${rank}_HOST" '%s' "$host"
        printf -v "SPARK_${rank}_LANE_A" '%s' "$lane_a"
        printf -v "SPARK_${rank}_LANE_B" '%s' "$lane_b"
      done <<<"$host_rows"
    fi
  fi

  # A model override without its own revision must never inherit the pinned
  # calibrated-release commit from the defaults.
  if ((model_id_explicit && !model_revision_explicit)) &&
    [[ "$MODEL_ID" != "$default_model_id" ]]; then
    MODEL_REVISION=
  fi

  [[ "$VISION" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$VISION" != auto:* && "$VISION" != off:* ) ]] ||
    release_die "VISION must be auto, off, rtx[:gpu] or spark[:rank]"
  case "$AUDIO" in auto|off) ;; *) release_die "AUDIO must be auto or off" ;; esac
  case "$MODEL_VARIANT" in flash|pro) ;; *) release_die "MODEL_VARIANT must be flash or pro" ;; esac
  case "$EXPERT_FORMAT" in native|exl3) ;; *) release_die "EXPERT_FORMAT must be native or exl3" ;; esac
  if [[ "$mode" == launch ]]; then
    [[ "$MODEL_VARIANT" != pro || "$EXPERT_FORMAT" == exl3 ]] ||
      release_die "DeepSeek V4 Pro requires EXPERT_FORMAT=exl3"
  fi
  case "$DSPARK_DRAFT_POLICY" in
    full|adaptive) ;;
    *) release_die "DSPARK_DRAFT_POLICY must be full or adaptive" ;;
  esac
  case "$SPARKINFER_EXL3" in
    auto|disable|force) ;;
    *) release_die "SPARKINFER_EXL3 must be auto, disable, or force" ;;
  esac
  if [[ "$mode" == launch ]]; then
    [[ "$SPARKINFER_EXL3" != disable || "$EXPERT_FORMAT" == native ]] ||
      release_die "SPARKINFER_EXL3=disable requires EXPERT_FORMAT=native"
    [[ "$SPARKINFER_EXL3" != force || "$EXPERT_FORMAT" == exl3 ]] ||
      release_die "SPARKINFER_EXL3=force requires EXPERT_FORMAT=exl3"
  fi
  if [[ "$mode" == stop ]]; then
    RTX_EXPERT_LAYERS="$(python3 "$(dirname "${BASH_SOURCE[0]}")/checkpoint-family.py" --onboard "$RTX_EXPERT_LAYERS")" || return 2
  else
    RTX_EXPERT_LAYERS="$(release_onboard "$RTX_EXPERT_LAYERS" "$MODEL_ID" "$MODEL_REVISION")" || return 2
  fi
  case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
  [[ "$COORDINATOR_GPU" =~ ^[0-9]+$ ]] || release_die "COORDINATOR_GPU must be a non-negative host GPU index"
  [[ -z "$COORDINATOR_GPU_UUID" || "$COORDINATOR_GPU_UUID" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] ||
    release_die "COORDINATOR_GPU_UUID must be empty or a physical NVIDIA GPU UUID"
  [[ -z "$COORDINATOR_GPU_PCI_BUS_ID" || "$COORDINATOR_GPU_PCI_BUS_ID" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
    release_die "COORDINATOR_GPU_PCI_BUS_ID must be empty or a full PCI bus ID"
  [[ -z "$COORDINATOR_GPU_UUID" && -z "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
    [[ -n "$COORDINATOR_GPU_UUID" && -n "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
    release_die "COORDINATOR_GPU_UUID and COORDINATOR_GPU_PCI_BUS_ID must be set together"
  if ((RELEASE_CONCURRENCY_EXPLICIT)); then
    if [[ "$mode" == stop ]]; then
      [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || release_die "CONCURRENCY must be positive"
    else
      [[ "$CONCURRENCY" =~ ^([1-9]|1[0-6])$ ]] || release_die "CONCURRENCY must be in 1..16"
    fi
  fi
  [[ "$PREFIX_CACHE_ENTRIES" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8])$ ]] ||
    release_die "PREFIX_CACHE_ENTRIES must be in 0..128"
  [[ "$PREFILL_BATCH_TOKENS" =~ ^[0-9]+$ ]] || release_die "PREFILL_BATCH_TOKENS must be non-negative"
  if [[ "$mode" == launch ]]; then
    ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) ||
      release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
  fi
  [[ "$SPARK_DEVICE_BUDGET_BYTES" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_DEVICE_BUDGET_BYTES must be a positive integer"
  [[ "$SPARK_REDUCTION_MIN_ROWS" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_REDUCTION_MIN_ROWS must be a positive integer"
  [[ "$EXPERT_PORT" =~ ^[0-9]+$ ]] && ((EXPERT_PORT >= 1 && EXPERT_PORT <= 65535)) || release_die "EXPERT_PORT must be in 1..65535"
  # The same shape scripts/launch/run-family.sh accepts, so one key means one
  # container name in the family launcher and the production launcher alike.
  [[ -z "$INSTANCE" || "$INSTANCE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] ||
    release_die "INSTANCE must be [A-Za-z0-9_.-]"
  [[ "$COORDINATOR_GPU_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || release_die "COORDINATOR_GPU_HEADROOM_GIB must be non-negative"
  release_validate_coordinator_gpu_budget "$COORDINATOR_GPU_BUDGET_GIB"
  [[ -z "$POOL_TOKENS" || "$POOL_TOKENS" == auto || "$POOL_TOKENS" =~ ^[0-9]+$ ]] || release_die "POOL_TOKENS must be auto or a non-negative integer"
  [[ -z "$POOL_TOKENS" || -z "$KV_POOL_SIZE" ]] || release_die "POOL_TOKENS conflicts with KV_POOL_SIZE"
  for release_integer_name in KV_POOL_TOKENS MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS; do
    value="${!release_integer_name}"
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || release_die "$release_integer_name must be a positive integer"
  done
  if [[ "$mode" == launch ]]; then
    [[ -z "$MAX_CONTEXT_TOKENS" ]] || ((MAX_CONTEXT_TOKENS <= 1048576)) ||
      release_die "MAX_CONTEXT_TOKENS must be in 1..1048576"
    [[ -z "$MAX_OUTPUT_TOKENS" ]] || ((MAX_OUTPUT_TOKENS <= 393216)) ||
      release_die "MAX_OUTPUT_TOKENS must be in 1..393216"
  fi
  if [[ "$mode" == launch && -n "$KV_POOL_TOKENS" ]]; then
    ((KV_POOL_TOKENS % 64 == 0)) || release_die "KV_POOL_TOKENS must be a multiple of 64"
  fi
  [[ -z "$KV_POOL_SIZE" || "$KV_POOL_SIZE" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] ||
    release_die "KV_POOL_SIZE must use B, MB, GB, MiB or GiB"
  [[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] ||
    release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
  [[ -z "$MEMORY_RESERVATION" || "$MEMORY_RESERVATION" =~ ^[0-9]+([.][0-9]{1,6})?((B|MB|GB|MiB|GiB)|%)$ ]] ||
    release_die "MEMORY_RESERVATION must be a byte size or percentage"
  [[ "$ADDR" == *:* ]] || release_die "ADDR must be HOST:PORT"
  [[ -n "$MODEL_ID" && "$MODEL_ID" == */* && "$MODEL_ID" != *[[:space:]]* ]] ||
    release_die "MODEL_ID must be a Hugging Face repository ID"
  [[ -z "$MODEL_REVISION" || "$MODEL_REVISION" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "MODEL_REVISION must be empty or a 40..64 lowercase hex revision"

  release_validate_tp2_options
  if [[ "$mode" == launch ]]; then
    release_validate_spark_topology

    [[ "$SPARK_COUNT" =~ ^[0-6]$ ]] || release_die "SPARK_COUNT must be in 0..6"

    local missing_b=0 present_b=0 spark_required="$SPARK_COUNT"
    for ((release_i = 0; release_i < spark_required; release_i++)); do
      local host_name="SPARK_${release_i}_HOST"
      local lane_a_name="SPARK_${release_i}_LANE_A"
      local lane_b_name="SPARK_${release_i}_LANE_B"
      if ((release_i < spark_required)); then
        [[ -n "${!host_name}" ]] || release_die "$host_name must not be empty"
        [[ "${!lane_a_name}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || release_die "$lane_a_name must be an IPv4 address"
      fi
      if [[ -n "${!lane_b_name}" ]]; then
        ((present_b += 1))
      else
        ((missing_b += 1))
      fi
    done
    ((present_b == 0 || missing_b == 0)) || release_die "secondary Spark rail must provide all $SPARK_COUNT active LANE_B values or none"
  else
    # Cleanup discovers the hosts itself; serving topology, GPU layout, model
    # format and the reduction rail are launch requirements, not stop safety.
    release_validate_stop_config
  fi

  for release_image_name in COORDINATOR_DOCKER_DEV COORDINATOR_DOCKER_INFERENCE SPARK_EXPERT_DOCKER_DEV SPARK_EXPERT_DOCKER_INFERENCE; do
    [[ -n "${!release_image_name}" && "${!release_image_name}" != *[[:space:]]* ]] || release_die "$release_image_name must be a Docker image reference"
  done

  RELEASE_CONFIG="$(realpath "$config")"
  RELEASE_MODEL_ID="$MODEL_ID"
  RELEASE_MODEL_REVISION="$MODEL_REVISION"
  # The native API advertises the launched checkpoint's id (the daemon derives
  # it from the snapshot path), or SERVED_MODEL_ID when the config names one.
  RELEASE_NATIVE_API_MODEL_ID="${SERVED_MODEL_ID:-$MODEL_ID}"
  # INSTANCE names a launch that runs beside others on disjoint hardware: its
  # coordinator container is cuteafd-coordinator-INSTANCE, exactly the suffix
  # scripts/launch/run-family.sh derives from the same key, so pickers and stops
  # can never touch another agent's server. Empty keeps the single shared
  # cuteafd-coordinator. Derived here, from the constant, so every caller
  # (run.sh, stop.sh, build.sh) uses the same name and a repeated load cannot
  # double-suffix it. Spark workers stay host+port keyed: their names already
  # carry the host and the expert port, so two launches on that host would
  # collide on the port regardless of INSTANCE.
  RELEASE_COORDINATOR_CONTAINER_NAME="${RELEASE_COORDINATOR_CONTAINER_NAME}${INSTANCE:+-$INSTANCE}"
}

# Stop-only configuration validation.
#
# Cleanup launches nothing, so an incomplete or unsupported serving topology
# must not block it. SPARK_COUNT/SPARK_TP/SPARK_EP are launch topology: stop
# accepts any non-negative integer count and positive integer degrees, or their
# absence, and cleans every Spark host the file names. LANE_A/B, GPU layout,
# model variant/format and EXL3 combination rules are also launch-only. What
# remains is the safety contract: the parser, known-key, quoting and every
# value-domain check above already ran, and a named Spark host must be a safe
# token because cleanup interpolates it into SSH arguments and container names.
# The accepted shape is Docker's own container-name rule
# ([A-Za-z0-9][A-Za-z0-9_.-]*): it preserves every alias the launcher can
# already use, including a trailing separator, while rejecting whitespace,
# shell metacharacters, `user@host` and IPv6 literals.
release_validate_stop_config() {
  local name value
  [[ -z "$SPARK_COUNT" || "$SPARK_COUNT" =~ ^[0-9]+$ ]] ||
    release_die "SPARK_COUNT must be a non-negative integer"
  [[ -z "$SPARK_TP" || "$SPARK_TP" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_TP must be a positive integer"
  [[ -z "$SPARK_EP" || "$SPARK_EP" =~ ^[1-9][0-9]*$ ]] ||
    release_die "SPARK_EP must be a positive integer"
  for name in $(compgen -v SPARK_ 2>/dev/null || true); do
    [[ "$name" =~ ^SPARK_[0-9]+_HOST$ ]] || continue
    value="${!name:-}"
    [[ -n "$value" ]] || continue
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
      release_die "$name is not a safe host token: $value"
  done
}

release_resolve_coordinator_gpu_identity() {
  local selector gpu_line observed_index observed_uuid observed_pci
  selector="${COORDINATOR_GPU_PCI_BUS_ID:-$COORDINATOR_GPU}"
  gpu_line="$(
    nvidia-smi --id="$selector" \
      --query-gpu=index,uuid,pci.bus_id \
      --format=csv,noheader,nounits
  )" || release_die "cannot resolve coordinator GPU selector: $selector"
  [[ -n "$gpu_line" && "$gpu_line" != *$'\n'* ]] ||
    release_die "coordinator GPU selector did not resolve to exactly one device: $selector"
  IFS=, read -r observed_index observed_uuid observed_pci <<<"$gpu_line"
  observed_index="$(release_trim "$observed_index")"
  observed_uuid="$(release_trim "$observed_uuid")"
  observed_pci="$(release_trim "$observed_pci")"
  [[ "$observed_index" =~ ^[0-9]+$ ]] ||
    release_die "coordinator GPU has an invalid host index: $observed_index"
  [[ "$observed_uuid" =~ ^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$ ]] ||
    release_die "coordinator GPU has an invalid physical UUID: $observed_uuid"
  [[ "$observed_pci" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
    release_die "coordinator GPU has an invalid PCI identity: $observed_pci"
  if [[ -n "$COORDINATOR_GPU_UUID" ]]; then
    [[ "$observed_uuid" == "$COORDINATOR_GPU_UUID" && "$observed_pci" == "$COORDINATOR_GPU_PCI_BUS_ID" ]] ||
      release_die "coordinator GPU PCI selector resolved to another physical device"
  else
    [[ "$observed_index" == "$COORDINATOR_GPU" ]] ||
      release_die "coordinator GPU ordinal resolved to an unexpected device"
  fi
  COORDINATOR_GPU_UUID="$observed_uuid"
  COORDINATOR_GPU_PCI_BUS_ID="$observed_pci"
  RELEASE_COORDINATOR_GPU_UUID="$observed_uuid"
  RELEASE_COORDINATOR_GPU_HOST_INDEX="$observed_index"
  RELEASE_COORDINATOR_GPU_PCI_BUS_ID="$observed_pci"
}

release_resolve_local_model_revision() {
  local hf_home="$1"
  local model_root="$hf_home/hub/models--${RELEASE_MODEL_ID//\//--}"
  if [[ -z "$RELEASE_MODEL_REVISION" ]]; then
    [[ -s "$model_root/refs/main" ]] ||
      release_die "MODEL_REVISION is empty and $RELEASE_MODEL_ID has no local refs/main"
    RELEASE_MODEL_REVISION="$(<"$model_root/refs/main")"
  fi
  [[ "$RELEASE_MODEL_REVISION" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "resolved model revision is not 40..64 lowercase hex: $RELEASE_MODEL_REVISION"
  [[ -d "$model_root/snapshots/$RELEASE_MODEL_REVISION" ]] ||
    release_die "model snapshot is missing: $RELEASE_MODEL_ID@$RELEASE_MODEL_REVISION"
  AUDIO="$(release_resolve_audio_mode "$AUDIO" "$model_root/snapshots/$RELEASE_MODEL_REVISION")"
}

# Two or three Spark ranks without explicit SPARK_TP/SPARK_EP keys form the
# opt-in compact EXL3 topology (TP2 or TP3), not RTX tensor parallelism. Keep
# the coordinator ceiling explicit; never raise it to fit. An explicit native
# replicated topology is never compact and bypasses these rules at every call
# site through this predicate.
release_spark_compact_active() {
  case "$SPARK_COUNT" in
    2|3) [[ -z "$SPARK_TP" && -z "$SPARK_EP" ]] ;;
    *) return 1 ;;
  esac
}


release_spark_values() {
  local field="$1" i name
  for ((i = 0; i < SPARK_COUNT; i++)); do
    name="SPARK_${i}_${field}"
    printf '%s\n' "${!name}"
  done
}

# ---------------------------------------------------------------------------
# Opt-in replicated expert-group topology (SPARK_TP x SPARK_EP = SPARK_COUNT).
#
# The default configuration sets neither key and keeps the legacy geometry:
# every Spark rank is one TP rank of a single replicated group (counts 0/2/4;
# count 2 without keys is the compact EXL3 TP2 layout, count 3 without keys is
# the compact EXL3 TP3 layout, and native count 3 requires both keys). Explicit
# keys are all-or-none and only valid for the approved native official
# topologies. The rank map is group-major: group = rank / TP and
# tp_rank = rank % TP.
# ---------------------------------------------------------------------------

# Release and WIP must advertise the same compiled extra Spark roles. TP4 is
# built separately; keep the qualified default until new roles have exports.
release_spark_tp_roles_default() {
  printf '%s\n' 'tp2;tp3;tp6'
}

release_spark_tp_roles_canonical() {
  local raw="$1" source_name="${2:-CUTEAFD_RELEASE_SPARK_TP_ROLES}"
  local entry prior
  local -a parts=() selected=()
  [[ -n "$raw" ]] || return 0
  [[ "$raw" != *";;"* && "$raw" != ";"* && "$raw" != *";" && "$raw" != *$'\n'* ]] ||
    release_die "$source_name is not a ';'-separated role list: $raw"
  IFS=';' read -ra parts <<<"$raw"
  for entry in "${parts[@]}"; do
    case "$entry" in
      tp2|tp3|tp6) ;;
      *) release_die "$source_name accepts only tp2, tp3 and tp6, got: $entry" ;;
    esac
    for prior in ${selected[@]+"${selected[@]}"}; do
      [[ "$prior" != "$entry" ]] ||
        release_die "$source_name lists $entry more than once"
    done
    selected+=("$entry")
  done
  printf '%s\n' "${selected[@]}" | sort | paste -sd';' -
}

release_spark_topology_explicit() {
  [[ -n "$SPARK_TP" || -n "$SPARK_EP" ]]
}

release_spark_tp() {
  if [[ -n "$SPARK_TP" ]]; then printf '%s\n' "$SPARK_TP"; else printf '%s\n' "$SPARK_COUNT"; fi
}

release_spark_ep() {
  if [[ -n "$SPARK_EP" ]]; then printf '%s\n' "$SPARK_EP"; else printf '1\n'; fi
}

release_spark_group() {
  local rank="$1" tp="$2"
  [[ "$rank" =~ ^[0-9]+$ && "$tp" =~ ^[1-9][0-9]*$ ]] ||
    release_die "invalid Spark rank/TP for group resolution: rank=$rank tp=$tp"
  printf '%s\n' "$((rank / tp))"
}

release_spark_tp_rank() {
  local rank="$1" tp="$2"
  [[ "$rank" =~ ^[0-9]+$ && "$tp" =~ ^[1-9][0-9]*$ ]] ||
    release_die "invalid Spark rank/TP for tp-rank resolution: rank=$rank tp=$tp"
  printf '%s\n' "$((rank % tp))"
}

# Print "rank group tp_rank" for every configured physical Spark rank.
release_spark_rank_map() {
  local tp rank
  tp="$(release_spark_tp)"
  for ((rank = 0; rank < SPARK_COUNT; rank++)); do
    printf '%s %s %s\n' "$rank" "$(release_spark_group "$rank" "$tp")" "$(release_spark_tp_rank "$rank" "$tp")"
  done
}

release_validate_spark_topology() {
  case "$SPARK_TP" in
    ""|[1-6]) ;;
    *) release_die "SPARK_TP must be in 1..6" ;;
  esac
  case "$SPARK_EP" in
    ""|1|2|3) ;;
    *) release_die "SPARK_EP must be 1, 2, or 3" ;;
  esac
  [[ -n "$SPARK_TP" && -n "$SPARK_EP" || -z "$SPARK_TP" && -z "$SPARK_EP" ]] ||
    release_die "SPARK_TP and SPARK_EP must be set together or omitted together"
  release_spark_topology_explicit || return 0

  [[ "$SPARK_COUNT" =~ ^[1-6]$ ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires SPARK_COUNT in 1..6"
  if [[ "$EXPERT_FORMAT" == exl3 ]]; then
    [[ "$SPARK_TP" == 3 && "$SPARK_EP" == 1 && "$SPARK_COUNT" == 3 ]] ||
      release_die "explicit SPARK_TP/SPARK_EP requires EXPERT_FORMAT=native except compact EXL3 TP3EP1"
    [[ "$SPARKINFER_EXL3" != disable ]] ||
      release_die "compact EXL3 TP3EP1 requires SPARKINFER_EXL3=auto or force"
    return 0
  fi
  [[ "$EXPERT_FORMAT" == native ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires EXPERT_FORMAT=native"
  [[ "$SPARKINFER_EXL3" == disable ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires SPARKINFER_EXL3=disable"
  ((SPARK_TP * SPARK_EP == SPARK_COUNT)) ||
    release_die "SPARK_TP(${SPARK_TP}) * SPARK_EP(${SPARK_EP}) must equal SPARK_COUNT(${SPARK_COUNT})"
  case "${SPARK_TP}x${SPARK_EP}" in
    2x1|3x1|4x1|6x1|2x2|3x2|2x3) ;;
    *) release_die "unsupported native Spark topology TP${SPARK_TP}EP${SPARK_EP}; approved: TP2/TP3/TP4/TP6 with EP1, TP2EP2, TP3EP2, TP2EP3" ;;
  esac
}

release_hosts_csv() { release_spark_values HOST | paste -sd, -; }
release_lane_a_csv() { release_spark_values LANE_A | paste -sd, -; }
release_lane_b_csv() {
  [[ -n "$SPARK_0_LANE_B" ]] || return 0
  release_spark_values LANE_B | paste -sd, -
}

# Optional RDMA/verbs tuning values shared by run.sh and the isolated candidate
# launcher. They are forwarded to both roles only when the operator sets them;
# an empty value keeps the transport's own device selection. The device map is
# `local-ip=device` comma-separated and must name unique IPv4 sources, which is
# what multi-homed six-rank hosts (rhea/moa) need to pin a rail.
release_validate_verbs_device_map() {
  local map="${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}"
  [[ -n "$map" ]] || return 0
  local -a entries=() seen_ips=()
  local entry ip dev prior
  IFS=',' read -ra entries <<<"$map"
  for entry in "${entries[@]}"; do
    [[ "$entry" == *=* ]] ||
      release_die "invalid CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP entry: '$entry' (expected local-ip=device)"
    ip="${entry%%=*}"
    dev="${entry#*=}"
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ && "$dev" =~ ^[A-Za-z0-9_.:-]+$ ]] ||
      release_die "invalid CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP entry: '$entry' (expected local-ip=device)"
    for prior in ${seen_ips[@]+"${seen_ips[@]}"}; do
      [[ "$ip" != "$prior" ]] ||
        release_die "duplicate CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP ip: $ip"
    done
    seen_ips+=("$ip")
  done
}

release_expert_hosts_csv() {
  local i lane separator=
  for ((i = 0; i < SPARK_COUNT; i++)); do
    lane="SPARK_${i}_LANE_A"
    printf '%sspark-%s=%s:%s' "$separator" "$i" "${!lane}" "$EXPERT_PORT"
    separator=,
  done
}

# ---------------------------------------------------------------------------
# Stop-side Spark host scope.
#
# Launch and topology helpers (release_spark_values) enumerate exactly the
# active SPARK_COUNT ranks. Cleanup must be a superset of that: a previous
# six-rank run can leave release or WIP containers on the fifth/sixth hosts
# even when the configuration currently selects a smaller serving set, and the
# default configuration names those hosts. stop.sh therefore selects every
# Spark host the configuration names, independent of SPARK_COUNT, before
# calling any stop helper. Duplicate names collapse to one host so a repeated
# name is not contacted twice. Launch/restart callers leave RELEASE_STOP_HOSTS
# unset and keep cleaning only the active ranks.
release_select_stop_hosts() {
  local name host seen_host index
  local -a indices=() seen=()
  RELEASE_STOP_HOSTS=()
  if [[ -n "${SPARK_HOSTS:-}" ]]; then
    local -a explicit_hosts=()
    IFS=, read -r -a explicit_hosts <<<"$SPARK_HOSTS"
    RELEASE_STOP_HOSTS+=("${explicit_hosts[@]}")
    seen+=("${explicit_hosts[@]}")
  fi

  # Enumerate every configured SPARK_<rank>_HOST variable, whatever its index.
  # The configuration grammar (release_known_key) is the single authority on
  # which ranks exist, so an out-of-grammar key is reported rather than
  # skipped, and a future added rank needs no second hard-coded bound here.
  for name in $(compgen -v SPARK_ 2>/dev/null || true); do
    [[ "$name" =~ ^SPARK_[0-9]+_HOST$ ]] || continue
    index="${name#SPARK_}"
    index="${index%_HOST}"
    [[ "$index" =~ ^(0|[1-9][0-9]*)$ ]] &&
      release_known_key "$name" ||
      release_die "unsupported Spark host key $name; stop cleanup follows the configured SPARK_<rank>_HOST grammar"
    [[ -n "${!name:-}" ]] || continue
    indices+=("$index")
  done
  ((${#indices[@]})) || return 0

  while IFS= read -r index; do
    name="SPARK_${index}_HOST"
    host="${!name:-}"
    for seen_host in ${seen[@]+"${seen[@]}"}; do
      [[ "$host" != "$seen_host" ]] || { host=""; break; }
    done
    [[ -n "$host" ]] || continue
    seen+=("$host")
    RELEASE_STOP_HOSTS+=("$host")
  done < <(printf '%s\n' "${indices[@]}" | sort -n)
}

# Emit the cleanup host set: every configured host once release_select_stop_hosts
# has run, otherwise the active ranks for launch/restart callers.
release_stop_hosts() {
  if [[ -n "${RELEASE_STOP_HOSTS+x}" && "${#RELEASE_STOP_HOSTS[@]}" -gt 0 ]]; then
    printf '%s\n' "${RELEASE_STOP_HOSTS[@]}"
  else
    release_spark_values HOST
  fi
}

release_stop_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping $container"
    # A profiled coordinator (release_profile_wrapper) keeps its own stop timeout.
    if [[ "$(docker inspect -f '{{index .Config.Labels "cuteafd.graceful-stop"}}' "$container")" == 1 ]]; then
      docker stop "$container" >/dev/null
    else
      docker stop -t 30 "$container" >/dev/null
    fi
  else
    echo "  coordinator: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
}

release_stop_host_api() {
  local addr="$1"
  local port="${addr##*:}"
  local pids
  pids="$(
    ss -ltnp "sport = :$port" 2>/dev/null |
      sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
      sort -u
  )"
  [[ -n "$pids" ]] || return 0
  local pid command
  for pid in $pids; do
    command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$command" == *cuteafd*coordinator* ]] ||
      release_die "port $port is owned by a non-CUTEAFD process: pid=$pid $command"
    echo "  coordinator: stopping host API pid=$pid"
    kill -TERM "$pid"
  done
  for _ in $(seq 1 300); do
    ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q . || return 0
    sleep 0.1
  done
  release_die "host API did not exit within 30 seconds"
}

release_stop_remote_containers() {
  local host="$1"
  local release_container="$2"
  local legacy_container="$3"
  release_ssh "$host" bash -s -- \
    "$host" "$release_container" "$legacy_container" <<'REMOTE'
set -euo pipefail
host="$1"
shift
for container in "$@"; do
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    continue
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  $host: stopping $container"
    # A profiled worker (PROFILE_SPARK) keeps its own stop signal and timeout.
    if [[ "$(docker inspect -f '{{index .Config.Labels "cuteafd.graceful-stop"}}' "$container")" == 1 ]]; then
      docker stop "$container" >/dev/null
    else
      docker stop -t 30 "$container" >/dev/null
    fi
  else
    echo "  $host: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
done
REMOTE
}

# Explicit --all cleanup only: never sweep worker ports during normal restart.
release_stop_all_worker_containers() {
  local host failed=0
  local -a pids=()
  while IFS= read -r host; do
    [[ -n "$host" ]] || continue
    release_ssh "$host" 'ids=$(docker ps -a --format "{{.Names}}" --filter "name=^cuteafd-spark-expert-.+-[0-9]+$" | grep -vE "^cuteafd-spark-expert-wip($|-)"); [ -z "$ids" ] || docker rm -f $ids >/dev/null' &
    pids+=("$!")
  done < <(release_stop_hosts)
  local pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0))
}

release_stop_services() {
  local coordinator_container="$1"
  local spark_container_prefix="$2"
  release_stop_local_container "$coordinator_container"
  release_stop_host_api "$ADDR"

  local host release_container legacy_container
  local failed=0
  local -a stop_hosts=()
  local -a stop_pids=()
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_stop_hosts)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
    release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
    legacy_container="cuteafd-phase0-tcp-expertd-${host}-${EXPERT_PORT}"
    release_stop_remote_containers \
      "$host" "$release_container" "$legacy_container" &
    stop_hosts+=("$host")
    stop_pids+=("$!")
  done
  local index
  for index in "${!stop_pids[@]}"; do
    if ! wait "${stop_pids[$index]}"; then
      echo "  ${stop_hosts[$index]}: failed to stop one or more CUTEAFD containers" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_persistent_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping persistent $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  coordinator: persistent $container is already stopped"
  fi
}

release_stop_persistent_remote_container() {
  local host="$1"
  local container="$2"
  release_ssh "$host" bash -s -- "$host" "$container" <<'REMOTE'
set -euo pipefail
host="$1"
container="$2"
if ! docker container inspect "$container" >/dev/null 2>&1; then
  exit 0
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
  echo "  $host: stopping persistent $container"
  docker stop -t 30 "$container" >/dev/null
else
  echo "  $host: persistent $container is already stopped"
fi
REMOTE
}

release_stop_wip_containers() {
  local coordinator_container="${1:-$(release_wip_container coordinator)}"
  local spark_container="${2:-$(release_wip_container spark-expert)}"
  local failed=0

  release_stop_persistent_local_container "$coordinator_container" || failed=1

  local host
  local -a hosts=() pids=()
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_stop_hosts)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
    release_stop_persistent_remote_container "$host" "$spark_container" &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop persistent $spark_container" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_wip_process_in_container() {
  local container="$1"
  local process_name="$2"
  docker exec -i "$container" bash -s -- "$process_name" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
}

release_stop_wip_coordinator() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local coordinator_container="$(release_wip_container coordinator)"

  if docker container inspect "$coordinator_container" >/dev/null 2>&1 &&
    [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]]; then
    echo "  coordinator: stopping WIP process $coordinator_process"
    release_stop_wip_process_in_container \
      "$coordinator_container" "$coordinator_process"
  fi
}

release_stop_wip_services() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local expert_process="${2:-expert-$EXPERT_PORT}"
  local coordinator_container="$(release_wip_container coordinator)"
  local spark_container="$(release_wip_container spark-expert)"
  local failed=0

  release_stop_wip_coordinator "$coordinator_process" || failed=1

  local host
  local -a hosts=() pids=()
  local -a active_hosts=()
  mapfile -t active_hosts < <(release_stop_hosts)
  for host in "${active_hosts[@]}"; do
    [[ -n "$host" ]] || continue
    (
      # Distinguish an unreachable host from one that simply has no running
      # WIP container. The remote command always exits 0 because of `|| true`,
      # so a nonzero ssh status is a transport/remote-shell failure that must
      # be reported instead of being swallowed as "nothing to stop".
      state="$(release_ssh "$host" \
        "docker inspect -f '{{.State.Running}}' '$spark_container' 2>/dev/null || true")" ||
        exit 1
      [[ "$state" == true ]] || exit 0
      release_ssh "$host" docker exec -i "$spark_container" \
        bash -s -- "$expert_process" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
    ) &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop WIP process $expert_process" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

# Build availability is independent of the four-rank runtime topology.
release_select_build_hosts() {
  local requested="${1:-}" host configured found prior name rank
  local -a configured_hosts=()
  if [[ -n "${SPARK_COUNT:-}" ]]; then
    mapfile -t configured_hosts < <(release_spark_values HOST)
  else
    # Callers that have not loaded a configuration may still set the host keys
    # directly; fall back to the bounded rank keys and keep only filled ones.
    for ((rank = 0; rank < 8; rank++)); do
      name="SPARK_${rank}_HOST"
      [[ -n "${!name:-}" ]] && configured_hosts+=("${!name}")
    done
  fi
  RELEASE_BUILD_HOSTS=()
  if [[ -z "$requested" ]]; then
    RELEASE_BUILD_HOSTS=("${configured_hosts[@]}")
    return 0
  fi
  [[ "$requested" != ,* && "$requested" != *, && "$requested" != *,,* ]] ||
    release_die "--spark-hosts contains an empty host"
  local -a requested_hosts
  IFS=, read -r -a requested_hosts <<< "$requested"
  for host in "${requested_hosts[@]}"; do
    found=0
    for configured in "${configured_hosts[@]}"; do
      [[ "$host" != "$configured" ]] || found=1
    done
    ((found)) || release_die "build host is not configured: $host"
    for prior in "${RELEASE_BUILD_HOSTS[@]}"; do
      [[ "$host" != "$prior" ]] || release_die "duplicate build host: $host"
    done
    RELEASE_BUILD_HOSTS+=("$host")
  done
}

# Native per-TP-rank routed weight for one 40-layer backbone layer, in bytes.
# These are the exact tensor windows from the official checkpoint geometry:
# 1,804,861,440 B/rank at TP4 raw, padded to a 640-wide kernel extent
# (2,005,401,600 B); TP6 (384), TP2 (1152) and TP3 (768) need no padding, so
# their values are the exact 2304/TP window times the uniform per-column cost
# (1,804,861,440 * TP4/TP = 7,219,445,760 / TP). They are weight arithmetic only
# and do not include workspace, staging or runtime headroom.
release_spark_layer_bytes() {
  local tp="$1" format="${2:-native}"
  if [[ "$format" == nvfp4 ]]; then
    local intermediate stored fc1_rows down_scale_cols
    case "$tp" in 3|4|6) ;; *) release_die "NVFP4 Spark weights require TP3, TP4 or TP6" ;; esac
    intermediate=$((2304 / tp))
    stored=$(((intermediate + 127) / 128 * 128))
    fc1_rows=$(((2 * stored + 127) / 128 * 128))
    down_scale_cols=$(((stored / 16 + 3) / 4 * 4))
    # Mirror Nvfp4Side::plane_sizes plus its four per-expert FP32 vectors.
    printf '%s\n' "$((384 * (3 * 5120 * stored / 2 + fc1_rows * 320 + 5120 * down_scale_cols + 16)))"
    return
  fi
  [[ "$format" == native ]] || release_die "unsupported Spark weight format: $format"
  case "$tp" in
    2) printf '%s\n' 3609722880 ;;
    3) printf '%s\n' 2406481920 ;;
    4) printf '%s\n' 2005401600 ;;
    6) printf '%s\n' 1203240960 ;;
    *) release_die "unsupported Spark TP degree: $1 (expected 2, 3, 4, or 6)" ;;
  esac
}

# Validate booleans independently of resolved GPU selection. Called again after
# command-line overrides, before launcher operations that change services.
release_validate_tp2_options() {
  local name
  for name in TP2_ATTENTION TP2_QUERY_PROJECTION TP2_OUTPUT_PROJECTION TP2_DSPARK_EXPERTS; do
    case "${!name}" in on|off) ;; *) release_die "$name must be on or off" ;; esac
  done
}

release_tp2_enabled() {
  [[ "$TP2_ATTENTION" == on || "$TP2_QUERY_PROJECTION" == on ||
     "$TP2_OUTPUT_PROJECTION" == on || "$TP2_DSPARK_EXPERTS" == on ]]
}

# Registry dev references are pulled only when absent; local campaign tags remain valid.
release_ensure_dev_image() {
  local image="$1" host="${2:-}"
  if [[ -n "$host" ]]; then
    release_ssh "$host" bash -s -- "$image" <<'REMOTE'
set -euo pipefail
image="$1"
docker image inspect "$image" >/dev/null 2>&1 && exit 0
registry="${image%%/*}"
[[ "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" == localhost ) ]] || {
  printf 'missing local development image %s; rebuild with scripts/build/build-dev-images.sh\n' "$image" >&2; exit 1;
}
docker pull "$image"
REMOTE
  else
    docker image inspect "$image" >/dev/null 2>&1 && return 0
    local registry="${image%%/*}"
    [[ "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" == localhost ) ]] ||
      release_die "missing local development image $image; rebuild with scripts/build/build-dev-images.sh"
    docker pull "$image"
  fi
}

# A sealed slot is files under the WIP root (the persistent container only bind-mounts
# it at /wip), so staging reads the root directly when the builder container is gone.
release_wip_host_root() {
  printf '%s' "${WIP_ROOT:-$HOME/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}"
}

# release_wip_cp CONTAINER HOST_ROOT PATH_UNDER_/wip DEST: docker cp while the builder
# container exists, else a plain copy from the same path under HOST_ROOT.
release_wip_cp() {
  local container="$1" root="$2" rel="$3" dest="$4"
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker cp "$container:/wip/$rel" "$dest"
  else
    [[ -e "$root/${rel%/.}" ]] || return 1
    cp -a "$root/$rel" "$dest"
  fi
}

release_stage_wip_layout() {
  local container="$1" slot="$2" role="$3" layout="$4"
  local raw="$layout.tmp/raw" root
  root="$(release_wip_host_root)"
  [[ -d "$root/slots/$slot/$role" ]] || docker container inspect "$container" >/dev/null 2>&1 ||
    release_die "WIP slot $slot is not under $root and $container is missing; build the slot with ./wip.sh first"
  rm -rf "$layout.tmp" && mkdir -p "$raw" "$layout.tmp/bin" "$layout.tmp/lib" "$layout.tmp/share"
  release_wip_cp "$container" "$root" "slots/$slot/$role/workspace/.cuteafd-wip/." "$raw/" ||
    release_die "WIP slot $slot has no $role artifacts in $container or $root"
  release_wip_cp "$container" "$root" "slots/$slot/$role/workspace/docker/release-entrypoint.sh" "$raw/" ||
    release_die "WIP slot $slot lacks docker/release-entrypoint.sh"
  mv "$raw/cuteafd" "$layout.tmp/bin/cuteafd"
  mv "$raw/libcuteafd_native.so" "$layout.tmp/lib/"
  # Host-ISA variants (scripts/build/host-isa.sh): bin/<isa>/cuteafd, lib/<isa>/libcuteafd_native.so.
  if [[ -d "$raw/bin-variants" ]]; then cp -a "$raw/bin-variants/." "$layout.tmp/bin/"; rm -rf "$raw/bin-variants"; fi
  if [[ -d "$raw/lib-variants" ]]; then cp -a "$raw/lib-variants/." "$layout.tmp/lib/"; rm -rf "$raw/lib-variants"; fi
  [[ ! -d "$raw/exl3" ]] || mv "$raw/exl3" "$layout.tmp/lib/exl3"
  [[ ! -d "$raw/fp8" ]] || mv "$raw/fp8" "$layout.tmp/lib/fp8"
  mv "$raw/"* "$layout.tmp/share/"
  mkdir -p "$layout.tmp/source/third_party" "$layout.tmp/source/scripts/build"
  local slot_source="slots/$slot/$role/workspace"
  release_wip_cp "$container" "$root" "$slot_source/third_party/sparkinfer" "$layout.tmp/source/third_party/"
  release_wip_cp "$container" "$root" "$slot_source/third_party/sparkinfer.lock.json" "$layout.tmp/source/third_party/"
  release_wip_cp "$container" "$root" "$slot_source/scripts/build/verify-sparkinfer-source.py" "$layout.tmp/source/scripts/build/"
  python3 "$layout.tmp/source/scripts/build/verify-sparkinfer-source.py" \
    --source "$layout.tmp/source/third_party/sparkinfer" --lock "$layout.tmp/source/third_party/sparkinfer.lock.json"
  rm -rf "$raw" "$layout"
  mv "$layout.tmp" "$layout"
}

