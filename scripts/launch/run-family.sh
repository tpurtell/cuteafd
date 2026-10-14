#!/usr/bin/env bash
# Launch any supported checkpoint: DeepSeek V4.1 (serve-native, its own section
# below), DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 or Qwen 3.8 Flash Next
# (the family's serve command on one or two RTX, routed experts on the first
# SPARK_COUNT Sparks, or SPARK_HOSTS in explicit rank order) from the release
# images named in the config, or a ./wip.sh slot's artifacts (--wip). ./run.sh
# starts this for every family; the family comes from the snapshot's
# config.json (scripts/lib/checkpoint-family.py), or --family. Containers use
# run.sh's names, so ./stop.sh stops them.
set -euo pipefail
launch_argv=("$@")
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
source "$repo_root/scripts/lib/cdi-gpus.sh"
config="$repo_root/cuteafd.config"
restart=0
restart_all=0
family=""
embedding_override=""
wip_slot=""
dry_run=0
# --set KEY=VALUE overrides one configuration key for this launch (validated by
# the same key grammar as the file).
declare -A cfg_overrides=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?--config requires FILE}"; shift 2 ;;
    --family) family="${2:?--family requires ID}"; shift 2 ;;
    --embedding-placement) embedding_override="${2:?--embedding-placement requires host or gpu}"; shift 2 ;;
    --set)
      setting="${2:?--set requires KEY=VALUE}"
      [[ "$setting" == *=* ]] || release_die "--set requires KEY=VALUE, not $setting"
      release_known_key "${setting%%=*}" || release_die "unknown configuration key: ${setting%%=*}"
      cfg_overrides[${setting%%=*}]="${setting#*=}"; shift 2 ;;
    --restart) restart=1; shift ;;
    --all) restart_all=1; shift ;;
    --wip) wip_slot="${2:?--wip requires SLOT}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "usage: $0 [--config FILE] [--family ID] [--set KEY=VALUE]... [--embedding-placement host|gpu] [--restart [--all]] [--wip SLOT] [--dry-run]" >&2; exit 2 ;;
  esac
done
((restart_all == 0 || restart == 1)) || release_die "--all requires --restart"
((restart == 0 || dry_run == 0)) || release_die "--restart and --dry-run are mutually exclusive"
[[ -z "$embedding_override" ]] || cfg_overrides[EMBEDDING]="$embedding_override"
# Overrides are appended to a copy of the file (a later line wins in every
# parser), so they get exactly the file's validation. The copy is named by its
# content and kept beside the other launch state.
if ((${#cfg_overrides[@]})); then
  [[ -f "$config" ]] || release_die "configuration file not found: $config"
  merged="$(cat "$config"; printf '
# run.sh --set
'; for key in $(printf '%s
' "${!cfg_overrides[@]}" | sort); do printf '%s=%s
' "$key" "${cfg_overrides[$key]}"; done)"
  merged_dir="$HOME/.cache/cuteafd/run-config"
  mkdir -p "$merged_dir"
  merged_path="$merged_dir/$(printf '%s
' "$merged" | sha256sum | cut -c1-16).config"
  printf '%s
' "$merged" >"$merged_path.tmp.$$" && mv "$merged_path.tmp.$$" "$merged_path"
  config="$merged_path"
fi
# Plain KEY=VALUE lines; the launch reads only the keys below.
declare -A cfg
while IFS='=' read -r key value; do
  release_known_key "$key" || release_die "unknown configuration key: $key"
  cfg[$key]="$value"
done < <(grep -E '^[A-Z_0-9]+=' "$config")
[[ -z "${SPARK_COUNT:-}" ]] || cfg[SPARK_COUNT]="$SPARK_COUNT"
[[ -z "${SPARK_HOSTS:-}" ]] || cfg[SPARK_HOSTS]="$SPARK_HOSTS"
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
# Owner-indexed plans name entry/head first; CDI exposes physical device order.
dual_owner_media_device() {
  local mode="$1" entry_device="$2" peer_device="$3"
  case "$mode" in
    rtx:0) printf 'rtx:%s' "$entry_device" ;;
    rtx:1) printf 'rtx:%s' "$peer_device" ;;
    rtx:*) release_die "dual encoder owner must be 0 or 1" ;;
    *) printf '%s' "$mode" ;;
  esac
}
# key NEW OLD [DEFAULT]: a renamed key; the pre-rename spelling works for one
# release with a warning.
key() {
  if [[ -n "${cfg[$1]:-}" ]]; then printf '%s' "${cfg[$1]}"
  elif [[ -n "${cfg[$2]:-}" ]]; then echo "warning: config key $2 is deprecated; use $1" >&2; printf '%s' "${cfg[$2]}"
  else printf '%s' "${3:-}"; fi
}
vision="$(get VISION auto)"
audio="$(get AUDIO auto)"
vision_replicas="$(get VISION_REPLICAS 1)"
[[ "$vision_replicas" =~ ^[1-6]$ ]] || release_die "VISION_REPLICAS must be 1..6"
[[ "$vision" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$vision" != auto:* && "$vision" != off:* ) ]] || release_die "VISION must be auto, off, rtx[:gpu] or spark[:rank]"
[[ "$audio" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$audio" != auto:* && "$audio" != off:* ) ]] || release_die "AUDIO must be auto, off, rtx[:gpu] or spark[:rank]"
coordinator_budget="$(get COORDINATOR_GPU_BUDGET_GIB)"
release_validate_coordinator_gpu_budget "$coordinator_budget"
coordinator_budget_args=()
[[ -z "$coordinator_budget" ]] || coordinator_budget_args=(--coordinator-gpu-budget-gib "$coordinator_budget")
# RDMA_BOND_BALANCE: the coordinator's expert QPs connect with RoCE v2 flow labels it chooses,
# so a coordinator port that is an LACP bond carries as many of them on each member: off
# (default: the kernel's per-QP labels, re-rolled at every start), labels (fixed labels, the
# same placement at every start) or probe (labels measured onto alternating members; see
# rust/crates/cuteafd-transport/src/bond.rs). Workers need no setting.
bond_balance="$(get RDMA_BOND_BALANCE off)"
case "$bond_balance" in off|labels|probe) ;; *) release_die "RDMA_BOND_BALANCE must be off, labels or probe" ;; esac
bond_args=()
[[ "$bond_balance" == off ]] || bond_args=(-e "CUTEAFD_RDMA_BOND_BALANCE=$bond_balance")
# HOST_ISA (auto, x86-64-v2 or x86-64-v4): the coordinator daemon's host CPU build;
# auto picks x86-64-v4 on AVX-512 hosts (rust/crates/cuteafd-ffi/src/host_isa.rs).
host_isa="$(get HOST_ISA auto)"
case "$host_isa" in
  auto) ;;
  x86-64-v2|x86-64-v4) bond_args+=(-e "CUTEAFD_HOST_ISA=$host_isa") ;;
  *) release_die "HOST_ISA must be auto, x86-64-v2 or x86-64-v4" ;;
esac
# ---------------------------------------------------------------------------
# Launch features every family shares. Each sets arrays the coordinator (and
# workers) `docker run` expand; DeepSeek V4.1's section calls the same ones.
# ---------------------------------------------------------------------------
# PROBE_DUMP_ROOT=/abs/host/dir: remote benchmark probes (POST /v1/bench/probe; `cuteafd bench
# fidelity run --dump-dir`, scripts/bench/glmf-teacher-kl.py) may stream full-vocabulary rows
# (dump_rows) into new leaves under it (CUTEAFD_PROBE_DUMP_ROOT, mounted at the same path); unset,
# remote probes cannot write rows.
shared_probe_args() {
probe_args=()
probe_root="$(get PROBE_DUMP_ROOT)"
if [[ -n "$probe_root" ]]; then
  [[ "$probe_root" == /* && -d "$probe_root" ]] || { echo "PROBE_DUMP_ROOT must be an existing absolute directory" >&2; exit 2; }
  release_require_writable_state "$probe_root"
  probe_args=(-v "$probe_root:$probe_root" -e "CUTEAFD_PROBE_DUMP_ROOT=$probe_root")
fi
}
# SPECULATION_TRACE=/abs/host/file.jsonl: the per-cycle speculation trace
# (CUTEAFD_SPECULATION_TRACE, written by serve-glm and serve-qwen4; read by
# scripts/qualify/glm5/glm-draft-trace.py and qualify/qwen4/qwen4-draft-trace.py).
# COORDINATOR_TRACE is its pre-rename key; images older than the rename read
# the family variables, which are set too for one release.
shared_trace_args() {
trace_args=()
# Experimental shared policy changes stay off until emitted-throughput gates pass.
for policy_key in DRAFT_COST_BUCKETS DRAFT_CONFIDENCE DRAFT_SURVIVAL_LENGTHS; do
  # DRAFT_SURVIVAL_LENGTHS (every shared-policy family, GLM 5.3 included):
  # verify drafts while calibrated survival >= 0.5 instead of pricing rows.
  if [[ $family =~ ^(glm5|deepseek_v4)$ && $policy_key != DRAFT_SURVIVAL_LENGTHS && -n "$(get "$policy_key")" ]]; then
    label="$family"; [[ $family != glm5 ]] || label="GLM 5.3"
    echo "$policy_key was removed for $label: it always runs the shared resource-priced draft policy; delete the key" >&2
    exit 2
  fi
  case "$(get "$policy_key" off)" in
    on) trace_args+=(-e "CUTEAFD_$policy_key=1") ;;
    off) trace_args+=(-e "CUTEAFD_$policy_key=0") ;;
    *) echo "$policy_key must be on or off" >&2; exit 2 ;;
  esac
done
trace="$(key SPECULATION_TRACE COORDINATOR_TRACE)"
if [[ -n "$trace" ]]; then
  [[ "$trace" == /* ]] || { echo 'SPECULATION_TRACE must be an absolute path' >&2; exit 2; }
  mkdir -p "$(dirname "$trace")"
  release_require_writable_state --file "$trace"
  trace_args+=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_SPECULATION_TRACE=$trace"
    -e "CUTEAFD_GLM_TRACE=$trace" -e "CUTEAFD_QWEN4_TRACE=$trace")
fi
# TARGET_HEAD: every family's BF16 target vocabulary head (rust shared::target_head). Unset or
# tensor (default) runs BF16 tensor cores with FP32 accumulation at every row count; exact is a
# diagnostic (the pedantic FP32 GEMM on CUDA cores). FP8 heads run their own programs. Forwarded
# only when set, so default launch lines are unchanged.
case "$(get TARGET_HEAD)" in
  "") ;;
  tensor|exact) trace_args+=(-e "CUTEAFD_TARGET_HEAD=$(get TARGET_HEAD)") ;;
  *) echo "TARGET_HEAD must be tensor or exact" >&2; exit 2 ;;
esac
}
# THERMAL=on (default): a background thread samples each coordinator GPU through NVML
# (temperature, T.Limit headroom, power and limit, SM clock, clock-event reasons and their
# cumulative throttle seconds) for /v1/stats, /v1/thermal, the console and every bench
# report's per-phase throttle summary; off disables it. A host power governor's state file
# (gpu-powerd's /run/gpu-powerd, when present) is mounted read-only so the report carries it.
# Spark workers sample their own GPU too (shared_worker_env_args).
shared_thermal_args() {
thermal="$(get THERMAL on)"
case "$thermal" in on|off) ;; *) echo "THERMAL must be on or off" >&2; exit 2 ;; esac
thermal_args=(-e "CUTEAFD_THERMAL=$thermal")
[[ ! -d /run/gpu-powerd ]] || thermal_args+=(-v /run/gpu-powerd:/run/gpu-powerd:ro)
}
# The coordinator environment every family reads:
# SPARK_INTAKE: how routed partials reach the coordinator GPU (auto, gpu, pinned
# or host; see rust/crates/cuteafd-daemon/src/shared/spark_intake.rs).
# CONSOLE_TEXT=on (default) lets the live console at / stream generated token text to
# viewers holding the console unlock cookie; CONSOLE_TEXT=off disables it for everyone.
# FP8_EXPERT_PREFILL: how FP8 expert packages run prefill row counts: auto
# (default: wire rows W8A8 with E4M3 x E4M3 gate/up, BF16 rows W8A16), w8a16
# (the former programs) or w8a8 (also quantizes the BF16 rows of experts on the
# coordinator GPU). Spark workers and the coordinator both read it.
# ATTENTION_PLACEMENT (validated per family before this) reaches the server too.
shared_coordinator_env_args() {
intake="$(get SPARK_INTAKE auto)"
case "$intake" in auto|gpu|pinned|host) ;; *) echo "SPARK_INTAKE must be auto, gpu, pinned or host" >&2; exit 2 ;; esac
console_text="$(get CONSOLE_TEXT on)"
case "$console_text" in on|off) ;; *) echo "CONSOLE_TEXT must be on or off" >&2; exit 2 ;; esac
fp8_prefill="$(get FP8_EXPERT_PREFILL auto)"
case "$fp8_prefill" in auto|w8a8|w8a16) ;; *) echo "FP8_EXPERT_PREFILL must be auto, w8a8 or w8a16" >&2; exit 2 ;; esac
attention_plan_rows="$(get ATTENTION_PLAN_ROWS auto)"
if [[ $attention_plan_rows != auto && ! $attention_plan_rows =~ ^[1-9][0-9]*$ ]]; then
  release_die "ATTENTION_PLAN_ROWS must be auto or a positive integer"
fi
shared_env_args=(-e "CUTEAFD_SPARK_INTAKE=$intake"
  -e "CUTEAFD_CONSOLE_TEXT=$([[ $console_text == on ]] && echo true || echo false)"
  -e "CUTEAFD_ATTENTION_PLACEMENT=$attention_placement" -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill")
if [[ $attention_plan_rows != auto ]]; then
  shared_env_args+=(-e "CUTEAFD_ATTENTION_PLAN_ROWS=$attention_plan_rows")
fi
shared_worker_env_args=(-e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill" -e "CUTEAFD_THERMAL=$(get THERMAL on)")
}
shared_bench_nonce_args() {
bench_nonce_env_args=()
[[ -z "${CUTEAFD_BENCH_NONCE_SEED:-}" ]] || bench_nonce_env_args+=(-e "CUTEAFD_BENCH_NONCE_SEED=$CUTEAFD_BENCH_NONCE_SEED")
# Private diagnostics: every CUTEAFD_DIAG_* variable in the launching
# environment reaches the coordinator unchanged (an env-gated hook reads its
# own name). Nothing else passes this way; unset, the argv is unchanged.
while IFS= read -r name; do
  bench_nonce_env_args+=(-e "$name=${!name}")
done < <(compgen -e CUTEAFD_DIAG_ | sort)
}
# PROFILE=switch[,switch...] turns on profiling for one launch, the same way in
# every family, through release-common.sh:
#   census            CUTEAFD_GRAPH_CENSUS=1 (graph capture/replay census)
#   transport-timing  CUTEAFD_PROTOCOL_V2_TCP_TIMING=1 (expert exchange timing)
#   expert-timing     CUTEAFD_EXPERT_TIMING=1 (Spark worker expert compute timing)
#   draft-timing      CUTEAFD_DRAFT_TIMING=1 (drafter device and host launch spans)
#   nsys              the coordinator runs under `nsys profile` (cuda, nvtx, osrt, CUDA
#                     graph nodes; no CPU sampling), collection paused until
#                     `docker exec COORDINATOR nsys start --session=cuteafd-profile`
#                     and written by `... nsys stop --session=cuteafd-profile` (or a
#                     `docker stop`, never `docker rm -f`) to PROFILE_DIR/nsys-HOST.nsys-rep.
#                     wip-cards opens the window after the sweep's first warm wave and
#                     closes it at sweep end: `--set CARD:PROFILE=nsys --levels ...`.
#                     PROFILE_NSYS_GPU_METRICS=all also samples every GPU's clocks,
#                     power and DRAM/SM activity into the same report (an index
#                     list uses nsys's device order, not CUDA's).
#   ncu               the coordinator runs under `ncu`, replaying the first
#                     PROFILE_NCU_COUNT (default 20) launches whose name matches the
#                     required regex PROFILE_NCU_KERNEL, collecting section set
#                     PROFILE_NCU_SET (default full), into PROFILE_DIR/ncu-HOST.ncu-rep,
#                     written when the coordinator stops (`docker stop`). There is
#                     no window: the first matches are usually during readiness.
#                     For steady-state kernels, use the native harnesses.
# nsys and ncu wrap the coordinator only (one of them per launch); Spark workers
# get the other switches. Both run as the host user; they need the driver's
# RmProfilingAdminOnly=0 and no extra capabilities.
# Every profiled container also gets CUTEAFD_PROFILE_HOST (the physical host,
# never the container id). PROFILE_DIR=/abs/dir is created on each host, mounted
# at the same path and passed as CUTEAFD_PROFILE_DIR; writers name files by host
# and rank. wip-cards owns PROFILE_DIR and copies Spark files back.
shared_profile_args() {
release_profile_docker_args "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)"
profile_args=(${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"})
release_profile_wrapper "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)" coordinator \
  "$(get PROFILE_NCU_KERNEL)" "$(get PROFILE_NCU_COUNT)" "$(get PROFILE_NCU_SET)" "$(get PROFILE_NSYS_GPU_METRICS)"
profile_wrapper=(${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"})
profile_wrapper_args=(${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]+"${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}"})
[[ -z "$(get PROFILE_DIR)" ]] || { mkdir -p "$(get PROFILE_DIR)"; release_require_writable_state "$(get PROFILE_DIR)"; }
}
shared_graph_census_args() {
graph_census_args=()
if [[ "${CUTEAFD_GRAPH_CENSUS:-}" == 1 ]]; then
  [[ -n "$wip_layout" ]] || { printf '%s\n' 'Graph census requires a WIP build' >&2; exit 2; }
  graph_census_args+=(-e CUTEAFD_GRAPH_CENSUS=1 -e 'RUST_LOG=info,cuteafd::graph_capture=debug')
  if [[ -n "${CUTEAFD_GRAPH_CENSUS_CONTROL:-}" ]]; then
    [[ "$CUTEAFD_GRAPH_CENSUS_CONTROL" == /* && -f "$CUTEAFD_GRAPH_CENSUS_CONTROL" ]] || { printf '%s\n' 'Graph census control must be an existing absolute file' >&2; exit 2; }
    # Mount its parent so atomic file replacement remains visible to the server.
    graph_census_args+=(-v "$(dirname "$CUTEAFD_GRAPH_CENSUS_CONTROL"):/run/cuteafd-graph-census:ro"
      -e "CUTEAFD_GRAPH_CENSUS_CONTROL=/run/cuteafd-graph-census/$(basename "$CUTEAFD_GRAPH_CENSUS_CONTROL")")
  fi
fi
if [[ -n "${CUTEAFD_GRAPH_EAGER:-}" ]]; then
  [[ -n "$wip_layout" ]] || { printf '%s\n' 'Graph eager override requires a WIP build' >&2; exit 2; }
  [[ "$CUTEAFD_GRAPH_EAGER" == layer_row ]] || { printf '%s\n' 'Graph eager override must be layer_row' >&2; exit 2; }
  [[ "${CUTEAFD_GRAPH_CENSUS:-}" != 1 ]] || { printf '%s\n' 'Graph eager A/B requires census timing disabled' >&2; exit 2; }
  graph_census_args+=(-e CUTEAFD_GRAPH_EAGER=layer_row)
fi
}
# API keys, the live console and usage records: arguments for the serve command
# (api_args) and the coordinator's mounts (api_mount_args), the same in every family.
# The in-server benchmark keeps its history (SQLite) on the host (bench_dir).
shared_api_state() {
local image="$1" api_instance="$2"
bench_dir="$HOME/.cache/cuteafd/bench"
# Share the host's checksum-verified fidelity references with the dashboard
# (no HF re-download, no 429s). Mounted outside /tmp/cuteafd-home: a bind mount
# there makes Docker create root-owned parents, and the non-root coordinator
# can then no longer create its placement handoff under that home.
fidelity_cache="$HOME/.cache/cuteafd/fidelity"
mkdir -p "$fidelity_cache"
mkdir -p "$bench_dir"
release_require_writable_state "$bench_dir"
api_args=() api_mount_args=()
agent_home="${CUTEAFD_AGENT_HOME:-$HOME/.local/share/cuteafd/agent}"
mkdir -p "$agent_home/dsh"; chmod 700 "$agent_home" "$agent_home/dsh"
api_mount_args+=(--mount "type=bind,src=$(readlink -f "$agent_home/dsh"),dst=/run/cuteafd-agent-dsh,readonly"
  -e CUTEAFD_AGENT_TOKEN_FILE=/run/cuteafd-agent-dsh/launch-token -e "CUTEAFD_AGENT_PORT=${CUTEAFD_AGENT_PORT:-3010}")
API_KEY_FILE="$(get API_KEY_FILE "${API_KEY_FILE:-}")"
ENABLE_BENCH="$(get ENABLE_BENCH off)"
release_prepare_api_key "$ENABLE_BENCH" "$api_instance"
usage="$(get USAGE on)"
case "$usage" in on|off) ;; *) echo "USAGE must be on or off" >&2; exit 2 ;; esac
console_supported=0
if release_console_supported "$image" "${wip_layout:+$wip_layout/bin/cuteafd}"; then
  console_supported=1
  release_prepare_console "$api_instance"
  release_require_writable_state "$USAGE_DIR"
  api_mount_args+=(--mount "type=bind,src=$CONSOLE_SECRET_FILE,dst=/run/cuteafd-console-secret,readonly" -v "$USAGE_DIR:/run/cuteafd/usage")
  api_args+=(--console-secret-file /run/cuteafd-console-secret --usage-dir /run/cuteafd/usage --usage "$usage")
fi
if [[ -n "$API_KEY_FILE" ]]; then
  [[ -f "$API_KEY_FILE" && -r "$API_KEY_FILE" ]] || { echo "API_KEY_FILE must name a readable file" >&2; exit 2; }
  API_KEY_FILE="$(readlink -f "$API_KEY_FILE")"
  api_mount_args+=(--mount "type=bind,src=$API_KEY_FILE,dst=/run/cuteafd-api-key,readonly")
  api_args+=(--api-key-file /run/cuteafd-api-key)
  api_mount_args+=(--mount "type=bind,src=$(readlink -f "$(dirname "$API_KEY_FILE")"),dst=/run/cuteafd-key-directory,readonly"
    -e CUTEAFD_API_KEYS_FILE=/run/cuteafd-key-directory/api-keys)
fi
case "$ENABLE_BENCH" in
  on) api_args+=(--enable-bench) ;;
  off) ;;
  *) echo "ENABLE_BENCH must be on or off" >&2; exit 2 ;;
esac
# GATEWAY=off drops the Messages/Responses/Realtime routes; unset keeps the binary default (on).
gateway="$(get GATEWAY "")"
case "$gateway" in on|off) api_args+=(--gateway "$gateway") ;; "") ;; *) echo "GATEWAY must be on or off" >&2; exit 2 ;; esac
}

# ---------------------------------------------------------------------------
# DeepSeek V4.1 (serve-native). Its launch differs from the other families in
# the release loader it validates with (scripts/lib/release-common.sh), its
# coordinator-first placement handoff, Spark TP role attestation, the
# fingerprint (CUTEAFD_RELEASE_CONFIG_SHA256) and readiness on the advertised
# model. Every launch feature shared with the other families (speculation
# trace and policy switches, profile wrappers, graph census, thermal sampler,
# probe dumps, Spark intake, console text, API keys and console, serving
# identity) comes from the shared helpers above, so it reaches V4.1 with no
# V4.1 code. Engram tables need nothing here: they are read from the
# checkpoint under the /hf/hub mount, kept local on raptor by sparknest.
# ---------------------------------------------------------------------------
launch_deepseek_v41() {
# The release loader validates every key (run.sh --set overrides are in $config).
release_load_config "$config"
# V4.1 serves its established ranges until its runtime consumes solver placement.
attention_placement="${ATTENTION_PLACEMENT:-auto}"
case "$attention_placement" in
  auto|heads) ;;
  context|layers) release_die "deepseek_v41 cannot run attention placement $attention_placement: runtime solver placement not implemented; existing ranges stay unchanged" ;;
  *) release_die "ATTENTION_PLACEMENT must be auto, context, layers or heads" ;;
esac
case "$EMBEDDING" in host|gpu) ;; *) release_die "EMBEDDING must be host or gpu" ;; esac
if [[ -n "$wip_slot" ]]; then
  [[ "$wip_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || release_die "invalid WIP slot name: $wip_slot"
  # WIP artifacts run inside the development images with a release-shaped
  # /opt/cuteafd layout bind-mounted from a per-host staging directory.
  COORDINATOR_DOCKER_INFERENCE="$COORDINATOR_DOCKER_DEV"
  SPARK_EXPERT_DOCKER_INFERENCE="$SPARK_EXPERT_DOCKER_DEV"
fi
release_validate_tp2_options
# RTX_GPUS can be overridden above, so re-check the explicit topology layout.
release_validate_spark_topology
topology_explicit=0
release_spark_topology_explicit && topology_explicit=1
spark_tp="$(release_spark_tp)"
spark_ep="$(release_spark_ep)"
# Extra SM121 expert roles the image must have been built with. Legacy TP4EP1
# and the shipped default require none, so a prebuilt v8 image without any role
# label keeps working unchanged.
spark_tp_roles_required=""
if ((topology_explicit)); then
  case "$spark_tp" in
    2) spark_tp_roles_required=tp2 ;;
    3) spark_tp_roles_required=tp3 ;;
    6) spark_tp_roles_required=tp6 ;;
  esac
fi
[[ -z "$HTTP_QUEUE_DEPTH" || ( "$HTTP_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ && "$HTTP_QUEUE_DEPTH" -le 4096 ) ]] || release_die "HTTP_QUEUE_DEPTH must be in 1..4096"
[[ "$HTTP_QUEUE_WAIT_MS" =~ ^[0-9]+$ ]] || release_die "HTTP_QUEUE_WAIT_MS must be non-negative"
[[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
case "$DSPARK" in on|off) ;; *) release_die "DSPARK must be on or off" ;; esac
# DSPARK_DRAFT_LIMIT: dSpark draft width per request, 1..7 (unset: the engine's 5).
dspark_draft_limit="$DSPARK_DRAFT_LIMIT"
[[ -z "$dspark_draft_limit" || "$dspark_draft_limit" =~ ^[1-7]$ ]] ||
  release_die "DSPARK_DRAFT_LIMIT must be in 1..7"
[[ -z "$dspark_draft_limit" || "$DSPARK" == on ]] ||
  release_die "DSPARK_DRAFT_LIMIT requires dSpark"
release_validate_verbs_device_map
# V41_FP8_HEAD: which vocabulary heads run from one resident E4M3 copy: all
# (target and dSpark draft), draft (the draft only; the target keeps its BF16
# head) or off. Unset passes nothing: the engine default is all. Planner and
# server read the same CUTEAFD_V41_FP8_HEAD, so both get it.
v41_fp8_head_args=()
case "${V41_FP8_HEAD:-}" in
  "") ;;
  off|draft|all)
    [[ -z "${CUTEAFD_V41_FP8_HEAD:-}" || "$CUTEAFD_V41_FP8_HEAD" == "$V41_FP8_HEAD" ]] ||
      release_die "V41_FP8_HEAD=$V41_FP8_HEAD conflicts with CUTEAFD_V41_FP8_HEAD=$CUTEAFD_V41_FP8_HEAD in the environment; set one"
    v41_fp8_head_args=(-e "CUTEAFD_V41_FP8_HEAD=$V41_FP8_HEAD") ;;
  *) release_die "V41_FP8_HEAD must be off, draft or all" ;;
esac
case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" =~ ^([0-9]|[1-3][0-9]|40)$ ]] || release_die "RTX_EXPERT_LAYERS must be auto or 0..40"
[[ "$CONCURRENCY" =~ ^([1-9]|1[0-6])$ ]] || release_die "CONCURRENCY must be in 1..16"
[[ "$PREFIX_CACHE_ENTRIES" =~ ^([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8])$ ]] || release_die "PREFIX_CACHE_ENTRIES must be in 0..128"
[[ "$MAX_CONTEXT_TOKENS" =~ ^[1-9][0-9]*$ ]] && ((MAX_CONTEXT_TOKENS <= 1048576)) || release_die "MAX_CONTEXT_TOKENS must be in 1..1048576"
[[ "$MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]] && ((MAX_OUTPUT_TOKENS <= 393216)) || release_die "MAX_OUTPUT_TOKENS must be in 1..393216"
[[ "$PREFILL_BATCH_TOKENS" =~ ^[0-9]+$ ]] && ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) || release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
[[ -z "$KV_POOL_SIZE" || "$KV_POOL_SIZE" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "KV_POOL_SIZE has an invalid unit"
[[ -z "$MEMORY_RESERVATION" || "$MEMORY_RESERVATION" =~ ^[0-9]+([.][0-9]{1,6})?((B|MB|GB|MiB|GiB)|%)$ ]] || release_die "MEMORY_RESERVATION has an invalid unit"
((restart == 0 || dry_run == 0)) || release_die "--restart and --dry-run are mutually exclusive"
release_validate_compact_spark
if ((SPARK_COUNT == 0)); then
  [[ "$RTX_EXPERT_LAYERS" == 40 && "$RTX_GPUS" != 1 ]] ||
    release_die "SPARK_COUNT=0 requires two RTX GPUs and RTX_EXPERT_LAYERS=40"
fi

# Resolve the one SSH option set used by every remote step below: preflight reads,
# expert launch, readiness polling and the EXIT cleanup. Placed after argument,
# config and value validation so a mistyped CUTEAFD_RELEASE_SSH_CONFIG fails before
# any host is contacted, and before the daemon/image checks so --dry-run and every
# action share one answer. Stock resolution stays the default.
release_configure_ssh_transport

for tool in docker ssh curl jq ss nvidia-smi sha256sum python3; do release_need "$tool"; done
docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE (pull the published pair, or build it with a config whose COORDINATOR_DOCKER_INFERENCE names this tag)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
release_resolve_local_model_revision "$hf_home"
release_resolve_coordinator_gpu_identity
profile_mib="$(nvidia-smi --id="$RELEASE_COORDINATOR_GPU_UUID" --query-gpu=memory.total --format=csv,noheader,nounits)"
profile_gib="$(python3 -c 'import sys; print(min(float(sys.argv[1])/1024, float(sys.argv[2]) if sys.argv[2] else float("inf")))' "$profile_mib" "$COORDINATOR_GPU_BUDGET_GIB")"
if python3 -c 'import sys; sys.exit(not(float(sys.argv[1]) <= 32))' "$profile_gib"; then
  if [[ -z "${cfg_overrides[EMBEDDING]:-}" ]] && ! grep -Eq '^[[:space:]]*EMBEDDING[[:space:]]*=[[:space:]]*[^#[:space:]]' "$config"; then
    EMBEDDING=host
  fi
fi
snapshot_rel="hub/models--${RELEASE_MODEL_ID//\//--}/snapshots/$RELEASE_MODEL_REVISION"
model_is_exl3="$(jq -r '.quantization_config.quant_method == "exl3"' "$hf_home/$snapshot_rel/config.json")"
# Multi-family EXL3 images ship exl3-kXX packages per decoder-tier family.
# The deployed checkpoint's resident tiers are [floor(bits), floor(bits)+1]
# for both raw integer-bit publications and staged fractional-bit snapshots.
exl3_family_tag=""
if [[ "$model_is_exl3" == true ]]; then
  # Family resolution is best-effort: without readable bits the remote
  # manifest read falls back to the legacy single-package location.
  exl3_bits="$(jq -er '.bits' "$hf_home/$snapshot_rel/quantize_config.json" 2>/dev/null || jq -er '.quantization_config.bits' "$hf_home/$snapshot_rel/config.json" 2>/dev/null || true)"
  if [[ "$exl3_bits" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    exl3_family_base="${exl3_bits%%.*}"
    exl3_family_tag="k${exl3_family_base}$((exl3_family_base + 1))"
  fi
fi
coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"
[[ "$TP2_DSPARK_EXPERTS" != on || "$DSPARK" == on ]] || release_die "TP2_DSPARK_EXPERTS requires dSpark"
[[ "$TP2_DSPARK_EXPERTS" != on || "$model_is_exl3" != true ]] || release_die "TP2_DSPARK_EXPERTS requires native expert weights"

reclaim_args=()
if ((restart)) && docker container inspect "$coordinator" >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && reclaim_args+=(--reclaim-pid "$pid")
  done < <(docker top "$coordinator" -eo pid 2>/dev/null | tail -n +2 || true)
fi
minimum_expert_layers=1
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" == 0 ]] || minimum_expert_layers="$RTX_EXPERT_LAYERS"
expert_format=native
model_is_nvfp4="$(jq -r '.quantization_config.moe_quant_algo // empty' "$hf_home/$snapshot_rel/config.json" 2>/dev/null || true)"
[[ "$model_is_nvfp4" == "NVFP4" ]] && expert_format=nvfp4
# Replicated groups require native experts; NVFP4 supports pure TP3/4/6.
# EXL3 uses implicit disjoint groups. Reject other modes before service changes.
if ((topology_explicit)); then
  [[ "$model_is_exl3" != true ]] ||
    release_die "explicit SPARK_TP/SPARK_EP requires the native official checkpoint; EXL3 is not supported"
  if [[ "$model_is_nvfp4" == "NVFP4" ]]; then
    [[ "$spark_ep" == 1 && ( "$spark_tp" == 3 || "$spark_tp" == 4 || "$spark_tp" == 6 ) ]] ||
      release_die "NVFP4 supports pure SPARK_TP=3/4/6 SPARK_EP=1, not replicated groups"
  fi
fi
if [[ "$model_is_exl3" == true ]]; then
  case "$exl3_family_tag" in
    k23) expert_format=exl3-k23 ;;
    k34) expert_format=exl3-k34 ;;
    # Unresolvable families keep the conservative native constants; the
    # dual fit model is only exercised for two-RTX launches.
    *) expert_format=native ;;
  esac
fi
# Auto placement must not pick an RTX boundary so low that the remote Spark
# weights cannot fit the device budget. The floor is derived from the actual
# budget and the resolved TP degree, not a hardcoded 20.
if ((topology_explicit)) && [[ "$RTX_EXPERT_LAYERS" == auto ]]; then
  remote_capacity=$((SPARK_DEVICE_BUDGET_BYTES / $(release_spark_layer_bytes "$spark_tp" "$expert_format")))
  ((remote_capacity > 40)) && remote_capacity=40
  topology_min_layers=$((40 - remote_capacity))
  ((topology_min_layers < 1)) && topology_min_layers=1
  ((minimum_expert_layers >= topology_min_layers)) || minimum_expert_layers="$topology_min_layers"
fi
gpu_selection_mode="$RTX_GPUS"
if [[ "$gpu_selection_mode" == auto ]] && python3 -c 'import sys; sys.exit(not(float(sys.argv[1]) <= 32))' "$profile_gib"; then
  gpu_selection_mode=1
fi
compact_selection_args=()
gpu_selection="$(python3 "$repo_root/scripts/launch/select-release-gpus.py" \
  --mode "$gpu_selection_mode" \
  --minimum-expert-layers "$minimum_expert_layers" \
  --primary-uuid "$RELEASE_COORDINATOR_GPU_UUID" \
  --concurrency "$CONCURRENCY" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --retained-turns "$PREFIX_CACHE_ENTRIES" \
  --kv-pool-size "$KV_POOL_SIZE" \
  --memory-reservation "$MEMORY_RESERVATION" \
  --expert-format "$expert_format" \
  "${compact_selection_args[@]}" "${reclaim_args[@]}")"
RELEASE_RTX_GPUS="$(jq -r '.count' <<<"$gpu_selection")"
((SPARK_COUNT != 0 || RELEASE_RTX_GPUS == 2)) || release_die "SPARK_COUNT=0 requires two selected RTX GPUs"
if release_tp2_enabled; then
  ((RELEASE_RTX_GPUS == 2)) || release_die "TP2 options require two selected RTX GPUs; use --rtx-gpus 2"
fi
mapfile -t release_gpu_uuids < <(jq -r '.gpus[].uuid' <<<"$gpu_selection")
mapfile -t release_gpu_indices < <(jq -r '.gpus[].index' <<<"$gpu_selection")
mapfile -t release_gpu_pci < <(jq -r '.gpus[].pci' <<<"$gpu_selection")
gpu_uuid_csv="$(IFS=,; echo "${release_gpu_uuids[*]}")"
gpu_index_csv="$(IFS=,; echo "${release_gpu_indices[*]}")"
gpu_pci_csv="$(IFS=,; echo "${release_gpu_pci[*]}")"
spark_first_layer="$(release_spark_first_layer "$RELEASE_RTX_GPUS" "$RTX_EXPERT_LAYERS")"
# Auto placement is resolved from live GPU memory, including on one RTX.
# Do not admit or start workers against a guessed local/remote boundary.
if [[ "$RTX_EXPERT_LAYERS" == auto ]] ||
   { ((RELEASE_RTX_GPUS == 1)) && [[ "$RTX_EXPERT_LAYERS" != 0 ]]; }; then
  spark_first_layer=runtime-plan
fi
# Admission against the resolved boundary. Weight-only is all the launcher can
# know before the expert service reports its workspace; a pass is not a
# launch-feasibility claim. An auto dual boundary is published by the
# coordinator and re-checked after plan.json is read below.
spark_admission="not-applicable"
if ((topology_explicit)); then
  if [[ "$spark_first_layer" == runtime-plan ]]; then
    spark_admission="PENDING (coordinator placement plan not yet published; weight-only check uses the real dynamic boundary)"
  else
    spark_admission="$(release_validate_spark_weight_admission "$spark_first_layer" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES" "$expert_format")"
  fi
fi
cuteafd_cdi_gpu_args "$gpu_uuid_csv" || exit 2

sparkinfer_commit="$(python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" --source "$repo_root/third_party/sparkinfer" --lock "$repo_root/third_party/sparkinfer.lock.json" --print-revision)"
wip_layout=""
if [[ -n "$wip_slot" ]]; then
  release_wip_slot_instance "$wip_slot"
  release_wip_serving_layout "${INSTANCE:-}"
  wip_coordinator_container="$(release_wip_container coordinator)"
  wip_spark_container="$(release_wip_container spark-expert)"
  wip_layout="$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT"
  release_ensure_dev_image "$COORDINATOR_DOCKER_INFERENCE"
  release_stage_wip_layout "$wip_coordinator_container" "$wip_slot" coordinator "$wip_layout"
  engine_commit="wip-$wip_slot-$(sha256sum "$wip_layout/lib/libcuteafd_native.so" | cut -c1-12)"
else
  engine_commit="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
  image_sparkinfer="$(docker image inspect -f '{{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
  [[ -n "$engine_commit" && "$engine_commit" != '<no value>' ]] || release_die "coordinator image has no engine revision"
  [[ "$image_sparkinfer" == "$sparkinfer_commit" ]] || release_die "coordinator image uses another SparkInfer revision (run ./build.sh)"
fi

# Select and admit the count before touching any serving container or Spark.
count_args=(--available-sparks "$RELEASE_AVAILABLE_SPARKS")
((RELEASE_SPARK_COUNT_EXPLICIT == 0)) || count_args+=(--spark-ranks "$SPARK_COUNT")
# HF_HOME=/hf: the planner resolves draft snapshots under it, like serving.
count_mounts=(-e HF_HOME=/hf ${v41_fp8_head_args[@]+"${v41_fp8_head_args[@]}"} -v "$(readlink -f "$hf_home/hub"):/hf/hub:ro")
if [[ -n "$wip_layout" ]]; then
  count_mounts+=(-v "$wip_layout/bin:/opt/cuteafd/bin:ro" -v "$wip_layout/lib:/opt/cuteafd/lib:ro"
    -v "$wip_layout/share:/opt/cuteafd/share:ro" --entrypoint /opt/cuteafd/bin/cuteafd)
fi
count_pool="${POOL_TOKENS:-auto}"; [[ "$count_pool" != auto ]] || count_pool=0
count_command=()
[[ -n "$wip_layout" ]] || count_command=(cuteafd)
[[ "$RTX_EXPERT_LAYERS" == auto ]] || count_args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
count_args+=(--embedding-placement "$EMBEDDING")
count_prefill=0
((RELEASE_PREFILL_EXPLICIT == 0)) || count_prefill="$PREFILL_BATCH_TOKENS"
[[ "$DSPARK" != off ]] || count_args+=(--native-mtp-layers 0)
count_plan="$(docker run --rm --network none "${count_mounts[@]}" "$COORDINATOR_DOCKER_INFERENCE" \
  "${count_command[@]}" plan "/hf/$snapshot_rel" --vision "$VISION" --audio off \
  --json --layout --require-ready "${count_args[@]}" --rtx "$RELEASE_RTX_GPUS" \
  --coordinator-gpu-budget-gib "$profile_gib" --coordinator-weight-budget-gib "$profile_gib" \
  --pool-tokens "$count_pool" --prefill-rows "$count_prefill" --context-tokens "$MAX_CONTEXT_TOKENS" \
  --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" \
  --spark-budget-gib "$(python3 -c "print($SPARK_DEVICE_BUDGET_BYTES / 2**30)")")"
selected_count="$(jq -er '.spark_ranks' <<<"$count_plan")"
if ((RELEASE_SPARK_COUNT_EXPLICIT)); then
  [[ "$selected_count" == "$SPARK_COUNT" ]] || release_die "planner changed exact Spark count $SPARK_COUNT to $selected_count"
fi
SPARK_COUNT="$selected_count"
PREFILL_BATCH_TOKENS="$(jq -er '.prefill_rows' <<<"$count_plan")"
# Native non-TP4 counts require the explicit ownership transport contract.
if [[ "$model_is_exl3" != true && "$SPARK_COUNT" != 4 && "$SPARK_COUNT" != 0 && "$topology_explicit" == 0 ]]; then
  SPARK_TP="$SPARK_COUNT" SPARK_EP=1
fi
release_validate_spark_topology
topology_explicit=0; release_spark_topology_explicit && topology_explicit=1
spark_tp="$(release_spark_tp)"; spark_ep="$(release_spark_ep)"
spark_tp_roles_required=""
if ((topology_explicit)) && [[ "$spark_tp" != 4 ]]; then spark_tp_roles_required="tp$spark_tp"; fi
[[ -n "$POOL_TOKENS" || -n "$KV_POOL_SIZE" ]] || POOL_TOKENS=auto

expert_capacity="$(release_v41_expert_capacity "$PREFILL_BATCH_TOKENS")" ||
  release_die "cannot resolve expert row capacity"
mapfile -t hosts < <(release_spark_values HOST)
mapfile -t lanes < <(release_spark_values LANE_A)
spark_exl3_identity=""
for host in "${hosts[@]}"; do
  # Every argument below is non-empty on purpose. OpenSSH joins the command
  # arguments into one remote shell line, so an empty argument is elided and
  # every later positional shifts; the optional EXL3 family tag therefore
  # travels as a sentinel and the host name sits ahead of it.
  spark_manifest="$(release_ssh -o ConnectTimeout=10 "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" "$sparkinfer_commit" "$snapshot_rel" "$host" "$model_is_exl3" "${exl3_family_tag:-__none__}" "${wip_slot:-__none__}" "${wip_spark_container:-__none__}" "${WIP_LAYOUT_SLOT:-__none__}" "${WIP_ROOT:-__default__}" "${WIP_INSTANCE:-__none__}" <<'REMOTE'
set -euo pipefail
image="$1"; engine="$2"; sparkinfer="$3"; snapshot_rel="$4"; host="$5"
exl3="$6"; exl3_family="${7:-__none__}"; wip_slot="${8:-__none__}"
wip_spark_container="${9:-cuteafd-spark-expert-wip}"
wip_layout_slot="${10:-$wip_slot}"
wip_root_arg="${11:-__default__}"; wip_instance_arg="${12:-__none__}"
[[ "$wip_instance_arg" != __none__ ]] || wip_instance_arg=
[[ "$wip_root_arg" != __default__ ]] || wip_root_arg="$HOME/.cache/cuteafd/builds/wip${wip_instance_arg:+-$wip_instance_arg}"
wip_cp() { # slot files come from the container when it exists, else from the WIP root
  if docker container inspect "$wip_spark_container" >/dev/null 2>&1; then docker cp "$wip_spark_container:/wip/$1" "$2"
  else [[ -e "$wip_root_arg/${1%/.}" ]] && cp -a "$wip_root_arg/$1" "$2"; fi
}
[[ "$exl3_family" == __none__ ]] && exl3_family=
# Every failure names the host and the check: this block runs over SSH, so a
# bare nonzero exit would otherwise surface as an unexplained transport error.
# Diagnostics go to stderr; stdout stays the EXL3 manifest alone.
die() { echo "spark preflight on $host: $*" >&2; exit 1; }
docker info >/dev/null 2>&1 || die "the Docker daemon is unavailable"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  registry="${image%%/*}"
  [[ "$wip_slot" != __none__ && "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" == localhost ) ]] || die "inference image is missing: $image (pull or distribute it)"
  docker pull "$image" >&2 || die "cannot pull $image"
fi
if [[ "$wip_slot" == __none__ ]]; then
[[ "$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" == "$engine" ]] ||
  die "$image has another engine revision"
fi
image_sparkinfer="$(docker image inspect -f '{{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' "$image")"
if [[ "$wip_slot" == __none__ && "$image_sparkinfer" != "$sparkinfer" ]]; then
  # WIP launches run the shared development image, which has its own rebuild.
  [[ "$wip_slot" == __none__ ]] && remedy="./build.sh, or pull the matching pair" ||
    remedy="scripts/build/build-dev-images.sh, then ./wip.sh --recreate"
  die "$image carries SparkInfer ${image_sparkinfer:-<none>} but this checkout pins $sparkinfer; rebuild with $remedy"
fi
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
[[ -d "$hf_home/$snapshot_rel" ]] || die "model snapshot is missing: $snapshot_rel"
if find "$hf_home/$snapshot_rel" -xtype l -print -quit | grep -q .; then
  die "model snapshot has dangling links: $snapshot_rel"
fi
if [[ "$wip_slot" != __none__ ]]; then
  layout="$HOME/.cache/cuteafd/wip-run/$wip_layout_slot"
  rm -rf "$layout.tmp" && mkdir -p "$layout.tmp/raw"
  wip_cp "slots/$wip_slot/spark-expert/workspace/.cuteafd-wip/." "$layout.tmp/raw/" ||
    die "WIP slot $wip_slot has no spark-expert artifacts in $wip_spark_container or $wip_root_arg"
  wip_cp "slots/$wip_slot/spark-expert/workspace/docker/release-entrypoint.sh" "$layout.tmp/raw/"
  mkdir -p "$layout.tmp/bin" "$layout.tmp/lib" "$layout.tmp/share"
  mv "$layout.tmp/raw/cuteafd" "$layout.tmp/bin/cuteafd"
  mv "$layout.tmp/raw/libcuteafd_native.so" "$layout.tmp/lib/"
  if [[ -d "$layout.tmp/raw/bin-variants" ]]; then cp -a "$layout.tmp/raw/bin-variants/." "$layout.tmp/bin/"; rm -rf "$layout.tmp/raw/bin-variants"; fi
  if [[ -d "$layout.tmp/raw/lib-variants" ]]; then cp -a "$layout.tmp/raw/lib-variants/." "$layout.tmp/lib/"; rm -rf "$layout.tmp/raw/lib-variants"; fi
  [[ ! -d "$layout.tmp/raw/exl3" ]] || mv "$layout.tmp/raw/exl3" "$layout.tmp/lib/exl3"
  mv "$layout.tmp/raw/"* "$layout.tmp/share/"
  mkdir -p "$layout.tmp/source/third_party" "$layout.tmp/source/scripts/build"
  slot_source="slots/$wip_slot/spark-expert/workspace"
  wip_cp "$slot_source/third_party/sparkinfer" "$layout.tmp/source/third_party/"
  wip_cp "$slot_source/third_party/sparkinfer.lock.json" "$layout.tmp/source/third_party/"
  wip_cp "$slot_source/scripts/build/verify-sparkinfer-source.py" "$layout.tmp/source/scripts/build/"
  python3 "$layout.tmp/source/scripts/build/verify-sparkinfer-source.py" \
    --source "$layout.tmp/source/third_party/sparkinfer" --lock "$layout.tmp/source/third_party/sparkinfer.lock.json" >&2 || die "slot SparkInfer verification failed"
  rm -rf "$layout.tmp/raw" "$layout" && mv "$layout.tmp" "$layout"
fi
if [[ "$exl3" == true && "$wip_slot" != __none__ ]]; then
  layout="$HOME/.cache/cuteafd/wip-run/$wip_layout_slot"
  if [ -f "$layout/lib/exl3/exl3-$exl3_family/manifest.json" ]; then cat "$layout/lib/exl3/exl3-$exl3_family/manifest.json"; else cat "$layout/lib/exl3/manifest.json"; fi
elif [[ "$exl3" == true ]]; then
  docker run --rm --network none --entrypoint /bin/sh "$image" -c \
    'if [ -f "/opt/cuteafd/lib/exl3/exl3-'"$exl3_family"'/manifest.json" ]; then cat "/opt/cuteafd/lib/exl3/exl3-'"$exl3_family"'/manifest.json"; else cat /opt/cuteafd/lib/exl3/manifest.json; fi'
fi
REMOTE
)" || release_die "Spark host preflight failed on $host (see the messages above)"
  if [[ "$model_is_exl3" == true ]]; then
    identity="$(release_exl3_package_identity "$sparkinfer_commit" <<<"$spark_manifest")"
    if release_spark_compact_active || [[ "$SPARK_COUNT" == 6 ]]; then
      [[ "$identity" != paired:* ]] ||
        release_die "SPARK_COUNT=$SPARK_COUNT requires disjoint EXL3 packages, not paired TP4"
      release_validate_exl3_compact_variants "$expert_capacity" "$exl3_family_tag" "$SPARK_COUNT" <<<"$spark_manifest"
    fi
    [[ -z "$spark_exl3_identity" || "$spark_exl3_identity" == "$identity" ]] ||
      release_die "Spark EXL3 packages differ across hosts; rebuild/distribute matching images"
    spark_exl3_identity="$identity"
  fi
done

# An explicit TP2/TP3/TP6 topology needs the matching SM121 expert role baked
# into the Spark image. The published universal release pair advertises every
# extra role at once (label `io.cuteafd.spark_tp_roles=tp2;tp3;tp6`, read with the pre-rename
# `io.cuteafd.v41.spark_tp_roles` as a fallback), so one
# published image pair serves all approved topologies and this launch selects the
# role it needs. A prebuilt legacy image carries no role label and keeps working
# for the default TP4EP1 path; an explicit topology is refused here, before any
# service is stopped or replaced.
spark_advertised_roles=""
if [[ -n "$spark_tp_roles_required" ]]; then
  for host in "${hosts[@]}"; do
    if [[ -n "${wip_slot:-}" ]]; then
      # Slots overlay a toolchain image: its label cannot describe their exports.
      advertised_roles="$(release_ssh -o ConnectTimeout=10 "$host" \
        "if docker container inspect '$wip_spark_container' >/dev/null 2>&1; then docker exec -i '$wip_spark_container' python3 - '$wip_slot' '${expert_format:-native}' '$spark_tp_roles_required'; else WIP_SLOTS=\"\$(r=${WIP_ROOT:-}; i=${WIP_INSTANCE:-}; echo \"\${r:-\$HOME/.cache/cuteafd/builds/wip\${i:+-\$i}}\")/slots\" python3 - '$wip_slot' '${expert_format:-native}' '$spark_tp_roles_required'; fi" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

slot = sys.argv[1]
root = pathlib.Path(os.environ.get('WIP_SLOTS', '/wip/slots')) / slot / 'spark-expert'
try:
    meta_bytes = (root / 'META.json').read_bytes()
    assert hashlib.sha256(meta_bytes).hexdigest() == (root / 'FINGERPRINT').read_text().strip(), 'metadata fingerprint mismatch'
    meta = json.loads(meta_bytes)
    assert meta.get('schema') == 1 and meta.get('slot') == slot and meta.get('role') == 'spark-expert', 'wrong slot identity'
    artifacts = root / 'workspace/.cuteafd-wip'
    manifest_bytes = (artifacts / 'V41_EXPERT_TP_AOT.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == meta.get('v41_expert_tp_manifest_sha256'), 'TP manifest hash mismatch'
    manifest = json.loads(manifest_bytes)
    roles = manifest.get('spark_tp_roles')
    assert manifest.get('schema') == 1 and isinstance(roles, list) and all(role in ('tp2', 'tp3', 'tp6') for role in roles), 'invalid TP roles'
    assert roles == meta.get('spark_tp_roles'), 'slot and artifact roles differ'
    assert manifest.get('symbols_verified') is True, 'TP symbols are not verified'
    if sys.argv[2] == 'nvfp4':
        assert sys.argv[3] in manifest.get('nvfp4_spark_tp_roles', []), 'NVFP4 TP role is not built'
    assert hashlib.sha256((artifacts / 'libcuteafd_native.so').read_bytes()).hexdigest() == manifest.get('native_library_sha256'), 'native library hash mismatch'
    print(';'.join(roles))
except (OSError, ValueError, AssertionError) as error:
    sys.exit(f'WIP slot {slot}: {error}')
PY
      )" || release_die "$host cannot verify the expert roles of WIP slot $wip_slot; rebuild the slot with ./wip.sh"
    else
      advertised_roles="$(
        release_ssh -o ConnectTimeout=10 "$host" \
          "docker image inspect -f '{{or (index .Config.Labels \"io.cuteafd.spark_tp_roles\") (index .Config.Labels \"io.cuteafd.v41.spark_tp_roles\")}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
      )" || release_die "$host cannot report the role label of $SPARK_EXPERT_DOCKER_INFERENCE; is the Spark image present on that host?"
    fi
    if [[ ";$advertised_roles;" != *";$spark_tp_roles_required;"* ]]; then
      if [[ -n "${wip_slot:-}" ]]; then
        release_die "$host WIP slot $wip_slot does not advertise required expert role $spark_tp_roles_required (advertised: ${advertised_roles:-<none>}); rebuild with CUTEAFD_WIP_SPARK_TP_ROLES=$spark_tp_roles_required"
      fi
      release_die "$host Spark image does not advertise required expert role $spark_tp_roles_required (advertised: ${advertised_roles:-<none>}); refusing an unbuilt TP$spark_tp topology: use the published universal release pair, or rebuild with CUTEAFD_RELEASE_SPARK_TP_ROLES=$spark_tp_roles_required"
    fi
    if [[ "${expert_format:-native}" == nvfp4 && -z "${wip_slot:-}" ]]; then
      release_ssh -o ConnectTimeout=10 "$host" docker run --rm --network none \
        --entrypoint python3 "$SPARK_EXPERT_DOCKER_INFERENCE" -c '
import hashlib, json, pathlib, sys
root = pathlib.Path("/opt/cuteafd")
manifest = json.loads((root / "share/V41_EXPERT_TP_AOT.json").read_text())
assert manifest.get("symbols_verified") is True, "NVFP4 TP symbols are not verified"
assert sys.argv[1] in manifest.get("nvfp4_spark_tp_roles", []), "NVFP4 TP role is not built"
assert hashlib.sha256((root / "lib/libcuteafd_native.so").read_bytes()).hexdigest() == manifest.get("native_library_sha256"), "NVFP4 library hash mismatch"
' "$spark_tp_roles_required" || release_die "$host lacks verified NVFP4 $spark_tp_roles_required exports; rebuild the Spark image"
    fi
    [[ -n "$spark_advertised_roles" ]] || spark_advertised_roles="$advertised_roles"
  done
fi

# Zero-Spark deployments hold every routed layer on the RTX pair; the daemon
# still requires four peer addresses but never connects to them.
if ((SPARK_COUNT == 0)); then
  peers="127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"
else
  peer_addresses=()
  for lane in "${lanes[@]}"; do peer_addresses+=("$lane:$EXPERT_PORT"); done
  peers="$(IFS=,; echo "${peer_addresses[*]}")"
fi
fingerprint="$(printf '%s\n' "$engine_commit" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$ADDR" "$RELEASE_RTX_GPUS" "$gpu_uuid_csv" "$gpu_pci_csv" "$CONCURRENCY" "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" "$HTTP_QUEUE_WAIT_MS" "$HOST_CACHE_BYTES" "$RTX_EXPERT_LAYERS" "${KV_POOL_SIZE}${POOL_TOKENS:+:planner=$POOL_TOKENS}" "$EMBEDDING" "$VISION" "$AUDIO" "$MEMORY_RESERVATION" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" "$DSPARK" "$DSPARK_DRAFT_POLICY" "${DRAFT_SURVIVAL_LENGTHS:-off}" "${dspark_draft_limit:-auto}" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${CUTEAFD_VERBS_APP_IB_PORT_NUM:-}" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "$SPARK_DEVICE_BUDGET_BYTES" "$spark_first_layer" "$SPARK_COUNT" "$(release_hosts_csv)" "$peers" "$spark_exl3_identity" "spark-topology=${spark_tp}x${spark_ep}:explicit=${topology_explicit}" "v41-spark-tp-roles=${spark_tp_roles_required}" | sha256sum | awk '{print $1}')"
spark_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"
# Launch features shared with every family, validated before a dry run returns
# and before any service is stopped.
shared_coordinator_env_args
shared_bench_nonce_args
shared_trace_args
shared_probe_args
shared_profile_args
shared_graph_census_args
shared_thermal_args

if ((dry_run)); then
  echo "Dry-run checks passed for native V4.1; no services changed."
  echo "  coordinator container: $coordinator"
  echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv"
  echo "  physical GPUs: $gpu_uuid_csv"
  echo "  TP2 attention/query/output/draft experts: $TP2_ATTENTION/$TP2_QUERY_PROJECTION/$TP2_OUTPUT_PROJECTION/$TP2_DSPARK_EXPERTS"
  echo "  Spark ranks: $SPARK_COUNT; hosts: $(release_hosts_csv)"
  echo "  Spark peers: $peers"
  echo "  Spark first routed layer: $spark_first_layer"
  if ((topology_explicit)); then
    echo "  Spark topology: TP=$spark_tp EP=$spark_ep (group=rank/TP, tp_rank=rank%TP; no dummy ranks)"
    while read -r global_rank group_index tp_rank; do
      echo "    global=$global_rank group=$group_index tp_rank=$tp_rank"
    done < <(release_spark_rank_map)
    echo "  Spark weight admission (workspace/staging NOT accounted; not a feasibility claim): $spark_admission"
    [[ -z "$spark_tp_roles_required" ]] ||
      echo "  Spark image must advertise V41 expert role: $spark_tp_roles_required (selected rank advertises: ${spark_advertised_roles:-<unreadable>})"
  fi
  echo "  coordinator memory reservation: ${MEMORY_RESERVATION:-runtime default}"
  echo "  prefill batch tokens: $PREFILL_BATCH_TOKENS; expert capacity: $expert_capacity"
  echo "  dSpark draft: policy=$DSPARK_DRAFT_POLICY limit=${dspark_draft_limit:-5}"
  [[ -z "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" ]] || echo "  RDMA device map: $CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP"
  [[ -z "${CUTEAFD_VERBS_APP_IB_PORT_NUM:-}" ]] || echo "  RDMA IB port: $CUTEAFD_VERBS_APP_IB_PORT_NUM"
  [[ -z "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" ]] || echo "  RDMA execution lanes: $CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES"
  echo "  release identity: $fingerprint"
  [[ -z "$spark_exl3_identity" ]] || echo "  Spark EXL3 package: $spark_exl3_identity"
  exit 0
fi
shared_api_state "$COORDINATOR_DOCKER_INFERENCE" "${INSTANCE:-default}"
if ((restart)); then
  release_stop_services "$coordinator" "$spark_prefix"
  ((restart_all == 0)) || release_stop_all_worker_containers
else
  docker inspect "$coordinator" >/dev/null 2>&1 && release_die "$coordinator already exists; use --restart"
  for i in "${!hosts[@]}"; do
    remote="${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}"
    release_ssh "${hosts[$i]}" "! docker inspect '$remote' >/dev/null 2>&1" || release_die "$remote already exists; use --restart"
  done
fi

ss -ltn "sport = :${ADDR##*:}" 2>/dev/null | tail -n +2 | grep -q . &&
  release_die "API port ${ADDR##*:} is already in use"
for i in "${!hosts[@]}"; do
  release_ssh "${hosts[$i]}" \
    "! ss -ltn 'sport = :$EXPERT_PORT' 2>/dev/null | tail -n +2 | grep -q ." ||
    release_die "${hosts[$i]}:$EXPERT_PORT is already in use; stop the development worker first"
done

cleanup() {
  docker rm -f "$coordinator" >/dev/null 2>&1 || true
  for i in "${!hosts[@]}"; do release_ssh "${hosts[$i]}" "docker rm -f '${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}' >/dev/null 2>&1 || true" || true; done
}
trap cleanup EXIT

placement_directory=
# Start the coordinator first whenever it can retain routed experts. Its live
# memory plan tells every worker which layers to skip; local expert loading can
# then overlap remote loading. Explicit all-remote single-RTX launches need no
# handoff. Planning uses only the local device and never connects to a worker.
if ((RELEASE_RTX_GPUS == 2)) || [[ "$RTX_EXPERT_LAYERS" != 0 ]]; then
  placement_directory=/tmp/cuteafd-home/placement
fi

# Optional RDMA tuning values travel to both roles only when the operator sets
# them, so a multi-homed six-rank launch can pin the rail without changing any
# default. Values were format-checked above by release_validate_verbs_device_map.
rdma_env_args=("${bond_args[@]}")
# Optional V4.1 coordinator switches from the operator's environment, forwarded only when set.
# (CONSOLE_TEXT and THERMAL are configuration keys, as for every family.)
for switch_name in CUTEAFD_STAGE_CHAIN CUTEAFD_WINDOW_BATCH CUTEAFD_TP2_TOKEN_SUMS CUTEAFD_V41_DEVICE CUTEAFD_V41_DEVICE_LANES CUTEAFD_SPARK_WRITE CUTEAFD_V41_FP8_HEAD CUTEAFD_V41_STAGING_FENCE; do
  # The V41_FP8_HEAD key (validated above) takes the place of its environment spelling.
  [[ "$switch_name" != CUTEAFD_V41_FP8_HEAD || ${#v41_fp8_head_args[@]} == 0 ]] || continue
  [[ -z "${!switch_name:-}" ]] || rdma_env_args+=(-e "$switch_name=${!switch_name}")
done
rdma_env_args+=(${v41_fp8_head_args[@]+"${v41_fp8_head_args[@]}"})
# RDMA_BOND_BALANCE (off, labels or probe): the coordinator's RoCE flow labels on a
# bonded port; see the generic launch below.
case "${RDMA_BOND_BALANCE:-off}" in
  off) ;;
  labels|probe) rdma_env_args+=(-e "CUTEAFD_RDMA_BOND_BALANCE=$RDMA_BOND_BALANCE") ;;
  *) release_die "RDMA_BOND_BALANCE must be off, labels or probe" ;;
esac
for rdma_env_name in CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP CUTEAFD_VERBS_APP_IB_PORT_NUM CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES; do
  [[ -n "${!rdma_env_name:-}" ]] && rdma_env_args+=(-e "$rdma_env_name=${!rdma_env_name}")
done


wip_mount_args=()
if [[ -n "$wip_layout" ]]; then
  wip_mount_args=(
    -v "$wip_layout/bin:/opt/cuteafd/bin:ro" -v "$wip_layout/lib:/opt/cuteafd/lib:ro"
    -v "$wip_layout/share:/opt/cuteafd/share:ro"
    -v "$wip_layout/source:/source:ro"
    -e PYTHONPATH=/source/third_party/sparkinfer
    -e "PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so
    --entrypoint /opt/cuteafd/share/release-entrypoint.sh
  )
fi

# Reuse the admitted count, shape and expert placement for the cold tower.
# Never re-plan a different local prefix or emit a second selection diagnostic.
encoder_rank=-1
encoder_hash=
encoder_port=$((EXPERT_PORT + 100))
vision_args=()
if [[ "$VISION" != off ]]; then
  encoder_plan="$count_plan"
  encoder_kind="$(jq -r '.encoder.kind.kind // "off"' <<<"$encoder_plan")"
  encoder_hash="$(jq -r '.encoder_plan_hash' <<<"$encoder_plan")"
  case "$encoder_kind" in
    spark)
      encoder_rank="$(jq -er '.encoder.kind.rank' <<<"$encoder_plan")"
      encoder_host="${hosts[$encoder_rank]}"
      encoder_peer="${lanes[$encoder_rank]}:$encoder_port"
      vision_args=(--vision-peers "$encoder_peer" --encoder-plan-hash "$encoder_hash" --encoder-revision "$RELEASE_MODEL_REVISION") ;;
    rtx)
      [[ "$(jq -er '.encoder.kind.gpu' <<<"$encoder_plan")" == 0 ]] ||
        release_die "V4.1 local vision currently requires RTX0; use VISION=rtx:0 or auto" ;;
    *) release_die "vision placement unavailable; use VISION=off or provide tower headroom" ;;
  esac
  echo "  vision placement: $encoder_kind rank=$encoder_rank hash=$encoder_hash"
fi

start_coordinator() {
echo "== starting native RTX coordinator =="
local -a args=(--vision "$VISION" --audio "$AUDIO" serve-native "${vision_args[@]}" --snapshot "/hf/$snapshot_rel" --native-lib /opt/cuteafd/lib/libcuteafd_native.so --peers "$peers" --rtx-gpus "$RELEASE_RTX_GPUS" --embedding-placement "$EMBEDDING" --listen "$ADDR" --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --max-context-tokens "$MAX_CONTEXT_TOKENS" --max-output-tokens "$MAX_OUTPUT_TOKENS")
[[ -z "${COORDINATOR_GPU_BUDGET_GIB:-}" ]] || args+=(--coordinator-gpu-budget-gib "$COORDINATOR_GPU_BUDGET_GIB")
args+=("${api_args[@]}")
args+=(--http-queue-depth "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" --http-queue-wait-ms "$HTTP_QUEUE_WAIT_MS")
[[ "$RTX_EXPERT_LAYERS" == auto ]] || args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
[[ "$HOST_CACHE_BYTES" == 0 ]] || args+=(--host-cache-bytes "$HOST_CACHE_BYTES")
[[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
if [[ -n "${POOL_TOKENS:-}" ]]; then
  planner_tokens="$POOL_TOKENS"; [[ "$planner_tokens" != auto ]] || planner_tokens=0
  args+=(--pool-tokens "$planner_tokens")
fi
[[ -z "$MEMORY_RESERVATION" ]] || args+=(--memory-reservation "$MEMORY_RESERVATION")
[[ "$DSPARK" != on ]] || args+=(--dspark)
[[ "$DSPARK" != on || "$DSPARK_DRAFT_POLICY" != full ]] || args+=(--dspark-fixed)
[[ -z "$dspark_draft_limit" ]] || args+=(--dspark-draft-limit "$dspark_draft_limit")
[[ "$TP2_ATTENTION" != on ]] || args+=(--tp2-attention)
[[ "$TP2_QUERY_PROJECTION" != on ]] || args+=(--tp2-query-projection)
[[ "$TP2_OUTPUT_PROJECTION" != on ]] || args+=(--tp2-output-projection)
[[ "$TP2_DSPARK_EXPERTS" != on ]] || args+=(--tp2-dspark-experts)
[[ "$spark_exl3_identity" != paired:* ]] || args+=(--exl3-paired-tp4)
# Explicit replicated topology is described to the coordinator once, and each
# worker derives group=rank/TP and tp_rank=rank%TP from the same two flags.
if ((topology_explicit)); then
  args+=(--spark-tp "$spark_tp" --spark-ep "$spark_ep")
fi
[[ -z "$placement_directory" ]] || args+=(--placement-directory "$placement_directory")
release_serving_docker_args
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$coordinator" --restart no "${CUTEAFD_CDI_ARGS[@]}" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  -e "CUDA_VISIBLE_DEVICES=$gpu_uuid_csv" \
  -e "CUTEAFD_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=${RUST_LOG:-info}" \
  -e "CUTEAFD_V41_IMAGE_ADMISSIONS=${CUTEAFD_V41_IMAGE_ADMISSIONS:-2}" \
  "${rdma_env_args[@]}" "${bench_nonce_env_args[@]}" "${trace_args[@]}" "${graph_census_args[@]}" ${profile_args[@]+"${profile_args[@]}"} \
  "${shared_env_args[@]}" "${thermal_args[@]}" "${probe_args[@]}" \
  "${wip_mount_args[@]}" \
  "${api_mount_args[@]}" -e "CUTEAFD_IMAGE=$COORDINATOR_DOCKER_INFERENCE" -v "$bench_dir:/run/cuteafd/bench" -v "$fidelity_cache:/run/cuteafd/fidelity" -e CUTEAFD_FIDELITY_CACHE=/run/cuteafd/fidelity \
  ${profile_wrapper_args[@]+"${profile_wrapper_args[@]}"} \
  -v "$(readlink -f "$hf_home/hub"):/hf/hub:ro" "$COORDINATOR_DOCKER_INFERENCE" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd "${args[@]}" >/dev/null
}
deadline=$((SECONDS + ${CUTEAFD_RELEASE_READY_TIMEOUT_SECONDS:-900}))
if [[ -n "$placement_directory" ]]; then
  start_coordinator
  until placement_plan="$(docker exec "$coordinator" cat "$placement_directory/plan.json" 2>/dev/null)"; do
    [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 100 "$coordinator" >&2 || true; release_die "coordinator exited before placement publication"; }
    ((SECONDS < deadline)) || release_die "timed out waiting for coordinator placement"
    sleep 1
  done
  # A failed P2P startup probe may reduce a selected two-GPU layout to one.
  # Never accept an expansion beyond the GPUs admitted by this launch.
  spark_first_layer="$(jq -er --argjson gpus "$RELEASE_RTX_GPUS" --arg requested "$RTX_EXPERT_LAYERS" '
    select(.version == 1)
    | select((.rtx_gpus | type) == "number" and (.rtx_gpus == 1 or .rtx_gpus == 2) and .rtx_gpus <= $gpus)
    | select((.nonce | type) == "string" and (.nonce | length) > 0)
    | select((.rtx_expert_layers | type) == "number")
    | select(.rtx_expert_layers == (.rtx_expert_layers | floor) and .rtx_expert_layers >= 0 and .rtx_expert_layers <= 40)
    | select(.rtx_expert_layers > 0 or .rtx_gpus == 1)
    | select($requested == "auto" or .rtx_expert_layers == ($requested | tonumber))
    | select(.spark_first_layer == ([.rtx_expert_layers, 39] | min))
    | .spark_first_layer' <<<"$placement_plan")" || release_die "invalid coordinator placement plan"
  echo "  runtime placement: RTX GPUs $(jq -r '.rtx_gpus' <<<"$placement_plan"), RTX layers $(jq -r '.rtx_expert_layers' <<<"$placement_plan"); Spark first layer $spark_first_layer"
  if ((topology_explicit)); then
    # Re-check against the boundary the coordinator actually published; this is
    # the dynamic value, not the auto placeholder.
    spark_admission="$(release_validate_spark_weight_admission "$spark_first_layer" "$spark_tp" "$SPARK_DEVICE_BUDGET_BYTES" "$expert_format")"
    echo "  runtime Spark weight admission (workspace/staging NOT accounted): $spark_admission"
  fi
fi

echo "== starting native Spark experts =="
pids=()
for i in "${!hosts[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  # ssh joins its arguments into one remote command line, which drops empty
  # arguments and shifts every later position; quote each one explicitly.
  remote_args=("$SPARK_EXPERT_DOCKER_INFERENCE" "$remote" "$i" "$expert_capacity" "$SPARK_DEVICE_BUDGET_BYTES" "$EXPERT_PORT" "$snapshot_rel" "$fingerprint" "$spark_first_layer" "$SPARK_COUNT" "$topology_explicit" "$spark_tp" "$spark_ep" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${CUTEAFD_VERBS_APP_IB_PORT_NUM:-}" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "${RUST_LOG:-info}" "${WIP_LAYOUT_SLOT:-${wip_slot:-__none__}}")
  if ((encoder_rank >= 0)); then
    remote_args+=("$encoder_rank" "$encoder_port" "$encoder_hash" "$RELEASE_MODEL_REVISION")
  fi
  # Docker options for this worker follow a sentinel, after every positional argument:
  # PROFILE switches (absent unless set) and the shared worker environment.
  release_profile_docker_args "${PROFILE:-}" "${PROFILE_DIR:-}" "$host"
  remote_args+=(--profile-args ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"} "${shared_worker_env_args[@]}")
  release_ssh "$host" "$(declare -f release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args); export -f release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args; bash -s -- $(printf '%q ' "${remote_args[@]}")" <<'REMOTE' &
set -euo pipefail
image="$1"; name="$2"; rank="$3"; capacity="$4"; budget="$5"; port="$6"; snapshot_rel="$7"; fingerprint="$8"; first_layer="$9"; world="${10}"
# Defaults keep a legacy invocation (ten positional arguments) valid.
topology_explicit="${11:-0}"; topology_tp="${12:-}"; topology_ep="${13:-}"
topology_args=()
[[ "$topology_explicit" != 1 ]] || topology_args=(--spark-tp "$topology_tp" --spark-ep "$topology_ep")
# Optional RDMA tuning, forwarded only when the operator set it.
rdma_env="${14:-}"; ib_port="${15:-}"; execution_lanes="${16:-}"
# The operator's RUST_LOG reaches the worker too (a quoted heredoc does not
# see the caller's environment).
rust_log="${17:-info}"
wip_slot="${18:-__none__}"
wip_layout_slot="$wip_slot"
encoder_args=()
if [[ "$rank" == "${19:--1}" ]]; then
  encoder_args=(--encoder --encoder-listen "0.0.0.0:${20}" --encoder-plan-hash "${21}" --encoder-revision "${22}" --encoder-max-tokens 1024)
fi
wip_args=()
if [[ "$wip_slot" != __none__ ]]; then
  layout="$HOME/.cache/cuteafd/wip-run/$wip_layout_slot"
  wip_args=(-v "$layout/bin:/opt/cuteafd/bin:ro" -v "$layout/lib:/opt/cuteafd/lib:ro"
    -v "$layout/share:/opt/cuteafd/share:ro"
    -v "$layout/source:/source:ro" -e PYTHONPATH=/source/third_party/sparkinfer
    -e "PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so
    --entrypoint /opt/cuteafd/share/release-entrypoint.sh)
fi
profile_args=()
for ((profile_at = 1; profile_at <= $#; profile_at++)); do
  [[ "${!profile_at}" != --profile-args ]] || { profile_args=("${@:profile_at+1}"); break; }
done
for ((profile_at = 0; profile_at + 1 < ${#profile_args[@]}; profile_at++)); do
  [[ "${profile_args[profile_at]}" != -v ]] || { mkdir -p "${profile_args[profile_at+1]%%:*}"; release_require_writable_state "${profile_args[profile_at+1]%%:*}"; }
done
rdma_args=()
[[ -z "$rdma_env" ]] || rdma_args+=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$rdma_env")
[[ -z "$ib_port" ]] || rdma_args+=(-e "CUTEAFD_VERBS_APP_IB_PORT_NUM=$ib_port")
[[ -z "$execution_lanes" ]] || rdma_args+=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES=$execution_lanes")
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
# A fixed INFO default makes the worker's structured startup evidence
# (`rank`/`world`/`role`/`intermediate`) observable; without it EnvFilter is
# ERROR and the readiness line never reaches the container log. This adds no
# positional argument, so the worker argument contract is unchanged.
release_serving_docker_args
cuteafd_cdi_gpu_args all || exit 2
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$name" --restart no "${CUTEAFD_CDI_ARGS[@]}" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e "CUTEAFD_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=$rust_log" "${rdma_args[@]}" "${wip_args[@]}" ${profile_args[@]+"${profile_args[@]}"} -v "$(readlink -f "$hf_home/hub"):/hf/hub:ro" "$image" cuteafd expertd-native --snapshot "/hf/$snapshot_rel" --native-lib /opt/cuteafd/lib/libcuteafd_native.so --rank "$rank" --world "$world" --rdma-endpoints 2 --capacity "$capacity" --device-budget-bytes "$budget" --first-layer "$first_layer" --listen "0.0.0.0:$port" "${topology_args[@]}" "${encoder_args[@]}" >/dev/null
REMOTE
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in "${!hosts[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  until release_ssh "$host" "test \"\$(docker inspect -f '{{.State.Status}}' '$remote' 2>/dev/null)\" = running && timeout 1 bash -c '</dev/tcp/127.0.0.1/$EXPERT_PORT'" 2>/dev/null; do
    ((SECONDS < deadline)) || { release_ssh "$host" "docker logs --tail 100 '$remote'" >&2 || true; release_die "$host native expert did not become ready"; }
    sleep 1
  done
done

# Loading leaves checkpoint pages cached on every Spark (sparknest passthrough), and GB10
# CUDA allocations do not reclaim page cache: drop it once every rank is resident.
# CUTEAFD_SPARK_DROP_PAGE_CACHE=0 keeps it.
if ((${#hosts[@]} > 0)) && [[ "${CUTEAFD_SPARK_DROP_PAGE_CACHE:-1}" != 0 ]] && command -v nest >/dev/null; then
  drop_hosts=(); for host in "${hosts[@]}"; do drop_hosts+=(--host "$host"); done
  nest drop-caches "${drop_hosts[@]}" >/dev/null || echo "warning: could not drop Spark page caches after loading" >&2
fi

# Acknowledge the published boundary for every handoff launch, including the
# single-RTX explicit-topology case the coordinator now publishes. Only a launch
# with no handoff starts the coordinator at this point.
if [[ -n "$placement_directory" ]]; then
  docker exec "$coordinator" sh -c 'cp "$1/plan.json" "$1/.ready-pending" && mv "$1/.ready-pending" "$1/ready.json"' sh "$placement_directory"
else
  start_coordinator
fi
api_url="http://127.0.0.1:${ADDR##*:}"
until curl -fsS "$api_url/health" >/dev/null 2>&1 &&
  release_api_advertises_native_model "$api_url" "$RELEASE_NATIVE_API_MODEL_ID"; do
  [[ "$(docker inspect -f '{{.State.Status}}' "$coordinator" 2>/dev/null)" == running ]] || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native coordinator exited during startup"; }
  ((SECONDS < deadline)) || { docker logs --tail 200 "$coordinator" >&2 || true; release_die "native API did not become ready"; }
  sleep 1
done
trap - EXIT
echo "CUTEAFD native API is ready at $api_url/v1/"
if ((console_supported)); then release_print_console_link "http://$(hostname):${ADDR##*:}"; fi
echo "  API model: $RELEASE_NATIVE_API_MODEL_ID"
echo "  checkpoint: $RELEASE_MODEL_ID@$RELEASE_MODEL_REVISION"
echo "  RTX layout: $RELEASE_RTX_GPUS GPU(s), host indices $gpu_index_csv ($gpu_uuid_csv)"
echo "  cache: FP4 compressed source, FP8 SWA, FP4 index"
echo "  concurrency: $CONCURRENCY; retained turns: $PREFIX_CACHE_ENTRIES"
echo "  context/output: $MAX_CONTEXT_TOKENS/$MAX_OUTPUT_TOKENS; dSpark: $DSPARK"
}

# The family: run.sh passes --family; a direct call reads the checkpoint's own config.json.
detected_family="$(release_config_family "$config")"
[[ -n "$family" ]] || family="$detected_family"
if [[ "$family" == deepseek_v41 ]]; then
  [[ "$detected_family" == deepseek_v41 ]] ||
    release_die "--family deepseek_v41 does not match the checkpoint ($detected_family)"
  attention_placement="$(get ATTENTION_PLACEMENT auto)"
  launch_deepseek_v41
  exit 0
fi
((dry_run == 0)) || release_die "--dry-run validates DeepSeek V4.1 launches only; $family checkpoints print their argv in the script tests"
# Validate the name before it is used to identify allocations during admission.
instance="$(get INSTANCE)"
[[ -z "$instance" || "$instance" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] || { echo "INSTANCE must be [A-Za-z0-9_.-]" >&2; exit 2; }
coordinator_name="cuteafd-coordinator${instance:+-$instance}"
model="$(get MODEL_ID)"
revision="$(get MODEL_REVISION)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
hub="$(readlink -f "$hf_home/hub")"
root="$hub/models--${model//\//--}"
[[ -n "$revision" ]] || revision="$(<"$root/refs/main")"
snapshot="/hf/hub/models--${model//\//--}/snapshots/$revision"
[[ -d "$root/snapshots/$revision" ]] || { echo "missing snapshot $model@$revision" >&2; exit 1; }
# Family, the serving command and the layers with routed experts.
described="$(python3 "$repo_root/scripts/lib/checkpoint-family.py" "$root/snapshots/$revision/config.json")" || exit 2
read -r detected model_type first_layer last_layer <<<"$described"
[[ -z "$family" || "$family" == "$detected" ]] ||
  { echo "--family $family does not match the checkpoint ($detected, model_type $model_type)" >&2; exit 2; }
family="$detected"
attention_placement="$(get ATTENTION_PLACEMENT auto)"
# auto is the family's recorded default (GLM 5.3 and GLM 5.3 Flash: context where the image
# carries its programs); the planner and the server resolve it. Explicit modes run only where an
# executor exists.
case "$attention_placement:$family" in
  auto:*|heads:*|context:glm5|context:glm5_flash|context:deepseek_v4|layers:qwen4) ;;
  # V4 layers is dev-only: a WIP slot may pass it to the daemon, which serves it only
  # when built with the dev-v4-layers Cargo feature and refuses it otherwise.
  layers:deepseek_v4) [[ -n "$wip_slot" ]] ||
    release_die "deepseek_v4 attention placement layers is a dev-only executor; run it from a WIP slot built with CUTEAFD_WIP_CARGO_FEATURES=dev-v4-layers" ;;
  context:*|layers:*) release_die "$family cannot run attention placement $attention_placement: executor not implemented; only heads is qualified" ;;
  *) release_die "ATTENTION_PLACEMENT must be auto, context, layers or heads" ;;
esac
case "$family" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm5) serve=serve-glm ;;
  glm5_flash) serve=serve-glmf ;;
  mimo_v2) serve=serve-mimo ;;
  qwen4) serve=serve-qwen4 ;;
  *) echo "run-family.sh serves DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 and Qwen 3.8 checkpoints, not $family (./run.sh serves DeepSeek V4.1)" >&2; exit 2 ;;
esac
# VISION and AUDIO default to auto for every family, as the release configs
# set them. Qualified MiMo, GLM Flash and Qwen towers are placed Spark-first by
# the encoder plan below; DeepSeek V4 and GLM 5.3 have no tower, so auto starts
# none and serves text only.
# Only snapshots shipping qualified MiMo audio opt into Spark-first auto.
release_validate_verbs_device_map
audio="$(release_resolve_audio_mode "$audio" "$root/snapshots/$revision")"
# Auto/spark placement is resolved by the encoder plan below.
# An explicit count or host list is exact. Unspecified means the entire inventory.
available_sparks=0
for rank in {0..7}; do
  [[ -n "$(get "SPARK_${rank}_HOST")" ]] || break
  available_sparks=$((available_sparks + 1))
done
count_explicit=0
[[ -z "$(get SPARK_COUNT)$(get SPARK_HOSTS)" ]] || count_explicit=1
ranks="$(get SPARK_COUNT "$available_sparks")"
if [[ -n "$(get SPARK_HOSTS)" ]]; then
  IFS=, read -r -a count_hosts <<<"$(get SPARK_HOSTS)"
  ranks="$(get SPARK_COUNT "${#count_hosts[@]}")"
  host_rows="$(release_spark_host_rows "$(get SPARK_HOSTS)" "$ranks")" || exit 2
  while read -r rank host lane_a lane_b; do
    cfg[SPARK_${rank}_HOST]="$host"
    cfg[SPARK_${rank}_LANE_A]="$lane_a"
    cfg[SPARK_${rank}_LANE_B]="$lane_b"
  done <<<"$host_rows"
fi
configured_ranks="$available_sparks"
backend="$(get EXPERT_BACKEND auto)"
case "$backend" in
  auto|spark) ;;
  local)
    ((count_explicit == 0 || ranks == 0)) || release_die "EXPERT_BACKEND=local conflicts with exact SPARK_COUNT/SPARK_HOSTS=$ranks"
    ranks=0 ;;
  *) echo "EXPERT_BACKEND must be auto, local or spark" >&2; exit 2 ;;
esac
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
# --wip SLOT serves a ./wip.sh slot: the development images run its artifacts, staged from
# the WIP containers into a release-shaped /opt/cuteafd layout per host (as ./run.sh --wip
# does for DeepSeek V4.1).
wip_layout="" wip_mount_args=() wip_worker_args=""
if [[ -n "$wip_slot" ]]; then
  [[ "$wip_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo "invalid WIP slot name: $wip_slot" >&2; exit 2; }
  WIP_INSTANCE="${WIP_INSTANCE:-$(get WIP_INSTANCE)}"
  release_wip_slot_instance "$wip_slot"
  release_wip_serving_layout "$instance"
  wip_coordinator_container="$(release_wip_container coordinator)"
  wip_spark_container="$(release_wip_container spark-expert)"
  coordinator_image="$(get COORDINATOR_DOCKER_DEV cuteafd-coordinator-dev)"
  release_ensure_dev_image "$coordinator_image"
  wip_layout="$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT"
  release_stage_wip_layout "$wip_coordinator_container" "$wip_slot" coordinator "$wip_layout"
  wip_mount_args=(-v "$wip_layout/bin:/opt/cuteafd/bin:ro" -v "$wip_layout/lib:/opt/cuteafd/lib:ro"
    -v "$wip_layout/share:/opt/cuteafd/share:ro"
    -v "$wip_layout/source:/source:ro" -e PYTHONPATH=/source/third_party/sparkinfer
    -e "PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so
    --entrypoint /opt/cuteafd/share/release-entrypoint.sh)
  wip_worker_args="-v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/bin:/opt/cuteafd/bin:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/lib:/opt/cuteafd/lib:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/share:/opt/cuteafd/share:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/source:/source:ro -e PYTHONPATH=/source/third_party/sparkinfer \
    -e PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so --entrypoint /opt/cuteafd/share/release-entrypoint.sh"
fi

qwen_exl3=0
qwen_nvfp4=0
qwen_mtp=0
if [[ "$family" == qwen4 ]]; then
  kv_format="$(get KV_FORMAT fp8)"
  case "$kv_format" in bf16|fp8) ;; *) release_die "KV_FORMAT must be bf16 or fp8" ;; esac
  # NVFP4: a ModelOpt publication with 4-bit weight groups (nvidia/Qwen3.8-Flash-Next-NVFP4).
  qwen_features="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("quantization_config", {}); m=q.get("quant_method", q.get("method")); g=q.get("config_groups", {}).values(); print(int(m == "exl3"), int(m == "modelopt" and any(x.get("weights", {}).get("num_bits") == 4 for x in g)), c.get("text_config", c).get("mtp_num_hidden_layers", 0))' "$root/snapshots/$revision/config.json")"
  read -r qwen_exl3 qwen_nvfp4 qwen_mtp <<<"$qwen_features"
fi
layer_args="--first-layer $first_layer"
[[ "$last_layer" == -1 ]] || layer_args+=" --last-layer $last_layer"
# Docker options only the Spark expert workers take (their environment and mounts).
spark_worker_env=""
# Options only the Spark expert workers take.
spark_worker_args=""
placement_handshake=0
if [[ ( "$family" == mimo_v2 || "$family" == glm5_flash ) && "${CUTEAFD_PLACEMENT_HANDSHAKE:-0}" == 1 ]]; then
  placement_handshake=1
  layer_args+=" --placement-handshake"
fi
# Snapshot of a model id (and optional revision) inside the containers.
snapshot_of() {
  local id="$1" rev="$2" dir="$hub/models--${1//\//--}"
  [[ -n "$rev" ]] || rev="$(<"$dir/refs/main")"
  [[ -d "$dir/snapshots/$rev" ]] || { echo "missing snapshot $id@$rev" >&2; return 1; }
  printf '%s' "/hf/hub/models--${id//\//--}/snapshots/$rev"
}
# SPECULATOR picks the drafter. Unset, each family takes the speculator its
# release card ran (examples/configs/*.config; scripts/tests/test_example_configs.py
# keeps the two in step): GLM 5.3 DFlash2; GLM 5.3 Flash its measured best;
# MiMo V2.6 Flash/Pro MOPD the bundled DFlash; DeepSeek V4 dSpark when the
# checkpoint carries it; Qwen local EXL3/NVFP4 MTP3; otherwise off.
#   dflash2  GLM 5.x / GLM 5.3 Flash: the DFlash2 checkpoint SPECULATOR_MODEL_ID
#            (e.g. incoai/GLM-5.3-DFlash2, incoai/GLM-5.3-Flash-DFlash2);
#            MiMo V2.6 Flash/Pro: the snapshot's own dflash/ drafter unless
#            SPECULATOR_MODEL_ID names one (Pro needs SPARK_COUNT=6)
#   mtp      MiMo V2 Flash, Qwen 3.8: the checkpoint's native MTP layers,
#            SPECULATOR_DEPTH drafts (qualified local Qwen default 3; otherwise 1)
#   dspark   DeepSeek V4 (its own drafter); GLM 5.3 Flash: the dSpark
#            checkpoint SPECULATOR_MODEL_ID (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
# Official Flash MOPD's bundled drafter defaults to single-copy FP8 (measured
# separately from its checkpoint BF16 target head/O); other MiMo defaults follow
# runtime weight policy. SPECULATOR_FP8=auto keeps drafter checkpoint format. GLM auto/unset
# drafts in single-copy FP8. on converts, off selects BF16. Pre-rename keys
# (DRAFT_MODEL_ID, DFLASH,
# MTP, DSPARK, DRAFT_FP8) still work for one release.
# GLM 5.3 Flash always drafts with an external speculator: the one measured
# fastest (emitted tok/s: C1/C4 code and an agentic reasoning session, 1 RTX +
# 2 Sparks and 2 RTX + 4 Sparks) for each checkpoint. DFlash2 led on every one
# measured (2026-10-04: wrldsuksgo2mars EXL3 K3.25, nvidia NVFP4, brandonmusic
# tr3 4bpw; agentic 1.3-1.5x dSpark); SPECULATOR=dspark selects the RedHat dSpark.
glm5_flash_speculator() {
  case "$1" in
    *) echo "dflash2 incoai/GLM-5.3-Flash-DFlash2" ;;
  esac
}
# Qualification is for these official checkpoints, not a shape-compatible sibling.
mimo_flash_mopd=0 mimo_bundled_dflash=0
[[ "$family:$model" != mimo_v2:XiaomiMiMo/MiMo-V2.6-Flash-MOPD ]] || mimo_flash_mopd=1
[[ $mimo_flash_mopd == 0 && "$family:$model" != mimo_v2:XiaomiMiMo/MiMo-V2.6-Pro-MOPD ]] || mimo_bundled_dflash=1
# DeepSeek V4 checkpoints carry their dSpark drafter beside the target (mtp.0.*,
# dspark_block_size in config.json); one without it serves without speculation.
dsv4_dspark=0
if [[ $family == deepseek_v4 ]]; then
  dsv4_dspark="$(python3 -c 'import json,sys; print(int(int(json.load(open(sys.argv[1])).get("dspark_block_size") or 0) > 0))' "$root/snapshots/$revision/config.json")"
fi
speculator="$(get SPECULATOR)"
default_drafter=""
if [[ -z "$speculator" ]]; then
  if [[ -n "$(get DRAFT_MODEL_ID)" || "$(get DFLASH off)" == on ]]; then speculator=dflash2
  elif [[ "$(get MTP 0)" != 0 ]]; then speculator=mtp
  elif [[ $family == deepseek_v4 && "$(get DSPARK off)" == on ]]; then speculator=dspark
  else speculator=off; fi
  [[ $speculator == off ]] || echo "warning: DRAFT_MODEL_ID/DFLASH/MTP/DSPARK are deprecated; use SPECULATOR=$speculator" >&2
  if [[ $speculator == off && $family == glm5 && -z ${cfg[SPECULATOR]+set} ]]; then
    speculator=dflash2 default_drafter=incoai/GLM-5.3-DFlash2
    echo "note: GLM 5.3 drafts with DFlash2 ($default_drafter); SPECULATOR=off disables it" >&2
  fi
  if [[ $speculator == off && $family == glm5_flash ]]; then
    read -r speculator default_drafter <<<"$(glm5_flash_speculator "$model")"
    if [[ -d "$hub/models--${default_drafter//\//--}" ]]; then
      echo "note: GLM 5.3 Flash drafts with $speculator ($default_drafter) for $model; SPECULATOR=off disables it" >&2
    else
      echo "warning: GLM 5.3 Flash drafts with $speculator by default but $default_drafter is not downloaded" \
        "(hf download $default_drafter); serving without a drafter" >&2
      speculator=off default_drafter=""
    fi
  fi
  if [[ $speculator == off && $mimo_bundled_dflash == 1 && -z ${cfg[MTP]+set} && -z ${cfg[DFLASH]+set} ]]; then
    speculator=dflash2
    echo "note: $model drafts with its bundled DFlash; SPECULATOR=off disables it" >&2
  fi
  if [[ $speculator == off && $dsv4_dspark == 1 && -z ${cfg[DSPARK]+set} ]]; then
    speculator=dspark
    echo "note: DeepSeek V4 drafts with its bundled dSpark; SPECULATOR=off disables it" >&2
  fi
  # Only the resident local paths are qualified (EXL3, and NVFP4 with its FP8
  # MTP package). Spark workers serve backbone layers, not mtp.layers.0; other
  # expert formats keep their opt-in status. An explicit SPECULATOR=off or
  # legacy MTP=0 disables the family default.
  if [[ $speculator == off && ( $qwen_exl3 == 1 || $qwen_nvfp4 == 1 ) && $qwen_mtp == 1 && $ranks == 0 && -z ${cfg[MTP]+set} ]]; then
    speculator=mtp
    echo "note: Qwen local $([[ $qwen_exl3 == 1 ]] && echo EXL3 || echo NVFP4) drafts with native MTP (default depth 3); SPECULATOR=off disables it" >&2
  fi
fi
case "$family:$speculator" in
  qwen4:mtp) ;; # The MTP layer's experts stay local even with Spark backbone experts.
  *:off|glm5:dflash2|glm5_flash:dflash2|glm5_flash:dspark|mimo_v2:dflash2|mimo_v2:mtp|deepseek_v4:dspark) ;;
  *) echo "SPECULATOR=$speculator does not apply to $family" >&2; exit 2 ;;
esac
draft_args=()
embedding="$(get EMBEDDING gpu)"
default_context=0
default_concurrency=8
if [[ "$family" == mimo_v2 ]]; then
  profile_gpu="$(get COORDINATOR_GPUS "$(get COORDINATOR_GPU 0)")"; profile_gpu="${profile_gpu%%,*}"
  profile_mib="$(nvidia-smi -i "$profile_gpu" --query-gpu=memory.total --format=csv,noheader,nounits)"
  profile_gib="$(python3 -c 'import sys; print(min(float(sys.argv[1])/1024, float(sys.argv[2]) if sys.argv[2] else float("inf")))' "$profile_mib" "$coordinator_budget")"
  if python3 -c 'import sys; sys.exit(not(float(sys.argv[1]) <= 32))' "$profile_gib"; then
    default_concurrency=16
    [[ -n "${cfg[EMBEDDING]:-}" ]] || embedding=host
    echo "MiMo 32 GB profile: logical GPU ${profile_gib} GiB, embedding=$embedding (EMBEDDING overrides), int8 KV, checkpoint-full context unless MAX_CONTEXT_TOKENS overrides" >&2
  fi
fi
case "$embedding" in host|gpu) ;; *) echo "EMBEDDING must be host or gpu" >&2; exit 2 ;; esac
family_args=(--embedding-placement "$embedding")
chat_template_mounts=()
# Vision-only override: no template inference and no changes to text-only prompts.
# Without it, VISION=auto serves a GLM Flash checkpoint whose own chat template cannot mark images
# as text: the encoder plan below turns vision off and says so. An explicit placement still fails.
chat_template_from="$(get CHAT_TEMPLATE_FROM)"
if [[ -n "$chat_template_from" && "$vision" != off ]]; then
  [[ "$family" == glm5_flash ]] || release_die "CHAT_TEMPLATE_FROM currently applies only to GLM Flash vision"
  if [[ -d "$chat_template_from" ]]; then
    chat_template_from="$(readlink -f "$chat_template_from")"
    release_validate_path_setting CHAT_TEMPLATE_FROM "$chat_template_from"
    if release_path_within "$chat_template_from" "$hub"; then
      chat_template_from="/hf/hub${chat_template_from#"$hub"}"
    else
      chat_template_mounts=(-v "$chat_template_from:$chat_template_from:ro")
    fi
  else
    [[ "$chat_template_from" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] &&
      release_path_has_no_dot_segment "$chat_template_from" || release_die "CHAT_TEMPLATE_FROM must be an existing snapshot or ORG/MODEL HF id"
    snapshot_of "$chat_template_from" "" >/dev/null || exit 2
  fi
  family_args+=(--chat-template-from "$chat_template_from")
fi
family_args+=(--vision "$vision" --audio "$audio")
case "$(get FULL_PREFILL_LOGITS off)" in
  on) family_args+=(--full-prefill-logits) ;;
  off) ;;
  *) echo "FULL_PREFILL_LOGITS must be on or off" >&2; exit 2 ;;
esac
dspark_args=()
if [[ $family == mimo_v2 ]]; then
  case "$(get MIMO_WEIGHT_POLICY auto)" in
    auto) ;; # The runtime and planner share the metadata qualifier.
    checkpoint) family_args+=(--weight-policy checkpoint) ;;
    *) echo "MIMO_WEIGHT_POLICY must be auto or checkpoint" >&2; exit 2 ;;
  esac
  for projection in HEAD O_PROJ; do
    mode="$(get "MIMO_FP8_$projection")"
    option=--fp8-head
    [[ $projection != O_PROJ ]] || option=--fp8-o-proj
    case "$mode" in
      ""|auto) ;;
      on) family_args+=("$option" true) ;;
      off) family_args+=("$option" false) ;;
      *) echo "MIMO_FP8_$projection must be auto, on or off" >&2; exit 2 ;;
    esac
  done
fi
case "$speculator" in
  dflash2|dspark)
    drafter="$(key SPECULATOR_MODEL_ID DRAFT_MODEL_ID)"
    [[ -n "$drafter" ]] || drafter="$default_drafter"
    if [[ $speculator == dspark && $family == deepseek_v4 ]]; then
      dspark_args=(--dspark)
    elif [[ -n "$drafter" ]]; then
      draft_snapshot="$(snapshot_of "$drafter" "$(key SPECULATOR_MODEL_REVISION DRAFT_MODEL_REVISION)")" ||
        { [[ -z "$default_drafter" ]] || echo "download it (hf download $drafter) or set SPECULATOR=off" >&2; exit 1; }
      draft_args=(--draft "$draft_snapshot")
    elif [[ $family == mimo_v2 ]]; then
      draft_args=(--draft "$snapshot")
    else
      echo "SPECULATOR=$speculator needs SPECULATOR_MODEL_ID (a ${speculator/dflash2/DFlash2} checkpoint)" >&2; exit 2
    fi ;;
  mtp)
    mtp_depth=1
    [[ ( $qwen_exl3 != 1 && $qwen_nvfp4 != 1 ) || $qwen_mtp != 1 ]] || mtp_depth=3
    family_args+=(--mtp "$(key SPECULATOR_DEPTH MTP "$mtp_depth")") ;;
esac
# SPECULATOR_DRAFTS: adaptive (default) or a fixed draft count per cycle
# (DFlash2 on GLM 5.x, GLM 5.3 Flash and MiMo V2; dSpark on GLM 5.3 Flash), for policy A/B runs.
drafts="$(get SPECULATOR_DRAFTS adaptive)"
if [[ "$drafts" != adaptive ]]; then
  [[ "$drafts" =~ ^[0-9]+$ && ($speculator == dflash2 || $family:$speculator == glm5_flash:dspark) ]] ||
    { echo "SPECULATOR_DRAFTS must be adaptive or a draft count, with SPECULATOR=dflash2 (or dspark on GLM 5.3 Flash)" >&2; exit 2; }
  draft_args+=(--draft-fixed "$drafts")
fi
# Prefix cache (MiMo, GLM 5.3, GLM 5.3 Flash, Qwen 3.8, DeepSeek V4): PREFIX_CACHE_ENTRIES
# snapshots per bank (prompts, turns; 0 = off), HOST_CACHE_BYTES of pinned host memory for
# snapshots the device evicts (e.g. 64GiB; 0 = off). POOL_TOKENS: paged KV tokens shared by
# live sequences and retained snapshots.
case $family in
  mimo_v2|glm5|glm5_flash|qwen4|deepseek_v4)
    family_args+=(--prefix-cache-entries "$(get PREFIX_CACHE_ENTRIES 20)")
    [[ "$(get HOST_CACHE_BYTES 0)" == 0 ]] || family_args+=(--host-cache-bytes "$(get HOST_CACHE_BYTES)") ;;
esac
if [[ ( $family == mimo_v2 || $family == qwen4 || $family == glm5_flash ) && $vision != off ]]; then
  # Text-only frozen daemons can predate the optional media-cache flag.
  [[ -z "$(get MEDIA_CACHE_BYTES)" ]] || family_args+=(--media-cache-bytes "$(get MEDIA_CACHE_BYTES)")
fi
if [[ $family == mimo_v2 ]]; then
  [[ -z "$(get HTTP_QUEUE_DEPTH)" ]] || family_args+=(--http-queue-depth "$(get HTTP_QUEUE_DEPTH)")
  [[ -z "$(get HTTP_QUEUE_WAIT_MS)" ]] || family_args+=(--http-queue-wait-ms "$(get HTTP_QUEUE_WAIT_MS)")
  [[ -z "$(get MIMO_COPY_WINDOWS)" ]] || { echo "MIMO_COPY_WINDOWS was removed: copies only extend agreeing neural drafts; delete the key" >&2; exit 2; }
  [[ "$(get MIMO_PREFIX_DRAFT off)" != on ]] || family_args+=(--mimo-prefix-draft)
  [[ "$(get MIMO_SNAPSHOT_WAIT off)" != on ]] || family_args+=(--mimo-snapshot-wait)
  if [[ "$(get MIMO_HOST_CACHE off)" == on ]]; then
    family_args+=(--mimo-host-cache)
    [[ -n "$(get HOST_CACHE_BYTES)" ]] || family_args+=(--host-cache-bytes auto)
  fi
  [[ -z "$(get MIMO_PREFILL_CHUNK_S)" ]] || family_args+=(--prefill-chunk-s "$(get MIMO_PREFILL_CHUNK_S)")
  # POOL_TOKENS=auto: the largest pool every GPU admits after all fixed costs (up to 2M tokens).
  # Default auto (measured 2026-10-03, MiMo V2.6 Pro 2 RTX + 6: 131072 -> 2,097,152 tokens, C1/C4/8K
  # prefill unchanged); a number pins the pool.
  mimo_pool="$(get POOL_TOKENS auto)"; [[ "$mimo_pool" != auto ]] || mimo_pool=0
  family_args+=(--pool-tokens "$mimo_pool")
  [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || family_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
  # PREFIX_PARTIAL=on: V4.1-style partial reuse (approximate; off = exact restores only).
  family_args+=(--prefix-partial "$(get PREFIX_PARTIAL off)")
  # KV_CACHE: int8 (the engine default: 8-bit full-attention records with FP32 scales per 32
  # dims; SWA rings stay BF16) or bf16.
  [[ -z "$(get KV_CACHE)" ]] || family_args+=(--kv-cache "$(get KV_CACHE)")
  # DECODE_GRAPHS=on: decode/verify steps replay per-layer CUDA graph segments captured at startup
  # (default off: neutral against the host-driven Spark exchange, 0.4-1.2 GiB of graphs).
  case "$(get DECODE_GRAPHS)" in
    "") ;;
    on) family_args+=(--decode-graphs true) ;;
    off) family_args+=(--decode-graphs false) ;;
    *) echo "DECODE_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
# EXPERT_INPUT is opt-in for MiMo's Spark exchange; unset preserves image defaults.
expert_input="$(get EXPERT_INPUT)"
if [[ -n "$expert_input" ]]; then
  [[ "$family" == mimo_v2 ]] || { echo "EXPERT_INPUT applies to MiMo checkpoints" >&2; exit 2; }
  case "$expert_input" in
    fp8|bf16|bf16-decode) family_args+=(--expert-input "$expert_input") ;;
    *) echo "EXPERT_INPUT must be fp8, bf16 or bf16-decode" >&2; exit 2 ;;
  esac
fi
# Qwen 3.8: QWEN_FP8_DECODE=on|off converts the GDN/attention projections to one
# resident E4M3 copy; QWEN_FP8_HEAD=on|off does the same for the head target and
# MTP share. Unset keeps the engine defaults.
if [[ $family == qwen4 ]]; then
  family_args+=(--kv-cache "$kv_format")
  for key in FP8_DECODE:--fp8-decode FP8_HEAD:--mtp-fp8-head; do
    mode="$(get "QWEN_${key%%:*}")"
    case "$mode" in
      "") ;;
      on) family_args+=("${key#*:}" true) ;;
      off) family_args+=("${key#*:}" false) ;;
      *) echo "QWEN_${key%%:*} must be on or off" >&2; exit 2 ;;
    esac
  done
fi
# POOL_TOKENS=auto (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen, DeepSeek V4): the largest pool the GPUs hold after the
# planner's remaining costs (up to 2M tokens). Every family defaults to auto,
# as its release cards ran; a number pins the pool. Qwen's former 32768-token
# pool admitted only seven 4096-output requests, below the default eight
# serving lanes; DeepSeek V4's former engine default was 262144 tokens (v2.0.0
# cards: 650K-1.3M tokens with auto).
if [[ $family =~ ^(glm5|qwen4|deepseek_v4)$ ]]; then
  pool="$(get POOL_TOKENS auto)"
  if [[ "$pool" == auto ]]; then
    pool=0
  fi
  family_args+=(--pool-tokens "$pool")
fi
if [[ -n "$(get COPY_DRAFTS)" || -n "$(get COPY_DRAFT_POLICY)" ]]; then
  label="$family"; [[ $family != glm5 ]] || label="GLM 5.3"
  for removed in COPY_DRAFTS COPY_DRAFT_POLICY; do
    [[ -z "$(get "$removed")" ]] || { echo "$removed was removed for $label: the shared resource-priced draft policy only extends agreeing neural drafts with copies; delete the key" >&2; exit 2; }
  done
fi
# DECODE_SHARE: the share of the time running requests keep while prompts
# prefill (the engine's default 0.2; 0 prefills whole prompts before the next
# step). Keys left unset pass nothing (images older than the options run).
[[ -z "$(get DECODE_SHARE)" ]] || family_args+=(--decode-share "$(get DECODE_SHARE)")
# RTX_EXPERT_LAYERS (DeepSeek V4, GLM 5.3 and MiMo): unset is uniformly pool-first
# auto; max is the explicit experts-first policy leaving at least a 262K pool.
# N, N% or all fix the RTX-resident routed layers, then the KV pool fills the
# remaining budget up to its target (refused below its floor). 0 leaves the
# backbone experts on the Sparks. With two RTX every onboard mode uses TP2
# halves; GPU1 whole-layer expert ranges are no longer supported.
if [[ $serve == serve-dsv4 || $serve == serve-glm || $serve == serve-mimo ]]; then
  local_layers="$(get RTX_EXPERT_LAYERS)"
  onboard_flag=--rtx-expert-layers
  fixed_onboard_flag=--local-expert-layers
  [[ $serve != serve-glm ]] || fixed_onboard_flag=--rtx-expert-layers
  if [[ $serve == serve-mimo ]]; then
    onboard_flag=--onboard
    fixed_onboard_flag=--onboard
  fi
  case "$(get RTX_EXPERT_PEER off)" in
    on) echo "GPU1 whole-layer expert ranges were replaced by TP2 halves (v3 P4); remove RTX_EXPERT_PEER" >&2; exit 2 ;;
    off) ;;
    *) echo "RTX_EXPERT_PEER must be on or off" >&2; exit 2 ;;
  esac
  case "$local_layers" in
    "") ;;
    auto|all|max) family_args+=("$onboard_flag" "$local_layers") ;;
    *%)
      percent="${local_layers%\%}"
      [[ "$percent" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v p="$percent" 'BEGIN { exit !(p <= 100) }' ||
        { echo "RTX_EXPERT_LAYERS must be auto, max, all, N or N% (0..100%)" >&2; exit 2; }
      family_args+=("$onboard_flag" "$local_layers") ;;
    *[!0-9]*) echo "RTX_EXPERT_LAYERS must be auto, max, all, N or N% (0..100%)" >&2; exit 2 ;;
    *) family_args+=("$fixed_onboard_flag" "$local_layers") ;;
  esac
fi
# GLM, GLM Flash, MiMo, Qwen: L2_PREFETCH (off, auto = 3/4 of the L2, or MiB;
# unset: auto for GLM 5.3 and GLM 5.3 Flash, off for MiMo and Qwen) pulls the
# next layer's weights into L2 during each one-lane decode step's Spark exchange; FP8_SCALES (amax, pow2, best) is the scale rule of the FP8
# copies made from BF16 weights at load.
if [[ $serve != serve-dsv4 ]]; then
  [[ -z "$(get L2_PREFETCH)" ]] || family_args+=(--l2-prefetch "$(get L2_PREFETCH)")
  [[ -z "$(get FP8_SCALES)" ]] || family_args+=(--fp8-scales "$(get FP8_SCALES)")
  if [[ ${#draft_args[@]} -gt 0 ]]; then
    if [[ $family == mimo_v2 ]]; then
      [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || family_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
      [[ -z "$(get DRAFT_SEQUENCES)" ]] || family_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
      mimo_draft_default=""
      if [[ $mimo_flash_mopd == 1 && $speculator == dflash2 && -z "$drafter" ]]; then
        mimo_draft_default=on
      fi
      case "$(key SPECULATOR_FP8 DRAFT_FP8 "$mimo_draft_default")" in
        "") ;;
        auto) family_args+=(--draft-representation checkpoint) ;;
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "MiMo SPECULATOR_FP8/DRAFT_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
    elif [[ $family == glm5 || $family == glm5_flash ]]; then
      case "$(key SPECULATOR_FP8 DRAFT_FP8 auto)" in
        auto) ;; # The engine default: single-copy FP8 drafter weights.
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "SPECULATOR_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
      [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || family_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
      [[ -z "$(get DRAFT_SEQUENCES)" ]] || family_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    elif [[ "$(key SPECULATOR_FP8 DRAFT_FP8 on)" == off ]]; then
      family_args+=(--draft-fp8 false)
    fi
  fi
fi
# SERVED_MODEL_ID: the public model id (default: the checkpoint's Hugging Face id).
served_args=()
served="$(get SERVED_MODEL_ID)"
[[ -z "$served" ]] || served_args=(--model-id "$served")
shared_probe_args
shared_trace_args
# Qwen serving defaults to qualified startup graphs; preserve explicit overrides.
if [[ $family == qwen4 ]]; then
  case "$(get QWEN_STARTUP_GRAPHS)" in
    "") ;;
    on) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=1) ;;
    off) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=0) ;;
    *) echo "QWEN_STARTUP_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
[[ -z "$wip_slot" ]] || spark_image="$(get SPARK_EXPERT_DOCKER_DEV cuteafd-spark-expert-dev)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
budget="$(get SPARK_DEVICE_BUDGET_BYTES 107374182400)"
gpu="$(get COORDINATOR_GPU 0)"
# Two coordinator GPUs (RTX_GPUS=auto/2 with COORDINATOR_GPU as V4.1's two-RTX config picks
# the other card, or an explicit COORDINATOR_GPUS=0,1): families with a
# head split (MiMo V2 Flash/Pro, GLM 5.x, GLM 5.3 Flash, DeepSeek V4) split every layer's
# attention heads and dense / shared-expert MLPs over both by default (COORDINATOR_SPLIT=auto
# or heads), one hidden all-reduce per layer over peer memory; experts, router, head and
# drafter stay on the first GPU. COORDINATOR_SPLIT=off serves from the first GPU alone. Auto
# selection uses one GPU for checkpoints without a split and for splits that are opt-in
# (`split_opt_in`: measured not to pay at the reference layouts); an explicit split request
# (RTX_GPUS=2, COORDINATOR_GPUS=a,b or COORDINATOR_SPLIT=heads) fails before containers
# start when the checkpoint has none. The container sees both GPUs in host order.
# COORDINATOR_SPLIT_GPU names the second GPU when only
# COORDINATOR_GPU is set (default the other of 0/1).
rtx_gpus="$(get RTX_GPUS auto)"
case "$rtx_gpus" in auto|1|2) ;; *) echo "RTX_GPUS must be auto, 1 or 2" >&2; exit 2 ;; esac
physical_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
coordinator_gpus="$(get COORDINATOR_GPUS)"
explicit_coordinator_gpus="$coordinator_gpus"
if [[ -z "$coordinator_gpus" ]]; then
  coordinator_gpus="$gpu"
  case "$rtx_gpus" in
    1) ;;
    2|auto)
      other="$(awk -v first="$gpu" '/^[0-9]+$/ && $0 != first {print; exit}' <<<"$physical_gpus")"
      [[ -z "$other" ]] || coordinator_gpus="$gpu,$other" ;;
  esac
fi
[[ "$coordinator_gpus" =~ ^[0-9]+(,[0-9]+)?$ ]] ||
  { echo "COORDINATOR_GPUS must name one or two GPU indices" >&2; exit 2; }
IFS=, read -r -a coordinator_gpus <<<"$coordinator_gpus"
gpu="${coordinator_gpus[0]}"
split="$(get COORDINATOR_SPLIT auto)"
case "$split" in auto|heads|off) ;; *) echo "COORDINATOR_SPLIT must be auto, heads or off" >&2; exit 2 ;; esac
second=""
if [[ ${#coordinator_gpus[@]} -ge 2 ]]; then
  second="${coordinator_gpus[1]}"
elif [[ "$split" == heads ]]; then
  second="$(get COORDINATOR_SPLIT_GPU $((1 - gpu)))"
fi
explicit_split=0
if [[ "$rtx_gpus" == 2 || "$explicit_coordinator_gpus" == *,* || "$split" == heads ]]; then
  explicit_split=1
fi
split_hint=""
split_opt_in=""
case "$family:$model_type" in
  deepseek_v4:*|glm5:*|glm5_flash:*|mimo_v2:mimo_v2|mimo_v2:mimo_v2_flash) ;;
  qwen4:*)
    if [[ "$ranks" == 0 ]]; then
      : # Qualified Spark-free pairs default to native TP2 and whole-layer ownership.
    else
      split_hint="Qwen dual requires a Spark-free local TP2 layout"
    fi ;;
  mimo_v2:*) split_hint="add MiMo head-split attention/projection kernels for $model_type" ;;
  *) split_hint="add coordinator head-split kernels for $model_type" ;;
esac
if [[ -n "$split_opt_in" && "$split" == auto && "$explicit_split" == 0 && -z "$split_hint" ]]; then
  echo "note: $family head split is opt-in ($split_opt_in); auto selected GPU $gpu alone" >&2
  second=""
fi
if [[ "$split" != off && "$explicit_split" == 1 ]]; then
  [[ -z "$split_hint" ]] ||
    echo "note: $family ($model_type) has no head split yet ($split_hint); serving from GPU $gpu alone" >&2
  [[ -n "$second" ]] ||
    { echo "RTX_GPUS=2 requires two physical coordinator GPUs; only GPU $gpu was selected" >&2; exit 2; }
fi
gpus="$gpu"
head_split=0
if [[ -n "$second" && "$split" != off ]]; then
  [[ "$second" =~ ^[0-9]+$ ]] || { echo "COORDINATOR_SPLIT_GPU must be a GPU index" >&2; exit 2; }
  [[ "$second" != "$gpu" ]] || { echo "the second coordinator GPU must differ from the first" >&2; exit 2; }
  if [[ -z "$split_hint" ]]; then
    if [[ "$family" == qwen4 ]]; then
      family_args+=(--rtx-expert-layers "$(get RTX_EXPERT_LAYERS all)")
      [[ "$attention_placement" != heads ]] || release_die "Qwen dual owns whole layers; head-split attention is not implemented"
      attention_placement=layers
    fi
    for selected in "$gpu" "$second"; do
      grep -qx "$selected" <<<"$physical_gpus" ||
        { echo "two-GPU head split requires physical GPU $selected, but nvidia-smi did not report it" >&2; exit 2; }
    done
    lower=$((gpu < second ? gpu : second)) upper=$((gpu < second ? second : gpu))
    gpus="$lower,$upper"
    head_split=1
    family_args+=(--device $((gpu == lower ? 0 : 1)) --split-device $((gpu == lower ? 1 : 0)))
  elif [[ "$explicit_split" != 1 ]]; then
    echo "note: $family ($model_type) has no head split; auto selected GPU $gpu alone" >&2
  fi
fi
if [[ "$family" == qwen4 && "$attention_placement" == layers && "$head_split" != 1 ]]; then
  release_die "Qwen whole-layer attention requires a Spark-free coordinator pair"
fi
# GLM 5.3 Flash: the MLA, dense and shared-expert projections are FP8 only,
# from the official FP8 release (GLM5_FLASH_FP8_MODEL_ID; "off" requires native
# FP8 block tensors in the primary checkpoint, else BF16 ones are quantized to
# 128x128 blocks at load). Resolve precision after the serving split: both
# layouts default to row128 KDA and an FP8 head; the two-GPU head split adds
# token-row KDA output ownership (--kda-output-shard --kda-prefill-expanded),
# which avoids rounding half-K BF16 partials (2026-10-05: C1 +7.6%, paired
# top-1 453 -> 460/512, KL +0.0008). GLM5_FLASH_KDA_SPLIT=partials keeps the
# old split path. Explicit current or legacy keys override either default.
# Each weight has one resident copy; the DFlash2 drafter stays FP8 by default.
# Its MLA pools hold POOL_TOKENS tokens (a key every
# family with a paged KV pool reads). GLM5_FLASH_FP8_PREFILL lists the prefill
# projections that run W8A8 (E4M3 activations per 128-K block): unset = the
# engine default mla,ffn, a list of mla,ffn,kda-in,kda-o / all (kda-* need
# GLM5_FLASH_KDA_FP8 row128/channel), or off (every FP8 weight W8A16). The
# GLMF_* spellings still work for one release.
if [[ $family == glm5_flash ]]; then
  local_layers="$(get RTX_EXPERT_LAYERS)"
  [[ -z "$local_layers" ]] || family_args+=(--rtx-expert-layers "$local_layers")
  fp8_model="$(key GLM5_FLASH_FP8_MODEL_ID GLMF_FP8_MODEL_ID zai-org/GLM-5.3-Flash)"
  if [[ "$fp8_model" != off ]]; then
    fp8_snapshot="$(snapshot_of "$fp8_model" "$(key GLM5_FLASH_FP8_MODEL_REVISION GLMF_FP8_MODEL_REVISION)")" || exit 1
    family_args+=(--fp8-decode --fp8-snapshot "$fp8_snapshot")
  fi
  kda_fp8="$(key GLM5_FLASH_KDA_FP8 GLMF_KDA_FP8 auto)"
  case "$kda_fp8" in
    ""|auto) kda_fp8=row128 ;;
    off|row128|channel) ;;
    *) echo "GLM5_FLASH_KDA_FP8 must be auto, off, row128 or channel" >&2; exit 2 ;;
  esac
  # Default auto (GLM 5.3 Flash 1 RTX + 2: 65536 -> 2,097,152 tokens, 44 GiB still free, speed unchanged).
  glmf_pool="$(get POOL_TOKENS auto)"; [[ "$glmf_pool" != auto ]] || glmf_pool=0
  family_args+=(--kda-fp8 "$kda_fp8" --pool-tokens "$glmf_pool")
  kda_split="$(get GLM5_FLASH_KDA_SPLIT auto)"
  case "$kda_split" in
    ""|auto) [[ $head_split == 0 || $kda_fp8 == off ]] || family_args+=(--kda-output-shard --kda-prefill-expanded) ;;
    partials) ;;
    *) echo "GLM5_FLASH_KDA_SPLIT must be auto or partials" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_INDEX_CACHE: auto selects compact on one GPU, keys on a split.
  # keys: every token's BF16 key | gate
  # row beside its latent record, 11,804 B per token) or compact (the pooled keys plus each
  # sequence's open pool, 6,172 B per token, the same pooled keys bit for bit; one GPU only,
  # a head split keeps keys).
  index_eligible=0; index_reason="head split keeps token keys"
  [[ $head_split != 0 ]] || { index_eligible=1; index_reason="one GPU; bit-identical pooled keys"; }
  index_cache="$(release_glmf_auto GLM5_FLASH_INDEX_CACHE "$(get GLM5_FLASH_INDEX_CACHE auto)" \
    compact keys "$index_eligible" "$index_reason")"
  case "$index_cache" in
    ""|keys) ;;
    compact) family_args+=(--index-cache compact) ;;
    *) echo "GLM5_FLASH_INDEX_CACHE must be keys or compact" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_PREFILL_LANES (1-4) lanes of GLM5_FLASH_PREFILL_LANE_ROWS rows (whole 64-row
  # pages up to 4096) take each Spark prefill chunk, every lane with its own exchange in flight;
  # unset keeps the engine's two lanes of 4096.
  prefill_lanes="$(get GLM5_FLASH_PREFILL_LANES)"
  if [[ -n "$prefill_lanes" ]]; then
    [[ "$prefill_lanes" =~ ^[1-4]$ ]] || { echo "GLM5_FLASH_PREFILL_LANES must be 1 to 4" >&2; exit 2; }
    family_args+=(--prefill-lanes "$prefill_lanes")
  fi
  lane_rows="$(get GLM5_FLASH_PREFILL_LANE_ROWS)"
  if [[ -n "$lane_rows" ]]; then
    if ! [[ "$lane_rows" =~ ^[1-9][0-9]*$ ]] || ((lane_rows > 4096 || lane_rows % 64 != 0)); then
      echo "GLM5_FLASH_PREFILL_LANE_ROWS must be a multiple of 64 up to 4096" >&2; exit 2
    fi
    family_args+=(--prefill-lane-rows "$lane_rows")
  fi
  # GLM5_FLASH_HEADROOM_GIB: GPU memory an automatic pool leaves free for runtime growth when
  # every other allocation precedes it (unset: the engine's 2 GiB; a 32 GB card takes 1).
  headroom="$(get GLM5_FLASH_HEADROOM_GIB)"
  if [[ -n "$headroom" ]]; then
    [[ "$headroom" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "GLM5_FLASH_HEADROOM_GIB must be a non-negative size in GiB" >&2; exit 2; }
    family_args+=(--headroom-gib "$headroom")
  fi
  # GLM5_FLASH_GRAPH_BUDGET_MIB: device memory the captured decode graphs may hold (the least
  # recently launched leave past it); unset: unbounded, with the planner's allowance reserved.
  graph_budget="$(get GLM5_FLASH_GRAPH_BUDGET_MIB)"
  if [[ -n "$graph_budget" ]]; then
    [[ "$graph_budget" =~ ^[1-9][0-9]*$ ]] || { echo "GLM5_FLASH_GRAPH_BUDGET_MIB must be a positive whole number of MiB" >&2; exit 2; }
    family_args+=(--graph-budget-mib "$graph_budget")
  fi
  fp8_head="$(key GLM5_FLASH_FP8_HEAD GLMF_FP8_HEAD auto)"
  case "$fp8_head" in
    ""|auto) fp8_head=on ;;
    on|off) ;;
    *) echo "GLM5_FLASH_FP8_HEAD must be auto, on or off" >&2; exit 2 ;;
  esac
  case "$fp8_head" in
    on) family_args+=(--fp8-head true) ;;
    off) family_args+=(--fp8-head false) ;;
  esac
  # GLM5_FLASH_DRAFT_LINEAR: w8a8 with an FP8 drafter; w8a16 runs the W8A16 GEMV in passes of
  # 64 rows), wide (the same bits in passes of 128 rows) or w8a8 (past one draft block, E4M3
  # activations per row and 128-wide K block on FP8 tensor cores). Drafts only.
  draft_linear_default=w8a16
  if [[ ${#draft_args[@]} != 0 && "$(key SPECULATOR_FP8 DRAFT_FP8 auto)" != off ]]; then draft_linear_default=w8a8; fi
  draft_linear="$(get GLM5_FLASH_DRAFT_LINEAR "$draft_linear_default")"
  case "$draft_linear" in
    ""|w8a16) ;;
    wide|w8a8) family_args+=(--draft-linear "$draft_linear") ;;
    *) echo "GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_TARGET_HEAD: GLM 5.3 Flash's own spelling of TARGET_HEAD (--target-head). auto
  # (default) forwards nothing: the shared setting applies (tensor cores at every row count over the
  # BF16 head; the FP8 head runs its own program). exact (diagnostic) keeps the pedantic GEMM; an
  # explicit tensor with the FP8 head is refused.
  target_head="$(get GLM5_FLASH_TARGET_HEAD auto)"
  case "$target_head" in
    ""|auto) ;;
    exact) family_args+=(--target-head exact) ;;
    tensor)
      if [[ $fp8_head == on ]]; then
        echo "GLM5_FLASH_TARGET_HEAD=tensor runs the BF16 head; set GLM5_FLASH_FP8_HEAD=off" >&2
        exit 2
      fi
      family_args+=(--target-head tensor) ;;
    *) echo "GLM5_FLASH_TARGET_HEAD must be auto, exact or tensor" >&2; exit 2 ;;
  esac
  fp8_prefill="$(key GLM5_FLASH_FP8_PREFILL GLMF_FP8_PREFILL)"
  if [[ " ${family_args[*]} " == *" --kda-output-shard "* ]]; then
    case ",$fp8_prefill," in
      *,all,*|*,kda-o,*)
        echo "GLM5_FLASH_FP8_PREFILL=$fp8_prefill (KDA output W8A8) does not combine with the split's token-row KDA output; set GLM5_FLASH_KDA_SPLIT=partials or drop kda-o" >&2
        exit 2 ;;
    esac
  fi
  case ",$fp8_prefill," in
    *,all,*|*,kda-in,*|*,kda-o,*)
      if [[ $kda_fp8 == off ]]; then
        echo "GLM5_FLASH_FP8_PREFILL=$fp8_prefill runs KDA W8A8 over FP8 KDA weights; set GLM5_FLASH_KDA_FP8=row128 or channel" >&2
        exit 2
      fi ;;
  esac
  case "$fp8_prefill" in
    "") ;;
    off) family_args+=(--fp8-prefill none) ;;
    *) family_args+=(--fp8-prefill "$fp8_prefill") ;;
  esac
  # GLM5_FLASH_EXL3_WORKER_PATH: how a Spark EXL3 worker uploads a call's inputs and waits for its
  # GPU work. async (the default): pinned staging, one batched asynchronous copy, the hidden rows
  # decoded in place and a polled stream; blocking: the earlier copies and a blocking synchronize.
  # Same bits.
  exl3_worker_path="$(get GLM5_FLASH_EXL3_WORKER_PATH async)"
  case "$exl3_worker_path" in
    async) ;;
    blocking) spark_worker_env+=" -e CUTEAFD_EXL3_WORKER_PATH=blocking" ;;
    *) echo "GLM5_FLASH_EXL3_WORKER_PATH must be async or blocking" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_EXL3_ROUTE_DUMP: an absolute directory on every Spark. Each worker appends its
  # calls' routes to DIR/routes.<executor>.bin, which SparkInfer's GLM Flash decode benchmark
  # replays (--routes file:PATH); GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS caps the calls (default 200000).
  exl3_route_dump="$(get GLM5_FLASH_EXL3_ROUTE_DUMP "")"
  exl3_route_dump_calls="$(get GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS 200000)"
  if [[ -n "$exl3_route_dump" ]]; then
    [[ "$exl3_route_dump" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
      { echo "GLM5_FLASH_EXL3_ROUTE_DUMP must be an absolute directory" >&2; exit 2; }
    [[ "$exl3_route_dump_calls" =~ ^[1-9][0-9]{0,8}$ ]] ||
      { echo "GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS must be a positive call count" >&2; exit 2; }
    [[ "$ranks" != 0 ]] || { echo "GLM5_FLASH_EXL3_ROUTE_DUMP records Spark expert calls; SPARK_COUNT=0 runs none" >&2; exit 2; }
    spark_worker_env+=" -v $exl3_route_dump:$exl3_route_dump -e CUTEAFD_EXL3_ROUTE_DUMP=$exl3_route_dump/routes"
    spark_worker_env+=" -e CUTEAFD_EXL3_ROUTE_DUMP_CALLS=$exl3_route_dump_calls"
  fi
  # GLM5_FLASH_KDA_STATE: the KDA recurrent state, f32 (default) or bf16: half the state and
  # prefix-mark bytes, computed in FP32 and rounded after every decode/verify/commit row and at
  # each chunked-prefill window end (bf16-tile: after every 16-row prefill tile). It runs the
  # BF16-projection KDA programs on one GPU (GLM5_FLASH_KDA_FP8=off, no head split).
  kda_state="$(get GLM5_FLASH_KDA_STATE f32)"
  case "$kda_state" in
    ""|f32) ;;
    bf16|bf16-tile)
      if [[ $kda_fp8 != off || $head_split != 0 ]]; then
        echo "GLM5_FLASH_KDA_STATE=$kda_state runs the BF16-projection KDA programs on one GPU; set GLM5_FLASH_KDA_FP8=off without a head split" >&2
        exit 2
      fi
      family_args+=(--kda-state "$kda_state") ;;
    *) echo "GLM5_FLASH_KDA_STATE must be f32, bf16 or bf16-tile" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_EXL3_SCHEDULE: the Spark EXL3 decode schedule, default or gb10. gb10 runs the
  # m1-gb10/m80-gb10 TP4 exports: the same products and sums (the same bits), with the weight
  # words staged evict-first in L2, and at m80 64x128 tiles at two CTAs per SM.
  exl3_schedule="$(get GLM5_FLASH_EXL3_SCHEDULE default)"
  case "$exl3_schedule" in
    default) ;;
    gb10)
      [[ "$ranks" != 0 ]] || { echo "GLM5_FLASH_EXL3_SCHEDULE=gb10 is a Spark expert schedule; SPARK_COUNT=0 runs none" >&2; exit 2; }
      spark_worker_args+=" --exl3-schedule gb10" ;;
    *) echo "GLM5_FLASH_EXL3_SCHEDULE must be default or gb10" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_PREFIX_MARKS: where prefix-cache snapshots keep their KDA state marks: unset or
  # arena (the engine default: a 2C + 2 device arena beside the KV pool) or pool (units of the
  # KV pool itself, evicted like any snapshot's rows, unit 0 reserved; no arena to reserve). Pool
  # marks turn the pinned host tier on (HOST_CACHE_BYTES=auto) unless HOST_CACHE_BYTES is set (0
  # keeps it off): snapshots the pool evicts move to RAM instead of being lost.
  prefix_marks="$(get GLM5_FLASH_PREFIX_MARKS)"
  case "$prefix_marks" in
    "") ;;
    arena|pool) family_args+=(--prefix-marks "$prefix_marks") ;;
    *) echo "GLM5_FLASH_PREFIX_MARKS must be arena or pool" >&2; exit 2 ;;
  esac
  [[ "$prefix_marks" != pool || -n "$(get HOST_CACHE_BYTES)" ]] || family_args+=(--host-cache-bytes auto)
  # PREFIX_CACHE_MARK_MIB: the device budget of the prefix mark arena, MiB (unset: serve-glmf's
  # 2048), as for MiMo; the arena still holds at least two marks per sequence plus two.
  [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || family_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
  # GLM5_FLASH_REPLAY_RECORDS: auto shares when eligible; own keeps their
  # own 321 MB, 642 MB with GLM5_FLASH_DECODE_ROWS=128) or shared (the prefill lanes' scratch, which
  # no decode step reads). One GPU whose pool is sized from measured memory (an automatic pool with
  # Spark experts).
  replay_eligible=0
  if [[ $head_split != 0 ]]; then replay_reason="head split keeps its own records"
  elif [[ $glmf_pool != 0 ]]; then replay_reason="fixed pool keeps its own records"
  elif [[ $ranks == 0 ]]; then replay_reason="local experts keep their own records"
  else replay_eligible=1; replay_reason="one GPU; automatic measured pool with Spark experts"; fi
  replay_records="$(release_glmf_auto GLM5_FLASH_REPLAY_RECORDS "$(get GLM5_FLASH_REPLAY_RECORDS auto)" \
    shared own "$replay_eligible" "$replay_reason")"
  case "$replay_records" in
    ""|own) ;;
    shared)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_REPLAY_RECORDS=shared keeps the records in one GPU's prefill scratch; serve it without a head split" >&2
        exit 2
      fi
      family_args+=(--replay-records shared) ;;
    *) echo "GLM5_FLASH_REPLAY_RECORDS must be own or shared" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_DECODE_ROWS: auto keeps 64; 128 remains opt-in after the C16 cost-model gate. With
  # 128 a step of more than 64 rows runs the wide _m128 programs (a build with
  # CUTEAFD_GLMF_WIDE_DECODE_ROWS=128) and a verify step schedules up to the GPU's whole sparse MLA
  # waves (127 rows on an RTX 5090); fewer rows keep the _m64 programs. One GPU only.
  decode_rows="$(get GLM5_FLASH_DECODE_ROWS auto)"
  if [[ "$decode_rows" == auto ]]; then
    decode_rows="$(release_glmf_auto GLM5_FLASH_DECODE_ROWS auto 128 64 0 \
      "128 rows remain opt-in; C16 and post-C16 C1 draft-cost gate")"
  fi
  case "$decode_rows" in
    ""|64) ;;
    128)
      [[ $head_split == 0 ]] ||
        { echo "GLM5_FLASH_DECODE_ROWS=128 runs the wide decode programs on one GPU; a head split takes 64" >&2; exit 2; }
      family_args+=(--decode-rows 128) ;;
    *) echo "GLM5_FLASH_DECODE_ROWS must be 64 or 128" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_PREFILL_BATCH: off (default: one prefill pass per prompt) or on (the prompts that
  # wait together prefill in one pass, each sequence's own programs over its rows, one Spark wave
  # per MoE layer for all of them).
  prefill_batch="$(get GLM5_FLASH_PREFILL_BATCH off)"
  case "$prefill_batch" in
    ""|off) ;;
    on) family_args+=(--prefill-batch) ;;
    *) echo "GLM5_FLASH_PREFILL_BATCH must be on or off" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_VERIFY_POLICY: which drafts a speculative step verifies under the verify budget (64
  # rows, or the GPU's whole sparse MLA waves with GLM5_FLASH_DECODE_ROWS=128), cost (default: the
  # same room for every sequence, the cost model's depth within it) or chain (each sequence's
  # drafts cut at GLM5_FLASH_SPEC_TAU, default 0.7, of cumulative draft probability, then the least
  # likely drafts across sequences dropped first).
  verify_policy="$(get GLM5_FLASH_VERIFY_POLICY cost)"
  case "$verify_policy" in
    ""|cost) ;;
    chain) family_args+=(--verify-policy chain) ;;
    *) echo "GLM5_FLASH_VERIFY_POLICY must be cost or chain" >&2; exit 2 ;;
  esac
  # GLM Flash always runs the resource-priced shared draft policy (v3 D3); the D2 switch is gone.
  if [[ -n "$(get GLM5_FLASH_DRAFT_POLICY)" ]]; then
    echo "GLM5_FLASH_DRAFT_POLICY was removed in v3 D3: GLM Flash always runs the shared draft policy; delete the key" >&2
    exit 2
  fi
  # GLM5_FLASH_ROUTE_RING_CHECK=on (diagnostic) also copies Spark layers' router ids through the local
  # route ring and counts disagreements with the staged ids.
  case "$(get GLM5_FLASH_ROUTE_RING_CHECK off)" in
    off) ;;
    on) trace_args+=(-e "CUTEAFD_GLMF_ROUTE_RING_CHECK=1") ;;
    *) echo "GLM5_FLASH_ROUTE_RING_CHECK must be on or off" >&2; exit 2 ;;
  esac
  case "$(get GLM5_FLASH_DRAFT_TRACE off)" in
    off) ;;
    on) trace_args+=(-e "CUTEAFD_GLMF_DRAFT_TRACE=1") ;;
    *) echo "GLM5_FLASH_DRAFT_TRACE must be on or off" >&2; exit 2 ;;
  esac
  remote_drafter="$(get GLM5_FLASH_REMOTE_DRAFTER)"
  case "$remote_drafter" in
    ""|on|off) ;;
    *) echo "GLM5_FLASH_REMOTE_DRAFTER must be on or off" >&2; exit 2 ;;
  esac
  spec_tau="$(get GLM5_FLASH_SPEC_TAU)"
  if [[ -n "$spec_tau" ]]; then
    [[ "$spec_tau" =~ ^(0?[.][0-9]*[1-9][0-9]*|1([.]0*)?)$ ]] || { echo "GLM5_FLASH_SPEC_TAU must be in (0, 1]" >&2; exit 2; }
    family_args+=(--spec-tau "$spec_tau")
  fi
fi
# INSTANCE names a launch that runs beside others on disjoint hardware
# (`cuteafd bench smoke` sets it): its coordinator container is
# cuteafd-coordinator-INSTANCE; empty keeps the one cuteafd-coordinator.
# SPARK_COUNT=0: routed experts fit the selected coordinator GPUs. GLM 5.3
# uses solver-selected TP2 (--rtx-expert-layers); the other families use
# --local-experts.
if [[ "$ranks" == 0 ]]; then
  case "$family" in
    glm5) ;;
    glm5_flash|mimo_v2|qwen4) family_args+=(--local-experts) ;;
    *) echo "SPARK_COUNT=0 serves GLM 5.3, GLM 5.3 Flash, MiMo V2 and Qwen 3.8, not $family" >&2; exit 2 ;;
  esac
fi
# Match persistent connections owned by the selected family's transport, not
# the Spark world size. Forward the resolved lane policy to the coordinator.
lane_env_args=()
case "$family" in
  deepseek_v4) rdma_endpoints=2 ;;
  glm5)
    selected_lanes="${CUTEAFD_GLM_PREFILL_LANES:-3}"
    [[ "$selected_lanes" =~ ^[1-4]$ ]] || { echo "CUTEAFD_GLM_PREFILL_LANES must be 1 to 4" >&2; exit 2; }
    if [[ "$ranks" == 0 ]]; then
      selected_lanes=1
      rdma_endpoints=0
    else
      rdma_endpoints=$((1 + (selected_lanes > 1 ? selected_lanes : 0)))
    fi
    lane_env_args+=(-e "CUTEAFD_GLM_PREFILL_LANES=$selected_lanes") ;;
  glm5_flash)
    rdma_endpoints="${prefill_lanes:-2}"
    if [[ "${CUTEAFD_GLMF_PREFILL_LANES:-}" == 1 ]]; then rdma_endpoints=1; fi
    [[ -z "${CUTEAFD_GLMF_PREFILL_LANES:-}" ]] || lane_env_args+=(-e "CUTEAFD_GLMF_PREFILL_LANES=$CUTEAFD_GLMF_PREFILL_LANES")
    # Planner and server must offer the same conditional drafter homes.
    # Explicit config overrides the legacy process-wide diagnostic switch.
    case "$remote_drafter" in
      on) lane_env_args+=(-e CUTEAFD_GLMF_REMOTE_DFLASH=1) ;;
      off) lane_env_args+=(-e CUTEAFD_GLMF_REMOTE_DFLASH=0) ;;
      "") [[ -z "${CUTEAFD_GLMF_REMOTE_DFLASH:-}" ]] || lane_env_args+=(-e "CUTEAFD_GLMF_REMOTE_DFLASH=$CUTEAFD_GLMF_REMOTE_DFLASH") ;;
    esac ;;
  qwen4) rdma_endpoints=1 ;;
  mimo_v2)
    hidden="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); t=c.get("text_config", c); print(t.get("hidden_size", 4096 if t.get("model_type") == "mimo_v2_flash" else 6144))' "$root/snapshots/$revision/config.json")"
    selected_lanes="${CUTEAFD_MIMO_PREFILL_LANES:-$([[ "$hidden" == 4096 ]] && printf 2 || printf 3)}"
    [[ "$selected_lanes" =~ ^[1-4]$ ]] || { echo "CUTEAFD_MIMO_PREFILL_LANES must be 1 to 4" >&2; exit 2; }
    lane_env_args+=(-e "CUTEAFD_MIMO_PREFILL_LANES=$selected_lanes")
    rdma_endpoints="$selected_lanes"
    # BF16-decode fits the wider FP8 prefill ring; only full BF16 changes
    # the maximum ingress frame. NVFP4 is detected by the worker itself.
    [[ "$expert_input" != bf16 ]] || spark_worker_args+=" --bf16-ingress" ;;
  *) echo "missing worker endpoint geometry for $family" >&2; exit 2 ;;
esac
if [[ -n "$expert_input" && "$expert_input" != fp8 && "$ranks" == 0 ]]; then
  echo "EXPERT_INPUT=$expert_input requires Spark experts" >&2
  exit 2
fi
# Resolve the count and cold placement before starting or removing containers.
# Media off still requires the shared expert and memory admission.
vision_peers=()
encoder_ranks=()
encoder_hash=""
encoder_max_tokens=4096
encoder_port=$((port + 1))
audio_encoder_port=$((port + 2))
audio_encoder_ranks=()
audio_peers=()
audio_encoder_hash=""
  plan_rtx=1; ((head_split == 0)) || plan_rtx=2
  plan_pool="$(get POOL_TOKENS auto)"; [[ "$plan_pool" != auto ]] || plan_pool=0
  total_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  plan_gib="${coordinator_budget:-$(python3 -c 'import sys; print(float(sys.argv[1])/1024)' "$total_mib")}"
  if [[ "$family" == qwen4 ]]; then
    free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if ((restart)); then
      own_pids="$(docker top "$coordinator_name" -eo pid 2>/dev/null | python3 -c 'import sys; print(" ".join(x.strip() for x in sys.stdin if x.strip().isdigit()))' || true)"
      if [[ -n "$own_pids" ]]; then
        own_mib="$(nvidia-smi --id="$gpu" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits \
          | python3 -c 'import csv,sys;p=set(sys.argv[1].split());print(sum(int(r[1].strip()) for r in csv.reader(sys.stdin) if len(r)==2 and r[0].strip() in p and r[1].strip().isdigit()))' "$own_pids")"
        free_mib=$((free_mib + own_mib))
      fi
    fi
    plan_gib="$(python3 -c 'import sys;f,t=map(float,sys.argv[1:3]);b=float(sys.argv[3]) if sys.argv[3] else t/1024;print(max(0,min(f/1024,b-(t-f)/1024)))' "$free_mib" "$total_mib" "$coordinator_budget")"
  fi
  plan_draft_args=()
  if [[ "$family" == mimo_v2 ]]; then
    plan_draft_args+=(--concurrency "$(get CONCURRENCY "$default_concurrency")"
      --prefix-cache-entries "$(get PREFIX_CACHE_ENTRIES 20)"
      --context-tokens "$(get MAX_CONTEXT_TOKENS "$default_context")"
      --prefill-rows 4096 --expert-input "${expert_input:-fp8}")
    [[ -z "${local_layers:-}" ]] || plan_draft_args+=(--rtx-expert-layers "$local_layers")
    [[ "$(get FULL_PREFILL_LOGITS off)" != on ]] || plan_draft_args+=(--full-prefill-logits)
    [[ "$(get DECODE_GRAPHS off)" != on ]] || plan_draft_args+=(--mimo-decode-graphs)
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
    [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || plan_draft_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
    [[ -z "$(get DRAFT_SEQUENCES)" ]] || plan_draft_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    if [[ "$(get MIMO_PREFIX_DRAFT off)" == on ]]; then
      plan_draft_args+=(--mimo-prefix-draft)
    fi
  fi
  # GLM 5.3 Flash: the counts serve-glmf allocates for the keys a launch sets (unset keys keep
  # the planner's defaults, which are serving's). Its recurrent state holds max(--slots, which
  # is 8, --max-sequences) slots, and its prefix mark arena counts min(--max-sequences,
  # DECODE_ROWS = 64) lanes, so both go to the plan as such beside the sequences.
  if [[ "$family" == glm5_flash ]]; then
    glmf_sequences="$(get CONCURRENCY)"
    if [[ -n "$glmf_sequences" ]]; then
      [[ "$glmf_sequences" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENCY must be a positive sequence count" >&2; exit 2; }
      plan_draft_args+=(--concurrency "$glmf_sequences" --state-slots "$((glmf_sequences > 8 ? glmf_sequences : 8))"
        --mark-lanes "$((glmf_sequences < 64 ? glmf_sequences : 64))")
    fi
    [[ -z "$(get PREFIX_CACHE_ENTRIES)" ]] || plan_draft_args+=(--prefix-cache-entries "$(get PREFIX_CACHE_ENTRIES)")
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
    # The plan resolves VISION=auto as serve-glmf would serve it: with the template source serving
    # gets, vision stays on; without one, a checkpoint template that cannot mark images plans text.
    [[ -z "$chat_template_from" || "$vision" == off ]] || plan_draft_args+=(--chat-template-from "$chat_template_from")
  fi
  if [[ "$family" == glm5_flash ]]; then
    plan_draft_args+=(--kda-fp8 "$kda_fp8" --kda-state "${kda_state:-f32}" --fp8-head "$([[ $fp8_head == on ]] && echo true || echo false)")
    [[ "${target_head:-auto}" == auto || -z "${target_head:-}" ]] || plan_draft_args+=(--target-head "$target_head")
    # Draft-head arithmetic borrows the same target weight, not a second resident head.
    plan_draft_args+=(--draft-head exact)
  fi
  [[ "$family" != glm5_flash || "${index_cache:-keys}" != compact ]] || plan_draft_args+=(--index-cache compact)
  # GLM 5.3 Flash: the server's prefix-mark store, so the plan reserves an arena only when serve-glmf allocates one.
  [[ "$family" != glm5_flash || -z "${prefix_marks:-}" ]] || plan_draft_args+=(--prefix-marks "$prefix_marks")
  # GLM 5.3 Flash: shared replay records (GLM5_FLASH_REPLAY_RECORDS above) live in the prefill
  # scratch, so the plan does not charge their own copy either.
  [[ "$family" != glm5_flash || "${replay_records:-own}" != shared ]] || plan_draft_args+=(--replay-records shared)
  # GLM 5.3 Flash plans the decode rows serving takes (GLM5_FLASH_DECODE_ROWS above): 128 rows
  # charge their wider decode workspace, selector, replay records and expert intake, as serving
  # admits them, before an encoder placement is chosen.
  if [[ "$family" == glm5_flash && "${decode_rows:-64}" == 128 ]]; then
    plan_draft_args+=(--decode-rows 128)
  fi
  if [[ "$family" == qwen4 ]]; then
    plan_draft_args+=(--kv-cache "$kv_format")
    for key in FP8_DECODE:--fp8-decode FP8_HEAD:--mtp-fp8-head; do
      case "$(get "QWEN_${key%%:*}")" in
        on) plan_draft_args+=("${key#*:}" true) ;;
        off) plan_draft_args+=("${key#*:}" false) ;;
      esac
    done
    [[ "$head_split" != 1 ]] || plan_draft_args+=(--rtx-expert-layers "$(get RTX_EXPERT_LAYERS all)")
  fi
  count_args=(--available-sparks "$available_sparks")
  if ((count_explicit)) || [[ "$backend" == local ]]; then count_args+=(--spark-ranks "$ranks"); fi
  # The plan charges the drafter this launch actually serves (SPECULATOR=off charges none).
  count_args+=(--speculator "$speculator")
  for pair in 'CONCURRENCY:--concurrency' 'PREFIX_CACHE_ENTRIES:--prefix-cache-entries' \
    'MAX_CONTEXT_TOKENS:--context-tokens' 'PREFILL_BATCH_TOKENS:--prefill-rows' \
    'RTX_EXPERT_LAYERS:--rtx-expert-layers'; do
    key_name="${pair%%:*}"; flag="${pair#*:}"; value="$(get "$key_name")"
    if [[ -n "$value" && "$value" != auto && " ${plan_draft_args[*]} " != *" $flag "* ]]; then
      plan_draft_args+=("$flag" "$value")
    fi
  done
  # HF_HOME=/hf: the planner resolves draft snapshots under it, like serving.
  plan_json="$(docker run --rm --network none "${lane_env_args[@]}" -e HF_HOME=/hf -v "$hub:/hf/hub:ro" "${wip_mount_args[@]}" \
    "$coordinator_image" cuteafd plan "$snapshot" --attention-placement "$attention_placement" --vision "$vision" --audio "$audio" --json --layout --require-ready \
    "${count_args[@]}" --spark-budget-gib "$(python3 -c 'import sys;print(int(sys.argv[1])/2**30)' "$budget")" \
    --rtx "$plan_rtx" --coordinator-gpu-budget-gib "$plan_gib" --coordinator-weight-budget-gib "$plan_gib" --pool-tokens "$plan_pool" --vision-replicas "$vision_replicas" "${plan_draft_args[@]}")"
  selected_ranks="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spark_ranks"])' <<<"$plan_json")"
  if ((count_explicit)) || [[ "$backend" == local ]]; then
    [[ "$selected_ranks" == "$ranks" ]] || release_die "planner changed exact Spark count $ranks to $selected_ranks"
  elif [[ "$selected_ranks" != "$ranks" ]]; then
    # Re-resolve count-dependent MTP, ingress and local flags before launching.
    export SPARK_COUNT="$selected_ranks"
    exec bash "$repo_root/scripts/launch/run-family.sh" "${launch_argv[@]}"
  fi
# Check the admitted count's images before --restart or weight loading. The worker's
# resident admission validates the sibling for its full 4096-row workspace even
# when only decode requests use BF16, so this preflight requires the same coverage.
if [[ -n "$expert_input" && "$expert_input" != fp8 ]]; then
  [[ "$ranks" != 0 ]] || { echo "EXPERT_INPUT=$expert_input requires Spark experts" >&2; exit 2; }
  store_dtype="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("quantization_config", {}).get("store_dtype", "fp8"))' "$root/snapshots/$revision/config.json")"
  case "$store_dtype" in
    fp8) expert_geometry=mimo ;;
    mxfp4)
      hidden="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("text_config", c).get("hidden_size", 0))' "$root/snapshots/$revision/config.json")"
      case "$hidden" in
        4096) expert_geometry=mimof ;;
        6144) expert_geometry=mimop ;;
        *) echo "EXPERT_INPUT=$expert_input has no MXFP4 MiMo package for hidden_size=$hidden" >&2; exit 2 ;;
      esac ;;
    *) echo "EXPERT_INPUT=$expert_input has no package for store_dtype=$store_dtype" >&2; exit 2 ;;
  esac
  printf -v preflight_command '%q ' docker run --rm -i --entrypoint python3 "$spark_image" - "$expert_geometry" "tp$ranks" 4096
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    if ! ssh "$host" "$preflight_command" < "$repo_root/scripts/launch/preflight-fp8-bf16.py"; then
      echo "$host cannot serve EXPERT_INPUT=$expert_input; build $expert_geometry:fp8 with CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$expert_geometry (or its WIP equivalent)" >&2
      exit 2
    fi
  done
fi
if { [[ ( "$family" == mimo_v2 || "$family" == qwen4 || "$family" == glm5_flash ) && "$vision" != off ]] &&
   python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("vision_config") else 1)' "$root/snapshots/$revision/config.json"; } ||
   [[ "$family" == mimo_v2 && "$audio" != off ]]; then
  selected="$(python3 -c '
import json,sys
p=json.load(sys.stdin); e=p.get("encoder")
assert p["placement_supported"] and p["fits"], "encoder deployment cannot fit: "+str(p.get("hints"))
k=e["kind"] if e else {"kind":"off"}; kind=k["kind"]; h=p["encoder_plan_hash"]
cap=p.get("max_image_tokens") or (1024 if sys.argv[1]=="qwen4" else 4096)
assert type(cap) is int and 1<=cap<=4096, "invalid encoder image cap"
assert len(h)==64 and all(c in "0123456789abcdef" for c in h), "invalid encoder plan hash"
if kind=="spark":
    ranks=[k["rank"]]+e["replicas"]
    assert len(ranks)==len(set(ranks)) and all(0<=r<p["spark_ranks"] for r in ranks)
    print("spark:"+str(k["rank"]),h,",".join(map(str,ranks)),cap)
elif kind=="rtx": print("rtx:"+str(k["gpu"]),h,"-",cap)
elif kind=="off": print("off",h,"-",cap)
else: raise ValueError("idle-host launch needs an explicit inventory")
' "$family" <<<"$plan_json")"
  read -r vision encoder_hash rank_csv encoder_max_tokens <<<"$selected"
  if [[ "$head_split" == 1 ]]; then
    vision="$(dual_owner_media_device "$vision" "$((gpu == lower ? 0 : 1))" "$((gpu == lower ? 1 : 0))")"
  fi
  if [[ "$vision" == spark:* ]]; then
    IFS=, read -r -a encoder_ranks <<<"$rank_csv"
    for encoder_rank in "${encoder_ranks[@]}"; do
      vision_peers+=("$(get "SPARK_${encoder_rank}_LANE_A"):$encoder_port")
    done
    family_args+=(--vision-peers "$(IFS=,; printf '%s' "${vision_peers[*]}")" --encoder-plan-hash "$encoder_hash" --encoder-revision "$revision")
  fi
  if [[ "$audio" != off ]]; then
    selected_audio="$(python3 -c '
import json,sys
p=json.load(sys.stdin); e=p.get("audio_encoder")
k=e["kind"] if e else {"kind":"off"}; kind=k["kind"]; h=p["encoder_plan_hash"]
assert len(h)==64 and all(c in "0123456789abcdef" for c in h), "invalid audio plan hash"
if kind=="spark":
    ranks=[k["rank"]]+e["replicas"]
    assert len(ranks)==len(set(ranks)) and all(0<=r<p["spark_ranks"] for r in ranks)
    print("spark:"+str(k["rank"]),h,",".join(map(str,ranks)))
elif kind=="rtx": print("rtx:"+str(k["gpu"]),h,"-")
elif kind=="off" and sys.argv[1]=="auto":
    if e and e.get("shortfall",0): print("audio auto disabled: "+e["reason"],file=sys.stderr)
    print("off",h,"-")
else: raise ValueError("enabled audio has no launchable admitted owner")
' "$audio" <<<"$plan_json")"
    read -r audio audio_encoder_hash audio_rank_csv <<<"$selected_audio"
    if [[ "$head_split" == 1 ]]; then
      audio="$(dual_owner_media_device "$audio" "$((gpu == lower ? 0 : 1))" "$((gpu == lower ? 1 : 0))")"
    fi
    if [[ "$audio" == spark:* ]]; then
      IFS=, read -r -a audio_encoder_ranks <<<"$audio_rank_csv"
      for audio_rank in "${audio_encoder_ranks[@]}"; do
        audio_peers+=("$(get "SPARK_${audio_rank}_LANE_A"):$audio_encoder_port")
      done
      family_args+=(--audio-peers "$(IFS=,; printf '%s' "${audio_peers[*]}")" --audio-encoder-plan-hash "$audio_encoder_hash" --audio-encoder-revision "$revision")
    fi
  fi
elif [[ "$vision" == spark* || "$vision" == rtx* || "$audio" != off ]]; then
  release_die "explicit media placement requires a supported bundled vision/audio checkpoint"
fi
# Replace the original policy with the selected placement, without duplicated flags.
for ((arg = 0; arg < ${#family_args[@]}; arg++)); do
  [[ "${family_args[arg]}" != --vision ]] || family_args[arg+1]="$vision"
  [[ "${family_args[arg]}" != --audio ]] || family_args[arg+1]="$audio"
done
peers=()
# --restart removes only this coordinator and its host-port workers. Broad
# cleanup is opt-in (--all); shared hosts may hold another instance's workers.
if [[ "$restart" == 1 ]]; then
  previous_csv=""
  if [[ "$ranks" == 0 && "$configured_ranks" != 0 ]]; then
    # Switching to local experts must release this coordinator's old workers.
    # A new local launch must not stop another instance's Spark workers.
    previous_csv="$(docker inspect --format '{{json .Config.Cmd}}' "$coordinator_name" 2>/dev/null \
      | python3 -c 'import json,sys; c=json.load(sys.stdin); print(c[c.index("--peers")+1])' 2>/dev/null || true)"
  fi
  docker rm -f "$coordinator_name" >/dev/null 2>&1 || true
  if [[ -n "$previous_csv" ]]; then
    IFS=, read -r -a previous_peers <<<"$previous_csv"
    for ((rank = 0; rank < configured_ranks; rank++)); do
      host="$(get "SPARK_${rank}_HOST")"; lane="$(get "SPARK_${rank}_LANE_A")"
      for peer in "${previous_peers[@]}"; do
        old_port="${peer##*:}"
        if [[ "${peer%:*}" == "$lane" && "$old_port" =~ ^[0-9]+$ ]]; then
          ssh "$host" "docker rm -f cuteafd-spark-expert-$host-$old_port >/dev/null 2>&1 || true"
        fi
      done
    done
  fi
  cleanup_ranks="$ranks"
  ((restart_all == 0)) || cleanup_ranks="$configured_ranks"
  for ((rank = 0; rank < cleanup_ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    if ((restart_all)); then
      ssh "$host" 'ids=$(docker ps -a --format "{{.Names}}" --filter "name=^cuteafd-spark-expert-.+-[0-9]+$" | grep -vE "^cuteafd-spark-expert-wip($|-)"); [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true'
    else
      ssh "$host" "docker rm -f cuteafd-spark-expert-$host-$port >/dev/null 2>&1 || true"
    fi
  done
fi
shared_coordinator_env_args
shared_thermal_args
# CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP (local-ip=device,...; each process picks the entry
# for its own fabric address) reaches the coordinator and every worker when it is set. Without
# it each opens its first RDMA device, which need not carry the fabric address (GB10 exposes
# several RDMA functions per port).
device_map="${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}"
device_map_env="" device_map_args=()
if [[ -n "$device_map" ]]; then
  [[ "$device_map" =~ ^[A-Za-z0-9.:=,_-]+$ ]] ||
    { echo "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP must be local-ip=device[,local-ip=device...]" >&2; exit 2; }
  device_map_env="-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$device_map"
  device_map_args=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$device_map")
fi
# SparkNest cache management stays available when installed. Without it the
# workers advise their checkpoint pages after loading; global sudo cache drops
# are an explicit operator opt-in, never required infrastructure.
spark_hosts=()
spark_host_names=()
for ((rank = 0; rank < ranks; rank++)); do
  spark_hosts+=(--host "$(get "SPARK_${rank}_HOST")")
  spark_host_names+=("$(get "SPARK_${rank}_HOST")")
done
drop_spark_caches() {
  if command -v nest >/dev/null; then nest drop-caches "${spark_hosts[@]}" >/dev/null; return; fi
  if [[ "${CUTEAFD_GLOBAL_PAGE_CACHE_DROP:-0}" != 1 ]]; then
    echo "SparkNest absent: workers use checkpoint fadvise(DONTNEED); no global cache drop."
    return 0
  fi
  local host failed=0
  for host in "${spark_host_names[@]}"; do
    ssh -n "$host" 'sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null' || failed=1
  done
  return "$failed"
}
if [[ -n "$wip_slot" ]]; then
  # The slot must exist everywhere before anything starts, and the development images must
  # carry the SparkInfer revision this checkout pins.
  pinned_sparkinfer="$(python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" --print-revision)"
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    ssh "$host" bash -s -- "$wip_slot" "$spark_image" "$pinned_sparkinfer" "$wip_spark_container" "$WIP_LAYOUT_SLOT" "${WIP_ROOT:-__default__}" "${WIP_INSTANCE:-__none__}" <<'STAGE' ||
set -euo pipefail
slot="$1" image="$2" pinned="$3"
container="$4" layout_slot="$5"
wip_root_arg="${6:-__default__}"; wip_instance_arg="${7:-__none__}"
[[ "$wip_instance_arg" != __none__ ]] || wip_instance_arg=
[[ "$wip_root_arg" != __default__ ]] || wip_root_arg="$HOME/.cache/cuteafd/builds/wip${wip_instance_arg:+-$wip_instance_arg}"
wip_cp() { # slot files come from the container when it exists, else from the WIP root
  if docker container inspect "$container" >/dev/null 2>&1; then docker cp "$container:/wip/$1" "$2"
  else [[ -e "$wip_root_arg/${1%/.}" ]] && cp -a "$wip_root_arg/$1" "$2"; fi
}
if ! docker image inspect "$image" >/dev/null 2>&1; then
  registry="${image%%/*}"
  [[ "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" == localhost ) ]] || { echo "missing local dev image $image" >&2; exit 1; }
  docker pull "$image"
fi
layout="$HOME/.cache/cuteafd/wip-run/$layout_slot" raw="$HOME/.cache/cuteafd/wip-run/$layout_slot.tmp/raw"
rm -rf "$layout.tmp" && mkdir -p "$raw" "$layout.tmp/bin" "$layout.tmp/lib" "$layout.tmp/share"
wip_cp "slots/$slot/spark-expert/workspace/.cuteafd-wip/." "$raw/"
wip_cp "slots/$slot/spark-expert/workspace/docker/release-entrypoint.sh" "$raw/"
mv "$raw/cuteafd" "$layout.tmp/bin/cuteafd"
mv "$raw/libcuteafd_native.so" "$layout.tmp/lib/"
[[ ! -d "$raw/exl3" ]] || mv "$raw/exl3" "$layout.tmp/lib/exl3"
[[ ! -d "$raw/fp8" ]] || mv "$raw/fp8" "$layout.tmp/lib/fp8"
mv "$raw/"* "$layout.tmp/share/"
mkdir -p "$layout.tmp/source/third_party" "$layout.tmp/source/scripts/build"
slot_source="slots/$slot/spark-expert/workspace"
wip_cp "$slot_source/third_party/sparkinfer" "$layout.tmp/source/third_party/"
wip_cp "$slot_source/third_party/sparkinfer.lock.json" "$layout.tmp/source/third_party/"
wip_cp "$slot_source/scripts/build/verify-sparkinfer-source.py" "$layout.tmp/source/scripts/build/"
python3 "$layout.tmp/source/scripts/build/verify-sparkinfer-source.py" \
  --source "$layout.tmp/source/third_party/sparkinfer" --lock "$layout.tmp/source/third_party/sparkinfer.lock.json"
rm -rf "$raw" "$layout" && mv "$layout.tmp" "$layout"
STAGE
      { echo "$host: WIP slot $wip_slot is not staged (build it with ./wip.sh --slot $wip_slot)" >&2; exit 1; }
  done
fi
((ranks == 0)) || drop_spark_caches || echo "warning: could not drop Spark page caches" >&2
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  lane="$(get "SPARK_${rank}_LANE_A")"
  peers+=("$lane:$port")
  encoder_args=""
  for encoder_rank in "${encoder_ranks[@]}"; do
    if [[ "$rank" == "$encoder_rank" ]]; then
      encoder_args="--encoder --encoder-listen 0.0.0.0:$encoder_port --encoder-plan-hash $encoder_hash --encoder-revision $revision --encoder-max-tokens $encoder_max_tokens"
    fi
  done
  for audio_rank in "${audio_encoder_ranks[@]}"; do
    if [[ "$rank" == "$audio_rank" ]]; then
      encoder_args+=" --audio-encoder --audio-encoder-listen 0.0.0.0:$audio_encoder_port --audio-encoder-plan-hash $audio_encoder_hash --audio-encoder-revision $revision"
    fi
  done
  release_profile_docker_args "$(get PROFILE)" "$(get PROFILE_DIR)" "$host"
  worker_profile="" worker_prep=""
  ((${#RELEASE_PROFILE_ARGS[@]} == 0)) || worker_profile=" $(printf '%q ' "${RELEASE_PROFILE_ARGS[@]}")"
  [[ -z "$(get PROFILE_DIR)" ]] || worker_prep="mkdir -p $(printf '%q' "$(get PROFILE_DIR)") && release_require_writable_state $(printf '%q' "$(get PROFILE_DIR)") && "
  ssh "$host" "$(declare -f release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args); release_serving_docker_args; cuteafd_cdi_gpu_args all || exit 2; ${worker_prep}docker run -d --name cuteafd-spark-expert-$host-$port --restart no \"\${CUTEAFD_CDI_ARGS[@]}\" --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \"\${RELEASE_SERVING_ARGS[@]}\" -e RUST_LOG=info ${shared_worker_env_args[*]}$spark_worker_env $wip_worker_args $device_map_env$worker_profile \
    -v \$(readlink -f \$HOME/.cache/huggingface/hub):/hf/hub:ro '$spark_image' \
    cuteafd expertd-native --snapshot '$snapshot' --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    --rank $rank --world $ranks --rdma-endpoints $rdma_endpoints --capacity 4096 --device-budget-bytes $budget $layer_args$spark_worker_args \
    --listen 0.0.0.0:$port $encoder_args >/dev/null" &
done
wait
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  # A placement worker bootstraps after media admission and waits for the
  # coordinator solve before expert weights load.
  ready_marker='worker ready'
  ((placement_handshake == 0)) || ready_marker='worker bootstrap ready'
  ready_deadline=$((SECONDS + 900))
  until ssh "$host" "docker logs cuteafd-spark-expert-$host-$port 2>&1 | grep -q '$ready_marker'"; do
    ((SECONDS < ready_deadline)) || { echo "$host readiness timed out" >&2; exit 1; }
    ssh "$host" "docker ps -q -f name=cuteafd-spark-expert-$host-$port | grep -q ." ||
      { echo "$host expert worker exited:" >&2; ssh "$host" "docker logs --tail 20 cuteafd-spark-expert-$host-$port" >&2; exit 1; }
    sleep 2
  done
done
# Bind cache identity to the actual SM121 export, never the coordinator's SM120
# backend. The worker publishes this only after native admission and owner readiness.
if ((${#audio_encoder_ranks[@]})); then
  audio_backend=""
  for audio_rank in "${audio_encoder_ranks[@]}"; do
    host="$(get "SPARK_${audio_rank}_HOST")"
    backend="$(ssh "$host" "docker logs cuteafd-spark-expert-$host-$port 2>&1" | python3 -c '
import re,sys
matches=re.findall(r"mimo_audio_fp32_v1/cuda[0-9]+/cufft[0-9]+/cublas[0-9.]+/cute_aot_sm121/export[0-9a-f]{64}",sys.stdin.read())
assert matches and len(set(matches))==1, "missing or conflicting ready audio backend identity"
print(matches[0])')"
    [[ -z "$audio_backend" || "$audio_backend" == "$backend" ]] || release_die "audio replicas have different backend exports"
    audio_backend="$backend"
  done
  family_args+=(--audio-encoder-backend "$audio_backend")
fi
# Loading leaves ~10 GiB of checkpoint pages cached per Spark (sparknest passthrough: the
# workers' own fadvise cannot reach them), and GB10 CUDA allocations do not reclaim page
# cache: drop it once every rank is resident. CUTEAFD_SPARK_DROP_PAGE_CACHE=0 keeps it.
if ((ranks > 0 && ${#spark_hosts[@]} > 0)) && [[ "${CUTEAFD_SPARK_DROP_PAGE_CACHE:-1}" != 0 ]]; then
  drop_spark_caches || echo "warning: could not drop Spark page caches after loading" >&2
fi
peer_csv="$(IFS=,; echo "${peers[*]}")"
peer_args=()
[[ -z "$peer_csv" ]] || peer_args=(--peers "$peer_csv")
((placement_handshake == 0 || ranks == 0)) || peer_args+=(--placement-handshake)
shared_api_state "$coordinator_image" "${instance:-default}"
family_args+=("${api_args[@]}")
shared_profile_args
shared_graph_census_args
shared_bench_nonce_args
tp2_env_args=()
for key in CUTEAFD_TP2_EXCHANGE CUTEAFD_ROUTE_CHECK; do
  [[ -z "${!key:-}" ]] || tp2_env_args+=(-e "$key=${!key}")
done
release_serving_docker_args
cuteafd_cdi_gpu_args "$gpus" || exit 2
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$coordinator_name" --restart no "${CUTEAFD_CDI_ARGS[@]}" --network host --ipc host \
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info "${shared_env_args[@]:0:4}" \
  "${bond_args[@]}" "${bench_nonce_env_args[@]}" "${tp2_env_args[@]}" "${graph_census_args[@]}" "${lane_env_args[@]}" "${thermal_args[@]}" \
  "${shared_env_args[@]:4}" -e "CUTEAFD_IMAGE=$coordinator_image" "${wip_mount_args[@]}" "${device_map_args[@]}" \
  -v "$hub:/hf/hub:ro" -v "$bench_dir:/run/cuteafd/bench" -v "$fidelity_cache:/run/cuteafd/fidelity" -e CUTEAFD_FIDELITY_CACHE=/run/cuteafd/fidelity \
  "${api_mount_args[@]}" "${chat_template_mounts[@]}" "${trace_args[@]}" "${probe_args[@]}" ${profile_args[@]+"${profile_args[@]}"} ${profile_wrapper_args[@]+"${profile_wrapper_args[@]}"} "$coordinator_image" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd "${coordinator_budget_args[@]}" $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so "${peer_args[@]}" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY "$default_concurrency")" --max-context "$(get MAX_CONTEXT_TOKENS "$default_context")" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" "${dspark_args[@]}" \
  "${family_args[@]}" "${draft_args[@]}" "${served_args[@]}" >/dev/null
url="http://127.0.0.1:${addr##*:}"
ready_deadline=$((SECONDS + 900))
until curl --max-time 5 -sf "$url/health" >/dev/null; do
  if (( SECONDS >= ready_deadline )); then
    echo "coordinator readiness timed out: $(curl --max-time 5 -s "$url/health" || true)" >&2
    docker logs --tail 30 "$coordinator_name" >&2
    exit 1
  fi
  docker ps -q -f "name=^$coordinator_name\$" | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 "$coordinator_name" >&2; exit 1; }
  sleep 2
done
echo "API ready at $url/v1/ ($(release_api_curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"

if ((console_supported)); then release_print_console_link "http://$(hostname):${addr##*:}"; fi
