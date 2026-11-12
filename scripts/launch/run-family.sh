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
# A failed bounded smoke teardown must be resolved before another launch or
# dry-run preflight can reuse this coordinator/fabric.
for cleanup_marker in "$HOME"/.cache/cuteafd/instances/*/unconfirmed-containers.json; do
  [[ ! -e "$cleanup_marker" ]] || release_die "unconfirmed container teardown: $cleanup_marker: $(cat "$cleanup_marker")"
done
namespace_root="$(readlink -f "${CUTEAFD_NAMESPACE_ROOT:-/mnt/sparknest}")"
namespace_hf_home="$namespace_root/hf-home"
# SparkNest is preferred whenever the selected namespace has a hub. An absent
# namespace selects the site's HF_HOME, including in isolated launcher tests.
site_hub=0
if [[ -d "$namespace_hf_home" ]]; then
  [[ -z "${CUTEAFD_NAMESPACE_ROOT:-}" ]] || export HF_HOME="$namespace_hf_home"
  container_hf_home="$namespace_hf_home"
else
  site_hub=1
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  container_hf_home="$HF_HOME"
fi
config="$repo_root/cuteafd.config"
restart=0
restart_all=0
family=""
embedding_override=""
wip_slot=""
dry_run=0
dry_run_total_capacity=0
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
    --total-capacity) dry_run_total_capacity=1; shift ;;
    *) echo "usage: $0 [--config FILE] [--family ID] [--set KEY=VALUE]... [--embedding-placement host|gpu] [--restart [--all]] [--wip SLOT] [--dry-run [--total-capacity]]" >&2; exit 2 ;;
  esac
done
((restart_all == 0 || restart == 1)) || release_die "--all requires --restart"
((restart == 0 || dry_run == 0)) || release_die "--restart and --dry-run are mutually exclusive"
((dry_run_total_capacity == 0 || dry_run == 1)) || release_die "--total-capacity requires --dry-run"
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
declare -A cfg cfg_file=()
while IFS='=' read -r key value; do
  release_known_key "$key" || release_die "unknown configuration key: $key"
  cfg[$key]="$value"
  # Only keys the file (or a launch override) set. Later expansion writes
  # SPARK_<rank>_* into cfg without the operator having named them.
  [[ -z "$value" ]] || cfg_file[$key]=1
done < <(grep -E '^[A-Z_0-9]+=' "$config")
[[ -z "${SPARK_COUNT:-}" ]] || { cfg[SPARK_COUNT]="$SPARK_COUNT"; cfg_file[SPARK_COUNT]=1; }
[[ -z "${SPARK_HOSTS:-}" ]] || { cfg[SPARK_HOSTS]="$SPARK_HOSTS"; cfg_file[SPARK_HOSTS]=1; }
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
# Refused before anything starts: a release never serves pages from a checkout.
case "$(get CONSOLE_ASSETS off)" in
  on) [[ -n "$wip_slot" ]] || release_die "CONSOLE_ASSETS=on serves pages from a checkout and needs --wip SLOT; release images serve their compiled-in pages" ;;
  off) ;;
  *) release_die "CONSOLE_ASSETS must be on or off" ;;
esac
# release_coordinator_device_args UUID...: the coordinator container's GPUs for
# every family. CUDA_VISIBLE_DEVICES lists them in logical order, primary first,
# so CUDA ordinal N is logical GPU N (entry/head and V4.1's larger ownership
# share on 0) whatever the host order, and plans' rtx:N owners need no mapping.
release_coordinator_device_args() {
  local csv
  csv="$(IFS=,; printf '%s' "$*")"
  cuteafd_cdi_gpu_args "$csv" || exit 2
  RELEASE_COORDINATOR_DEVICE_ARGS=("${CUTEAFD_CDI_ARGS[@]}" -e "CUDA_VISIBLE_DEVICES=$csv")
}
# key NEW OLD [DEFAULT]: a renamed key; the pre-rename spelling works for one
# release with a warning.
key() {
  if [[ -n "${cfg[$1]:-}" ]]; then printf '%s' "${cfg[$1]}"
  elif [[ -n "${cfg[$2]:-}" ]]; then echo "warning: config key $2 is deprecated; use $1" >&2; printf '%s' "${cfg[$2]}"
  else printf '%s' "${3:-}"; fi
}
source "$repo_root/scripts/lib/family-keys.sh"
source "$repo_root/scripts/lib/attention-placement.sh"
vision="$(get VISION auto)"
audio="$(get AUDIO auto)"
vision_replicas="$(get VISION_REPLICAS 1)"
[[ "$vision_replicas" =~ ^[1-8]$ ]] || release_die "VISION_REPLICAS must be 1..8"
[[ "$vision" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$vision" != auto:* && "$vision" != off:* ) ]] || release_die "VISION must be auto, off, rtx[:gpu] or spark[:rank]"
[[ "$audio" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$audio" != auto:* && "$audio" != off:* ) ]] || release_die "AUDIO must be auto, off, rtx[:gpu] or spark[:rank]"
# One device memory policy for every family (docs/options.md): caps, reserves
# and KV target/min/max become global cuteafd flags on the plan and the serve
# argv alike; the Spark cap and reserve also reach every Spark worker.
coordinator_budget="$(get GPU_MEMORY_CAP)"
memory_get() { get "$1"; }
release_memory_args memory_policy_args memory_get
# The plan takes the resolved per-GPU cap (the smaller of the card and the key) separately.
release_memory_args plan_memory_args memory_get GPU_MEMORY_CAP
release_memory_args spark_memory_args memory_get GPU_MEMORY_CAP GPU_RESERVE FAMILY_RESERVE MEMORY_PADDING GRAPH_DISABLE_LOW_MEM KV_TARGET_TOKENS KV_MIN_TOKENS KV_MAX_TOKENS
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
local feature_key feature_root
for feature_key in MEDIA_FEATURES_DIR AUDIO_FEATURES_DIR; do
  feature_root="$(get "$feature_key")"
  [[ -n "$feature_root" ]] || continue
  [[ "$feature_root" == /* && -d "$feature_root" ]] \
    || release_die "$feature_key must be an existing absolute directory"
  probe_args+=(-v "$feature_root:$feature_root:ro" -e "CUTEAFD_$feature_key=$feature_root")
done
}
# SPECULATION_TRACE=/abs/host/file.jsonl: the per-cycle speculation trace
# (CUTEAFD_SPECULATION_TRACE, written by serve-glm and serve-qwen4; read by
# scripts/qualify/glm5/glm-draft-trace.py and qualify/qwen4/qwen4-draft-trace.py).
shared_trace_args() {
trace_args=()
# Optional diagnostic length rule; production selection uses resource prices.
case "$(get DRAFT_SURVIVAL_LENGTHS off)" in
  on) trace_args+=(-e CUTEAFD_DRAFT_SURVIVAL_LENGTHS=1) ;;
  off) trace_args+=(-e CUTEAFD_DRAFT_SURVIVAL_LENGTHS=0) ;;
  *) echo "DRAFT_SURVIVAL_LENGTHS must be on or off" >&2; exit 2 ;;
esac
# Draft length policy (cuteafd_core::DraftMode): simple by default; simple-rc, simple-derived, or shared
# for the resource-priced one. Forwarded only when set.
case "$(get DRAFT_POLICY)" in
  "") ;;
  shared|simple|simple-rc|simple-derived) trace_args+=(-e "CUTEAFD_DRAFT_POLICY=$(get DRAFT_POLICY)") ;;
  *) echo "DRAFT_POLICY must be shared, simple, simple-rc or simple-derived" >&2; exit 2 ;;
esac
# How the simple draft rules price rows (linear by default); forwarded only when set.
case "$(get DRAFT_ROW_COST)" in
  "") ;;
  linear|power|log) trace_args+=(-e "CUTEAFD_DRAFT_ROW_COST=$(get DRAFT_ROW_COST)") ;;
  *) echo "DRAFT_ROW_COST must be linear, power or log" >&2; exit 2 ;;
esac
trace="$(get SPECULATION_TRACE)"
if [[ -n "$trace" ]]; then
  [[ "$trace" == /* ]] || { echo 'SPECULATION_TRACE must be an absolute path' >&2; exit 2; }
  mkdir -p "$(dirname "$trace")"
  release_require_writable_state --file "$trace"
  trace_args+=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_SPECULATION_TRACE=$trace")
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
# (gpu-powerd's /run/gpu-powerd, when present) is mounted read-only at the daemon's own
# state path so the report carries it. CUTEAFD_GPU_POWERD_DIR names the host directory the
# governor publishes in (a real host leaves it unset); script tests point it at a fixture
# directory so a launch line does not depend on which host runs the test.
# Spark workers sample their own GPU too (shared_worker_env_args).
shared_thermal_args() {
thermal="$(get THERMAL on)"
case "$thermal" in on|off) ;; *) echo "THERMAL must be on or off" >&2; exit 2 ;; esac
thermal_args=(-e "CUTEAFD_THERMAL=$thermal")
powerd_dir="${CUTEAFD_GPU_POWERD_DIR:-/run/gpu-powerd}"
[[ ! -d "$powerd_dir" ]] || thermal_args+=(-v "$powerd_dir:/run/gpu-powerd:ro")
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
# HOST_WAIT (block, default; or spin) is how serving's host threads wait:
# block spins 10 ms in each critical-path wait, then sleeps (the Spark exchange
# worker spins as long, then paces its polling): a less loaded machine; spin
# spins throughout, one busy core per waiting thread. IDLE_PARK_SECONDS
# (default 30): with no active request for that long every polling thread
# parks. Coordinator and Spark workers both read them.
shared_coordinator_env_args() {
intake="$(get SPARK_INTAKE auto)"
case "$intake" in auto|gpu|pinned|host) ;; *) echo "SPARK_INTAKE must be auto, gpu, pinned or host" >&2; exit 2 ;; esac
console_text="$(get CONSOLE_TEXT on)"
case "$console_text" in on|off) ;; *) echo "CONSOLE_TEXT must be on or off" >&2; exit 2 ;; esac
spark_prefill_reply_format="$(get SPARK_PREFILL_REPLY_FORMAT "")"
spark_decode_reply_format="$(get SPARK_DECODE_REPLY_FORMAT "")"
for phase in prefill decode; do
  key="spark_${phase}_reply_format"
  case "${!key}" in ""|bf16|fp8|nvfp4) ;; *) release_die "SPARK_${phase^^}_REPLY_FORMAT must be bf16, fp8 or nvfp4" ;; esac
done
fp8_prefill="$(get FP8_EXPERT_PREFILL auto)"
case "$fp8_prefill" in auto|w8a8|w8a16) ;; *) echo "FP8_EXPERT_PREFILL must be auto, w8a8 or w8a16" >&2; exit 2 ;; esac
attention_plan_rows="$(get ATTENTION_PLAN_ROWS auto)"
if [[ $attention_plan_rows != auto && ! $attention_plan_rows =~ ^[1-9][0-9]*$ ]]; then
  release_die "ATTENTION_PLAN_ROWS must be auto or a positive integer"
fi
shared_env_args=(-e "HF_HOME=$container_hf_home" -e "CUTEAFD_SPARK_INTAKE=$intake"
  -e "CUTEAFD_CONSOLE_TEXT=$([[ $console_text == on ]] && echo true || echo false)"
  -e "CUTEAFD_ATTENTION_PLACEMENT=$attention_placement" -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill"
  -e "CUTEAFD_SPARK_ADMISSION_TIMEOUT_SECONDS=${CUTEAFD_RELEASE_READY_TIMEOUT_SECONDS:-900}")
if [[ -n $spark_prefill_reply_format ]]; then
  shared_env_args+=(-e "CUTEAFD_SPARK_PREFILL_REPLY_FORMAT=$spark_prefill_reply_format")
fi
if [[ -n $spark_decode_reply_format ]]; then
  shared_env_args+=(-e "CUTEAFD_SPARK_DECODE_REPLY_FORMAT=$spark_decode_reply_format")
fi
if [[ $attention_plan_rows != auto ]]; then
  shared_env_args+=(-e "CUTEAFD_ATTENTION_PLAN_ROWS=$attention_plan_rows")
fi
if [[ -n "${CUTEAFD_PUSH_SOCKET:-}" ]]; then
  shared_env_args+=(-e "CUTEAFD_PUSH_SOCKET=$CUTEAFD_PUSH_SOCKET")
fi
# Unset keys leave the binary defaults (block, 30 s).
wait_env_args=()
host_wait="$(get HOST_WAIT "")"
if [[ -n $host_wait ]]; then
  case "$host_wait" in spin|block) ;; *) release_die "HOST_WAIT must be spin or block" ;; esac
  wait_env_args+=(-e "CUTEAFD_HOST_WAIT=$host_wait")
fi
idle_park="$(get IDLE_PARK_SECONDS "")"
if [[ -n $idle_park ]]; then
  [[ $idle_park =~ ^[0-9]+(\.[0-9]+)?$ && ! $idle_park =~ ^0+(\.0+)?$ ]] \
    || release_die "IDLE_PARK_SECONDS must be a positive number of seconds"
  wait_env_args+=(-e "CUTEAFD_IDLE_PARK_SECONDS=$idle_park")
fi
shared_env_args+=("${wait_env_args[@]}")
shared_worker_env_args=(-e "HF_HOME=$container_hf_home" -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill" -e "CUTEAFD_THERMAL=$(get THERMAL on)"
  "${wait_env_args[@]}")
if [[ -n "${CUTEAFD_PUSH_SOCKET:-}" ]]; then
  shared_worker_env_args+=(-e "CUTEAFD_PUSH_SOCKET=$CUTEAFD_PUSH_SOCKET")
fi
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
# CUTEAFD_DIAG_CUDA_LAUNCH_BLOCKING=1 makes CUDA report a kernel fault at the
# faulting launch instead of a later API call (diagnostic only; slow).
[[ "${CUTEAFD_DIAG_CUDA_LAUNCH_BLOCKING:-}" != 1 ]] || bench_nonce_env_args+=(-e CUDA_LAUNCH_BLOCKING=1)
# CUTEAFD_DIAG_CUDA_DEVICE_MAX_CONNECTIONS / _MAX_COPY_CONNECTIONS=N (1-32) set
# CUDA's hardware work-queue counts for the coordinator (A/B only).
local key value
for key in CONNECTIONS COPY_CONNECTIONS; do
  value=$(printenv "CUTEAFD_DIAG_CUDA_DEVICE_MAX_$key") || continue
  [[ "$value" =~ ^[0-9]+$ ]] && (( value >= 1 && value <= 32 )) \
    || { echo "run-family: CUTEAFD_DIAG_CUDA_DEVICE_MAX_$key must be 1-32, got '$value'" >&2; exit 2; }
  bench_nonce_env_args+=(-e "CUDA_DEVICE_MAX_$key=$value")
done
}
# PROFILE=switch[,switch...] turns on profiling for one launch, the same way in
# every family, through release-common.sh:
#   census            CUTEAFD_GRAPH_CENSUS=1 (graph capture/replay census) and
#                     CUTEAFD_STARTUP_MEMORY_PROBE=1 (the startup memory ledger:
#                     free/tracked/untracked bytes per device at each load stage,
#                     logged as `startup memory probe`). With PROFILE_DIR the
#                     native per-capture records go to
#                     PROFILE_DIR/graph-census-captures-HOST.jsonl (without it,
#                     /tmp/cuteafd-graph-census-captures.jsonl in the container).
#   transport-timing  CUTEAFD_PROTOCOL_V2_TCP_TIMING=1 (expert exchange timing)
#   expert-timing     CUTEAFD_EXPERT_TIMING=1 (Spark worker expert compute timing)
#   expert-routes     CUTEAFD_EXPERT_ROUTES=1 (needs PROFILE_DIR): the coordinator logs every
#                     serving Spark wave's top-k expert ids, one JSONL record per (step,
#                     MoE layer), to PROFILE_DIR/expert-routes-HOST.jsonl (format: the
#                     cuteafd-transport route_log module; scored offline by
#                     scripts/bench/expert-route-balance.py). GLM 5.3 also saves the MoE
#                     input rows (BF16, up to 64 per layer-step) of the first
#                     PROFILE_EXPERT_INPUT_SAMPLE (default 64, 0 = none) serving
#                     layer-steps as PROFILE_DIR/expert-input-HOST-NNNNN.safetensors.
#                     PROFILE_EXPERT_ROUTE_WEIGHTS=on adds each narrow wave's gate weights.
#   draft-timing      CUTEAFD_DRAFT_TIMING=1 (drafter device and host launch spans)
#   nsys              the coordinator runs under `nsys profile` (cuda, nvtx, osrt, CUDA
#                     graph nodes; no CPU sampling), collection paused until
#                     `docker exec COORDINATOR nsys start --session=cuteafd-profile`
#                     and written by `... nsys stop --session=cuteafd-profile` (or a
#                     `docker stop`, never `docker rm -f`) to PROFILE_DIR/nsys-HOST.nsys-rep.
#                     wip-cards opens the window after the sweep's first warm wave and
#                     closes it at sweep end: `--set CARD:PROFILE=nsys --levels ...`.
#                     PROFILE_NSYS_GRAPH_TRACE=graph traces whole graphs instead of
#                     individual nodes (default node).
#                     PROFILE_NSYS_CPU=on adds CPU sampling and context switches
#                     (host-side stalls; perturbs host timing).
#                     PROFILE_NSYS_GPU_METRICS=all also samples every GPU's clocks,
#                     power and DRAM/SM activity into the same report (an index
#                     list uses nsys's device order, not CUDA's).
#   ncu               the coordinator runs under `ncu`, replaying PROFILE_NCU_COUNT
#                     (default 20) launches whose name matches the required regex
#                     PROFILE_NCU_KERNEL after skipping the first PROFILE_NCU_SKIP
#                     matching launches (default 0; `ncu --launch-skip`, so the
#                     replayed ones are steady-state serving, not readiness),
#                     collecting section set PROFILE_NCU_SET (default full), into
#                     PROFILE_DIR/ncu-HOST.ncu-rep, written when the coordinator
#                     stops (`docker stop`).
# nsys and ncu wrap the coordinator (one of them per launch); Spark workers get
# the other switches. PROFILE_SPARK=nsys|ncu wraps Spark expert workers the same
# way, independently of PROFILE (coordinator and Sparks in one launch, or either
# alone; it needs PROFILE_DIR too):
#   PROFILE_SPARK_RANKS  comma list of the wrapped Spark ranks (default 0: every
#                     profiled rank perturbs the exchange timing).
#   nsys              paused under session cuteafd-profile in each wrapped worker;
#                     wip-cards opens and closes it with the coordinator's window
#                     (`ssh HOST docker exec WORKER nsys start|stop --session=...`).
#                     PROFILE_SPARK_NSYS_GPU_METRICS=all samples the GB10's metrics.
#                     Report: PROFILE_DIR/nsys-HOST-rankR.nsys-rep on that Spark.
#   ncu               PROFILE_SPARK_NCU_KERNEL (required regex), PROFILE_SPARK_NCU_COUNT
#                     (default 20), PROFILE_SPARK_NCU_SKIP (default 0) and
#                     PROFILE_SPARK_NCU_SET (default full), as for the coordinator.
#                     Report: PROFILE_DIR/ncu-HOST-rankR.ncu-rep, written when the
#                     worker stops (`docker stop`, never `docker rm -f`).
# A wrapped container carries the cuteafd.graceful-stop label (SIGINT, 300 s), so
# ./stop.sh and smoke teardown stop it gracefully before removing it. Both
# profilers run as the host user; they need the driver's
# RmProfilingAdminOnly=0 and no extra capabilities. The binary is resolved inside
# the container at start, not from the host's install: /usr/local/bin/<tool> if
# it is there, otherwise the newest versioned copy (ncu under
# /opt/nvidia/nsight-compute/<version>/ncu; nsys also accepts
# /usr/local/cuda/bin/nsys, then /opt/nvidia/nsight-systems/<version>/nsys).
# Every profiled container also gets CUTEAFD_PROFILE_HOST (the physical host,
# never the container id). PROFILE_DIR=/abs/dir is created on each host, mounted
# at the same path and passed as CUTEAFD_PROFILE_DIR; writers name files by host
# and rank. wip-cards owns PROFILE_DIR and copies Spark files back.
shared_profile_args() {
release_profile_docker_args "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)" "$(get PROFILE_SPARK)"
profile_args=(${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"})
# Only the coordinator owns this observer; Spark workers must not subtract it.
if [[ -n "$(get PROFILE_DIR)" && -f "$(get PROFILE_DIR)/observer-context.json" ]]; then
  profile_args+=(-e "CUTEAFD_OBSERVER_CONTEXT_FILE=$(get PROFILE_DIR)/observer-context.json")
fi
release_profile_sample_args "$(get PROFILE)" "$(get PROFILE_EXPERT_INPUT_SAMPLE)" "$(get PROFILE_EXPERT_ROUTE_WEIGHTS)"
profile_args+=(${RELEASE_PROFILE_SAMPLE_ARGS[@]+"${RELEASE_PROFILE_SAMPLE_ARGS[@]}"})
RELEASE_PROFILE_NSYS_GRAPH_TRACE="$(get PROFILE_NSYS_GRAPH_TRACE)" \
RELEASE_PROFILE_NSYS_CPU="$(get PROFILE_NSYS_CPU)" release_profile_wrapper "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)" coordinator \
  "$(get PROFILE_NCU_KERNEL)" "$(get PROFILE_NCU_COUNT)" "$(get PROFILE_NCU_SET)" "$(get PROFILE_NSYS_GPU_METRICS)" "$(get PROFILE_NCU_SKIP)"
profile_wrapper=(${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"})
profile_wrapper_args=(${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]+"${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}"})
[[ -z "$(get PROFILE_DIR)" ]] || { mkdir -p "$(get PROFILE_DIR)"; release_require_writable_state "$(get PROFILE_DIR)"; }
}
# Sets spark_profile_wrapper and spark_profile_wrapper_args for one Spark rank
# (empty unless PROFILE_SPARK selects it). Arguments: HOST RANK WORLD.
spark_profile_wrapper_for() {
release_profile_spark_wrapper "$(get PROFILE_SPARK)" "$(get PROFILE_SPARK_RANKS)" "$(get PROFILE_DIR)" "$1" "$2" "$3" \
  "$(get PROFILE_SPARK_NCU_KERNEL)" "$(get PROFILE_SPARK_NCU_COUNT)" "$(get PROFILE_SPARK_NCU_SET)" \
  "$(get PROFILE_SPARK_NCU_SKIP)" "$(get PROFILE_SPARK_NSYS_GPU_METRICS)"
spark_profile_wrapper=(${RELEASE_PROFILE_WRAPPER[@]+"${RELEASE_PROFILE_WRAPPER[@]}"})
spark_profile_wrapper_args=(${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]+"${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}"})
}
shared_graph_census_args() {
graph_census_args=()
if [[ "$family" == qwen3_5 ]]; then
  case "$(get DECODE_GRAPHS on)" in
    on) graph_census_args+=(-e CUTEAFD_QWEN35_GRAPHS=1) ;;
    off) graph_census_args+=(-e CUTEAFD_QWEN35_GRAPHS=0) ;;
    *) release_die "DECODE_GRAPHS must be on or off" ;;
  esac
fi
if [[ "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5 || "$family" == glm5_flash || "$family" == qwen4 || "$family" == deepseek_v4 || "$family" == deepseek_v41 || "$family" == qwen3_5 || "$family" == muse_spark || "$family" == step5 ]]; then
  local key graph_trace
  for key in GPU_GRAPH_RESERVE GPU_GRAPH_RESERVE_EXTRA GPU_GRAPH_RESERVE_EXTRA_CAP; do
    [[ -z "$(get "$key")" ]] || graph_census_args+=(-e "CUTEAFD_$key=$(get "$key")")
  done
  graph_trace="$(get GRAPH_TRACE)"
  if [[ -n "$graph_trace" ]]; then
    [[ "$graph_trace" == /* && -d "$(dirname "$graph_trace")" ]] || { printf '%s\n' 'GRAPH_TRACE needs an absolute path in an existing directory' >&2; exit 2; }
    graph_census_args+=(-v "$(dirname "$graph_trace"):/run/cuteafd-graph-trace" -e "CUTEAFD_GRAPH_TRACE=/run/cuteafd-graph-trace/$(basename "$graph_trace")")
  fi
fi
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
# A store deleted under a running container comes back root-owned (Docker
# recreates missing bind sources) and the dashboard then fails with EACCES.
release_require_owned_state "$bench_dir" "$fidelity_cache"
mkdir -p "$fidelity_cache" "$bench_dir"
release_require_writable_state "$bench_dir" "$fidelity_cache"
api_args=() api_mount_args=()
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
  # Locked pages show the operator how to print the unlock link; never the secret.
  console_port="${ADDR:-${addr:-0.0.0.0:8000}}"
  api_mount_args+=(-e "CUTEAFD_CONSOLE_LINK_COMMAND=$(release_console_link_command "http://$(hostname):${console_port##*:}")")
fi
if [[ -n "$API_KEY_FILE" ]]; then
  [[ -f "$API_KEY_FILE" && -r "$API_KEY_FILE" ]] || { echo "API_KEY_FILE must name a readable file" >&2; exit 2; }
  API_KEY_FILE="$(readlink -f "$API_KEY_FILE")"
  api_mount_args+=(--mount "type=bind,src=$API_KEY_FILE,dst=/run/cuteafd-api-key,readonly")
  api_args+=(--api-key-file /run/cuteafd-api-key)
fi
# CONSOLE_ASSETS=on (WIP launches only) serves the dashboard pages and their
# shared assets from this checkout, re-read on every page load: edit, save,
# reload the browser, with no rebuild or restart. A release image always serves
# its compiled-in pages.
if [[ "$(get CONSOLE_ASSETS off)" == on ]]; then
  for asset_dir in cuteafd-api cuteafd-bench cuteafd-usage; do
    api_mount_args+=(--mount "type=bind,src=$repo_root/rust/crates/$asset_dir/assets,dst=/run/cuteafd-assets/$asset_dir,readonly")
  done
  api_mount_args+=(-e CUTEAFD_CONSOLE_PAGE=/run/cuteafd-assets/cuteafd-api/console.html
    -e CUTEAFD_CONSOLE_ASSETS=/run/cuteafd-assets/cuteafd-api
    -e CUTEAFD_BENCH_PAGE=/run/cuteafd-assets/cuteafd-bench/bench.html
    -e CUTEAFD_USAGE_PAGE=/run/cuteafd-assets/cuteafd-usage/usage.html)
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
# V4.1 code. Engram tables need no mount of their own: they are read from the
# checkpoint under the full namespace mount, kept local on raptor by sparknest
# (ENGRAM_WARM below optionally warms them into the page cache after readiness).
# ---------------------------------------------------------------------------
launch_deepseek_v41() {
# The release loader validates every key (run.sh --set overrides are in $config).
release_load_config "$config"
release_memory_args memory_policy_args release_memory_variable
release_memory_args plan_memory_args release_memory_variable GPU_MEMORY_CAP
release_memory_args spark_memory_args release_memory_variable GPU_MEMORY_CAP GPU_RESERVE FAMILY_RESERVE MEMORY_PADDING GRAPH_DISABLE_LOW_MEM KV_TARGET_TOKENS KV_MIN_TOKENS KV_MAX_TOKENS
# V4.1 serves its established ranges until its runtime consumes solver placement.
attention_placement="${ATTENTION_PLACEMENT:-auto}"
release_check_attention_placement "$attention_placement" deepseek_v41
case "$EMBEDDING" in host|gpu) ;; *) release_die "EMBEDDING must be host or gpu" ;; esac
if [[ -n "$wip_slot" ]]; then
  [[ "$wip_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || release_die "invalid WIP slot name: $wip_slot"
  # WIP artifacts run inside the development images with a release-shaped
  # /opt/cuteafd layout bind-mounted from a per-host staging directory.
  COORDINATOR_DOCKER_INFERENCE="$COORDINATOR_DOCKER_DEV"
  SPARK_EXPERT_DOCKER_INFERENCE="$SPARK_EXPERT_DOCKER_DEV"
fi
release_validate_tp2_options
[[ -z "$HTTP_QUEUE_DEPTH" || ( "$HTTP_QUEUE_DEPTH" =~ ^[1-9][0-9]*$ && "$HTTP_QUEUE_DEPTH" -le 4096 ) ]] || release_die "HTTP_QUEUE_DEPTH must be in 1..4096"
[[ "$HTTP_QUEUE_WAIT_MS" =~ ^[0-9]+$ ]] || release_die "HTTP_QUEUE_WAIT_MS must be non-negative"
[[ "$HOST_CACHE_BYTES" == auto || "$HOST_CACHE_BYTES" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "HOST_CACHE_BYTES must be auto, 0, or a byte size"
case "$SPECULATOR" in dspark|off) ;; *) release_die "SPECULATOR must be dspark or off for DeepSeek V4.1" ;; esac
# Unset preserves V4.1's adaptive five-token maximum.
draft_length="$DRAFT_LENGTH"
[[ -z "$draft_length" || "$draft_length" =~ ^[1-9][0-9]*$ ]] ||
  release_die "DRAFT_LENGTH must be a positive integer"
release_validate_verbs_device_map
# V41_FP8_HEAD: unset/auto is resolved once by the shared loader parser to
# packed (exact BF16 target weights, coarse dSpark plane), on one or two RTX.
# all keeps one E4M3 copy, off retains BF16, draft retains BF16 plus draft FP8.
# Forward the same unresolved setting to planner and server; do not duplicate
# their representation policy in the launcher.
v41_fp8_head_args=()
case "${V41_FP8_HEAD:-}" in
  "") ;;
  auto|off|draft|all|packed)
    [[ -z "${CUTEAFD_V41_FP8_HEAD:-}" || "$CUTEAFD_V41_FP8_HEAD" == "$V41_FP8_HEAD" ]] ||
      release_die "V41_FP8_HEAD=$V41_FP8_HEAD conflicts with CUTEAFD_V41_FP8_HEAD=$CUTEAFD_V41_FP8_HEAD in the environment; set one"
    v41_fp8_head_args=(-e "CUTEAFD_V41_FP8_HEAD=$V41_FP8_HEAD") ;;
  *) release_die "V41_FP8_HEAD must be auto, off, draft, all or packed" ;;
esac
# CUDA_MODULE_LOADING (LAZY or EAGER; unset: the CUDA default, LAZY), coordinator
# only. A lazily loaded kernel's first launch waits for the device to idle, which
# never happens behind a stream wait only the engine thread releases (a cold 5090
# card hung in a first-shape cuBLAS draft GEMM). EAGER loads every library kernel
# at startup instead: about 17 s and 212 MiB more on an RTX 5090.
v41_module_loading_args=()
case "$(get CUDA_MODULE_LOADING)" in
  "") ;;
  LAZY|EAGER) v41_module_loading_args=(-e "CUDA_MODULE_LOADING=$(get CUDA_MODULE_LOADING)") ;;
  *) release_die "CUDA_MODULE_LOADING must be LAZY or EAGER" ;;
esac
# L2_PREFETCH (off, auto = 3/4 of the L2, or MiB; unset: off): one-lane decode and
# verify passes pull the next layer's weights into L2 while each layer's routed
# experts are on the Sparks (serve-native --l2-prefetch).
v41_l2_prefetch="$(get L2_PREFETCH)"
[[ -z "$v41_l2_prefetch" || "$v41_l2_prefetch" =~ ^(off|auto|[0-9]+)$ ]] ||
  release_die "L2_PREFETCH must be off, auto or a MiB count"
# ENGRAM_WARM (on|off; default off): after readiness, read both Engram tables once
# into the page cache on a background thread (serve-native --engram-warm), for hosts
# with RAM for the tables whose checkpoint store has slow random reads.
# MAPPED_SCALE_PRELOAD_GIB (GiB; default 0) caps host memory for preloaded Engram
# tables: a cap that holds every table, rows and scales (about 189 GiB for the official
# checkpoint), loads them all before readiness, so no row is read from storage again;
# a smaller cap that holds the scales (about 6 GiB) preloads those alone.
v41_engram_warm="$(get ENGRAM_WARM off)"
case "$v41_engram_warm" in on|off) ;; *) release_die "ENGRAM_WARM must be on or off" ;; esac
case "$RTX_GPUS" in auto|1|2) ;; *) release_die "RTX_GPUS must be auto, 1, or 2" ;; esac
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" == all || "$RTX_EXPERT_LAYERS" =~ ^[0-9]+$ ]] ||
  release_die "DeepSeek V4.1 RTX_EXPERT_LAYERS must be auto, all or a non-negative layer count"
# V4.1 has 40 routed-expert layers, so `all` is exactly 40 here.
[[ "$RTX_EXPERT_LAYERS" != all ]] || RTX_EXPERT_LAYERS=40
((!RELEASE_CONCURRENCY_EXPLICIT)) || [[ "$CONCURRENCY" =~ ^([1-9]|[1-5][0-9]|6[0-4])$ ]] || release_die "CONCURRENCY must be in 1..64"
[[ "$PREFIX_CACHE_ENTRIES" == auto || "$PREFIX_CACHE_ENTRIES" =~ ^[0-9]+$ ]] || release_die "PREFIX_CACHE_ENTRIES must be auto or non-negative"
[[ "$MAX_CONTEXT_TOKENS" == auto ]] || { [[ "$MAX_CONTEXT_TOKENS" =~ ^[1-9][0-9]{0,6}$ ]] && ((MAX_CONTEXT_TOKENS <= 1048576)); } || release_die "MAX_CONTEXT_TOKENS must be auto or in 1..1048576"
[[ "$MAX_OUTPUT_TOKENS" =~ ^-?[0-9]+$ ]] || release_die "MAX_OUTPUT_TOKENS must be an integer (<= 0: bounded only by the context)"
[[ "$PREFILL_BATCH_TOKENS" =~ ^[0-9]+$ ]] && ((PREFILL_BATCH_TOKENS >= 80 && PREFILL_BATCH_TOKENS <= 4096)) || release_die "PREFILL_BATCH_TOKENS must be in 80..4096"
[[ -z "$KV_POOL_SIZE" || "$KV_POOL_SIZE" =~ ^[0-9]+([.][0-9]{1,6})?(B|MB|GB|MiB|GiB)?$ ]] || release_die "KV_POOL_SIZE has an invalid unit"
((restart == 0 || dry_run == 0)) || release_die "--restart and --dry-run are mutually exclusive"

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
hf_home="${HF_HOME:-$namespace_root/hf-home}"
release_resolve_local_model_revision "$hf_home"
# COORDINATOR_GPUS (ordered, primary first) or the lowest-index GPU is logical
# GPU 0; with an explicit pair, the selection below checks that pair and never
# substitutes the free-memory pick.
release_resolve_coordinator_gpus "$COORDINATOR_GPUS" "$RTX_GPUS"
profile_bytes="$(release_cuda_total_bytes "$RELEASE_COORDINATOR_GPU_UUID")" || release_die "cannot query coordinator CUDA capacity"
profile="$(release_coordinator_profile "$profile_bytes" "$GPU_MEMORY_CAP")" || release_die "cannot resolve coordinator capacity profile"
read -r profile_cap default_concurrency <<<"$profile"
((RELEASE_CONCURRENCY_EXPLICIT)) || CONCURRENCY="$default_concurrency"
[[ "$PREFIX_CACHE_ENTRIES" != auto ]] || PREFIX_CACHE_ENTRIES=$((CONCURRENCY <= 16 ? 20 : (5 * CONCURRENCY + 3) / 4))
if ((profile_cap <= 32 << 30)); then
  if [[ -z "${cfg_overrides[EMBEDDING]:-}" ]] && ! grep -Eq '^[[:space:]]*EMBEDDING[[:space:]]*=[[:space:]]*[^#[:space:]]' "$config"; then
    EMBEDDING=host
  fi
fi
snapshot_rel="hub/models--${RELEASE_MODEL_ID//\//--}/snapshots/$RELEASE_MODEL_REVISION"
release_hub_mounts "$namespace_root" "$site_hub" "$hf_home" "$snapshot_rel"
namespace_mount_args=("${RELEASE_HUB_MOUNTS[@]}")
[[ "$(jq -r '.quantization_config.quant_method == "exl3"' "$hf_home/$snapshot_rel/config.json")" != true ]] ||
  release_die "unsupported: EXL3 for DeepSeek V4/V4.1 Flash"
coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"
[[ "$TP2_DSPARK_EXPERTS" != on || "$SPECULATOR" == dspark ]] || release_die "TP2_DSPARK_EXPERTS requires dSpark"

RELEASE_RECLAIM_ARGS=()
((restart == 0)) || release_coordinator_reclaim_args "$coordinator"
minimum_expert_layers=1
[[ "$RTX_EXPERT_LAYERS" == auto || "$RTX_EXPERT_LAYERS" == 0 ]] || minimum_expert_layers="$RTX_EXPERT_LAYERS"
expert_format=native
model_is_nvfp4="$(jq -r '.quantization_config.moe_quant_algo // empty' "$hf_home/$snapshot_rel/config.json" 2>/dev/null || true)"
[[ "$model_is_nvfp4" == "NVFP4" ]] && expert_format=nvfp4
gpu_selection_mode="$RTX_GPUS"
# A 32 GB card serves V4.1 alone unless COORDINATOR_GPUS names the pair.
if [[ "$gpu_selection_mode" == auto && "$COORDINATOR_GPUS" != *,* ]] &&
  ((profile_cap <= 32 << 30)); then
  gpu_selection_mode=1
fi
compact_selection_args=()
((dry_run_total_capacity == 0)) || compact_selection_args+=(--total-capacity)
gpu_selection="$(python3 "$repo_root/scripts/lib/coordinator-gpus.py" select \
  --selectors "$COORDINATOR_GPUS" \
  --mode "$gpu_selection_mode" \
  --minimum-expert-layers "$minimum_expert_layers" \
  --concurrency "$CONCURRENCY" \
  --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --retained-turns "$PREFIX_CACHE_ENTRIES" \
  --kv-pool-size "$KV_POOL_SIZE" \
  --gpu-memory-cap "$GPU_MEMORY_CAP" \
  --expert-format "$expert_format" \
  "${compact_selection_args[@]}" ${RELEASE_RECLAIM_ARGS[@]+"${RELEASE_RECLAIM_ARGS[@]}"})" || exit 2
RELEASE_RTX_GPUS="$(jq -r '.count' <<<"$gpu_selection")"
release_resolve_tp2_dspark_experts deepseek_v41 "$RELEASE_RTX_GPUS"
if release_tp2_enabled; then
  ((RELEASE_RTX_GPUS == 2)) || release_die "TP2 options require two selected RTX GPUs; use --rtx-gpus 2"
fi
mapfile -t release_gpu_uuids < <(jq -r '.gpus[].uuid' <<<"$gpu_selection")
mapfile -t release_gpu_indices < <(jq -r '.gpus[].index' <<<"$gpu_selection")
mapfile -t release_gpu_pci < <(jq -r '.gpus[].pci' <<<"$gpu_selection")
gpu_uuid_csv="$(IFS=,; echo "${release_gpu_uuids[*]}")"
gpu_index_csv="$(IFS=,; echo "${release_gpu_indices[*]}")"
gpu_pci_csv="$(IFS=,; echo "${release_gpu_pci[*]}")"
release_coordinator_device_args "${release_gpu_uuids[@]}"

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

case "${PREFILL_PACKING:-}" in
  ""|unrestricted|exact) ;;
  *) release_die "PREFILL_PACKING must be unrestricted or exact" ;;
esac
# Select and admit the count before touching any serving container or Spark.
count_args=(--available-sparks "$RELEASE_AVAILABLE_SPARKS" --mapped-scale-preload-gib "$(get MAPPED_SCALE_PRELOAD_GIB 0)")
# Pass loader-resolved values: the planner's plain config parser does not parse
# shell quoting/comments or supply the loader's default Spark inventory.
count_config_text="$(printf '%s=%s\n' TP2_DSPARK_EXPERTS "$TP2_DSPARK_EXPERTS" SPECULATOR "$SPECULATOR" DRAFT_LENGTH "$draft_length" EXPERT_USAGE "$(get EXPERT_USAGE on)"
  for ((rank = 0; rank < RELEASE_AVAILABLE_SPARKS; rank++)); do
    host_key="SPARK_${rank}_HOST"
    printf '%s=%s\n' "$host_key" "${!host_key}"
  done)"
count_config_dir="$HOME/.cache/cuteafd/run-config"
mkdir -p "$count_config_dir"
count_config="$count_config_dir/plan-$(printf '%s\n' "$count_config_text" | sha256sum | cut -c1-16).config"
printf '%s\n' "$count_config_text" >"$count_config.tmp.$$" && mv "$count_config.tmp.$$" "$count_config"
count_args+=(--config /run/cuteafd-plan.config)
v41_decode_lanes="$(get DECODE_LANES)"
[[ -z "$v41_decode_lanes" || "$v41_decode_lanes" =~ ^[12]$ ]] || release_die "DECODE_LANES must be 1 or 2"
[[ -z "$v41_decode_lanes" ]] || count_args+=(--decode-lanes "$v41_decode_lanes")
((RELEASE_SPARK_COUNT_EXPLICIT == 0)) || count_args+=(--spark-ranks "$SPARK_COUNT")
# Keep checkpoint paths unchanged inside the full namespace mount.
count_mounts=(-e "HF_HOME=$container_hf_home" ${v41_fp8_head_args[@]+"${v41_fp8_head_args[@]}"} "${namespace_mount_args[@]}"
  -v "$count_config:/run/cuteafd-plan.config:ro")
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
v41_prefill_waves="$(get PREFILL_WAVES)"
[[ -z "$v41_prefill_waves" || "$v41_prefill_waves" =~ ^[1-4]$ ]] || release_die "PREFILL_WAVES must be 1 to 4 for DeepSeek V4.1"
[[ -z "$v41_prefill_waves" ]] || count_args+=(--prefill-waves "$v41_prefill_waves")
[[ "$SPECULATOR" != off ]] || count_args+=(--native-mtp-layers 0)
[[ -z "$KV_POOL_SIZE" ]] || count_args+=(--kv-pool-size "$KV_POOL_SIZE")
count_args+=(--speculator "$SPECULATOR")
[[ -z "$draft_length" ]] || count_args+=(--draft-length "$draft_length")
deployment_peers=()
for ((rank = 0; rank < RELEASE_AVAILABLE_SPARKS; rank++)); do
  lane_key="SPARK_${rank}_LANE_A"
  deployment_peers+=("${!lane_key}:$EXPERT_PORT")
done
((${#deployment_peers[@]} == 0)) || count_args+=(--deployment-peers "$(IFS=,; echo "${deployment_peers[*]}")")
release_reject_unused_keys deepseek_v41 $(_v41_allowed_keys)
readiness_args=(--require-ready)
[[ -z "$wip_slot" ]] || readiness_args=()
# Plan and serve admit resident weights against the same resolved budget.
v41_weight_budget_gib="$(release_gib "$profile_cap")"
count_plan="$(docker run --rm --network none "${count_mounts[@]}" "$COORDINATOR_DOCKER_INFERENCE" \
  "${count_command[@]}" plan "$(readlink -f "$hf_home/$snapshot_rel")" --vision "$VISION" --audio off \
  --deployment "${readiness_args[@]}" "${count_args[@]}" --rtx "$RELEASE_RTX_GPUS" \
  "${plan_memory_args[@]}" --gpu-memory-cap "$profile_cap" --coordinator-weight-budget-gib "$v41_weight_budget_gib" \
  --pool-tokens "$count_pool" --prefill-rows "$count_prefill" --max-context-tokens "$MAX_CONTEXT_TOKENS" \
  --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --host-cache-bytes "$HOST_CACHE_BYTES")"
release_validate_deployment "$count_plan" deepseek_v41
release_deployment_spark_roles "$count_plan" "$SPARK_COUNT"
release_admit_coordinator_gpus "$gpu_uuid_csv" "$count_plan" ${RELEASE_RECLAIM_ARGS[@]+"${RELEASE_RECLAIM_ARGS[@]}"}
resolved_decode_lanes="$(jq -er ' .decode_lanes | select(. == 2)' <<<"$count_plan")" ||
  release_die "DeepSeek V4.1 deployment requires two decode lanes"
selected_count="$(jq -er '.spark_ranks' <<<"$count_plan")"
if ((RELEASE_SPARK_COUNT_EXPLICIT)); then
  ((selected_count <= SPARK_COUNT)) || release_die "planner requested more Spark ranks than configured: $selected_count > $SPARK_COUNT"
  [[ "$selected_count" == "$SPARK_COUNT" ]] || echo "Spark roles use $selected_count of $SPARK_COUNT configured ranks"
fi
if ((spark_expert_world > 0)); then
  v41_worker_budget="$(release_spark_worker_budget "$count_plan" "$spark_expert_world")" || exit 1
fi
release_prefill_shape "$count_plan" deepseek_v41
PREFILL_BATCH_TOKENS="$PREFILL_ROWS"
V41_PREFILL_WAVES="$PREFILL_WAVES"
# The reserve the units cut (the standard 8192, or the planner's fit cap).
V41_PREFILL_RESERVE_ROWS="$(jq -er '.prefill_reserve_rows | select(type == "number" and . > 0)' <<<"$count_plan")" ||
  release_die "the deepseek_v41 deployment plan names no prefill reserve"
spark_tp="$(jq -er '.topology.tp' <<<"$count_plan")"
spark_tp_roles_required="$(jq -r '.artifact_requirements.spark_tp_role // ""' <<<"$count_plan")"
SPECULATOR="$(jq -er '.speculator' <<<"$count_plan")"
KV_POOL_SIZE="$(jq -r '.admission.kv_pool_size // ""' <<<"$count_plan")"
[[ -n "$POOL_TOKENS" || -n "$KV_POOL_SIZE" ]] || POOL_TOKENS=auto

expert_capacity="$(release_v41_expert_capacity "$PREFILL_BATCH_TOKENS")" ||
  release_die "cannot resolve expert row capacity"
mapfile -t hosts < <(release_spark_values HOST)
mapfile -t lanes < <(release_spark_values LANE_A)
for i in "${spark_worker_ranks[@]}"; do
  host="${hosts[$i]}"
  spark_manifest="$(release_ssh -o ConnectTimeout=10 "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$engine_commit" "$sparkinfer_commit" "$snapshot_rel" "$host" "${wip_slot:-__none__}" "${wip_spark_container:-__none__}" "${WIP_LAYOUT_SLOT:-__none__}" "${WIP_ROOT:-__default__}" "${WIP_INSTANCE:-__none__}" "$namespace_root" "$hf_home" <<'REMOTE'
set -euo pipefail
image="$1"; engine="$2"; sparkinfer="$3"; snapshot_rel="$4"; host="$5"
wip_slot="${6:-__none__}"
wip_spark_container="${7:-cuteafd-spark-expert-wip}"
wip_layout_slot="${8:-$wip_slot}"
wip_root_arg="${9:-__default__}"; wip_instance_arg="${10:-__none__}"
namespace_root="${11:-/mnt/sparknest}"
hf_home="${12:?missing HF_HOME}"
[[ "$wip_instance_arg" != __none__ ]] || wip_instance_arg=
[[ "$wip_root_arg" != __default__ ]] || wip_root_arg="$HOME/.cache/cuteafd/builds/wip${wip_instance_arg:+-$wip_instance_arg}"
wip_cp() { # slot files come from the container when it exists, else from the WIP root
  if docker container inspect "$wip_spark_container" >/dev/null 2>&1; then docker cp "$wip_spark_container:/wip/$1" "$2"
  else [[ -e "$wip_root_arg/${1%/.}" ]] && cp -a "$wip_root_arg/$1" "$2"; fi
}
# Every failure names the host and the check: this block runs over SSH, so a
# bare nonzero exit would otherwise surface as an unexplained transport error.
# Diagnostics go to stderr; stdout stays empty on success.
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

REMOTE
)" || release_die "Spark host preflight failed on $host (see the messages above)"

done

# A native Spark TP other than TP4 needs the matching SM121 expert role baked
# into the Spark image. The published universal release pair advertises every
# extra role at once (label `io.cuteafd.spark_tp_roles`, read with the pre-rename
# `io.cuteafd.v41.spark_tp_roles` as a fallback), so one published image pair
# serves every planned Spark count and this launch selects the role it needs.
# A missing role is refused here, before any service is stopped or replaced.
spark_advertised_roles=""
if [[ -n "$spark_tp_roles_required" ]]; then
  for i in "${spark_expert_ranks[@]}"; do
    host="${hosts[$i]}"
    if [[ -n "${wip_slot:-}" ]]; then
      # Slots overlay a toolchain image: its label cannot describe their exports.
      advertised_roles="$(release_ssh -o ConnectTimeout=10 "$host" \
        "if [ \"\$(docker container inspect -f '{{.State.Running}}' '$wip_spark_container' 2>/dev/null)\" = true ]; then docker exec -i '$wip_spark_container' python3 - '$wip_slot' '${expert_format:-native}' '$spark_tp_roles_required'; else WIP_SLOTS=\"\$(r=${WIP_ROOT:-}; i=${WIP_INSTANCE:-}; echo \"\${r:-\$HOME/.cache/cuteafd/builds/wip\${i:+-\$i}}\")/slots\" python3 - '$wip_slot' '${expert_format:-native}' '$spark_tp_roles_required'; fi" <<'PY'
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
    assert manifest.get('schema') == 1 and isinstance(roles, list) and all(isinstance(role, str) and role.startswith('tp') and role[2:].isdigit() and 1 <= int(role[2:]) <= 8 for role in roles), 'invalid TP roles'
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
      release_die "$host Spark image does not advertise required expert role $spark_tp_roles_required (advertised: ${advertised_roles:-<none>}); refusing an unbuilt TP$spark_tp world: use the published universal release pair, or rebuild with CUTEAFD_RELEASE_SPARK_TP_ROLES=$spark_tp_roles_required"
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

peers="$(jq -er '.peers | join(",")' <<<"$count_plan")"
fingerprint="$(printf '%s\n' "$engine_commit" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" "$ADDR" "$RELEASE_RTX_GPUS" "$gpu_uuid_csv" "$gpu_pci_csv" "$CONCURRENCY" "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" "$HTTP_QUEUE_WAIT_MS" "$HOST_CACHE_BYTES" "$RTX_EXPERT_LAYERS" "${KV_POOL_SIZE}${POOL_TOKENS:+:planner=$POOL_TOKENS}" "$EMBEDDING" "$VISION" "$AUDIO" "memory=${memory_policy_args[*]}" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" "prefill-waves=${V41_PREFILL_WAVES:-}" "prefill-reserve=${V41_PREFILL_RESERVE_ROWS:-}" "$SPECULATOR" "${DRAFT_SURVIVAL_LENGTHS:-off}"${DRAFT_POLICY:+ "draft-policy=$DRAFT_POLICY"}${DRAFT_ROW_COST:+ "draft-row-cost=$DRAFT_ROW_COST"} "${draft_length:-auto}" "$TP2_ATTENTION" "$TP2_QUERY_PROJECTION" "$TP2_OUTPUT_PROJECTION" "$TP2_DSPARK_EXPERTS" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "${v41_worker_budget:-}" "$SPARK_COUNT" "$(release_hosts_csv)" "$peers" "spark-tp=${spark_tp}" "v41-spark-tp-roles=${spark_tp_roles_required}" | sha256sum | awk '{print $1}')"
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
  echo "  Spark owners: coordinator-authoritative typed admission"
  echo "  Spark TP: $spark_tp"
  [[ -z "$spark_tp_roles_required" ]] ||
    echo "  Spark image must advertise V41 expert role: $spark_tp_roles_required (selected rank advertises: ${spark_advertised_roles:-<unreadable>})"
  echo "  graph node padding: ${GRAPH_NODE_PADDING:-2.5%} (independent of GPU_RESERVE)"
  echo "  memory policy: ${memory_policy_args[*]:-defaults (full GPU cap, family reserves, 2M KV target)}"
  echo "  prefill batch tokens: $PREFILL_BATCH_TOKENS; expert capacity: $expert_capacity"
  echo "  dSpark draft: ${draft_length:+fixed }${draft_length:-adaptive (maximum 5)}"
  [[ -z "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" ]] || echo "  RDMA device map: $CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP"
  [[ -z "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" ]] || echo "  RDMA execution lanes: $CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES"
  echo "  release identity: $fingerprint"
  printf '%s\n' "$count_plan"
  exit 0
fi
shared_api_state "$COORDINATOR_DOCKER_INFERENCE" "${INSTANCE:-default}"
if ((restart)); then
  release_stop_services "$coordinator" "$spark_prefix"
  ((restart_all == 0)) || release_stop_all_worker_containers
else
  docker inspect "$coordinator" >/dev/null 2>&1 && release_die "$coordinator already exists; use --restart"
  for i in "${spark_worker_ranks[@]}"; do
    remote="${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}"
    release_ssh "${hosts[$i]}" "! docker inspect '$remote' >/dev/null 2>&1" || release_die "$remote already exists; use --restart"
  done
fi

ss -ltn "sport = :${ADDR##*:}" 2>/dev/null | tail -n +2 | grep -q . &&
  release_die "API port ${ADDR##*:} is already in use"
for i in "${spark_worker_ranks[@]}"; do
  worker_port="$EXPERT_PORT"
  release_spark_role_has_experts "$count_plan" "$i" || worker_port=$((EXPERT_PORT + 100))
  release_ssh "${hosts[$i]}" \
    "! ss -ltn 'sport = :$worker_port' 2>/dev/null | tail -n +2 | grep -q ." ||
    release_die "${hosts[$i]}:$worker_port is already in use; stop the development worker first"
done

cleanup() {
  docker rm -f "$coordinator" >/dev/null 2>&1 || true
  for i in "${spark_worker_ranks[@]}"; do release_ssh "${hosts[$i]}" "docker rm -f '${spark_prefix}-${hosts[$i]}-${EXPERT_PORT}' >/dev/null 2>&1 || true" || true; done
}
trap cleanup EXIT

# Workers bootstrap first; the coordinator selects exact owners over TCP.
# Optional RDMA tuning values travel to both roles only when the operator sets
# them, so a multi-homed six-rank launch can pin the rail without changing any
# default. Values were format-checked above by release_validate_verbs_device_map.
rdma_env_args=("${bond_args[@]}")
# Optional V4.1 coordinator switches from the operator's environment, forwarded only when set.
# (CONSOLE_TEXT and THERMAL are configuration keys, as for every family.)
for switch_name in CUTEAFD_STAGE_CHAIN CUTEAFD_WINDOW_BATCH CUTEAFD_V41_FP8_HEAD; do
  # The V41_FP8_HEAD key (validated above) takes the place of its environment spelling.
  [[ "$switch_name" != CUTEAFD_V41_FP8_HEAD || ${#v41_fp8_head_args[@]} == 0 ]] || continue
  [[ -z "${!switch_name:-}" ]] || rdma_env_args+=(-e "$switch_name=${!switch_name}")
done
rdma_env_args+=(${v41_fp8_head_args[@]+"${v41_fp8_head_args[@]}"} ${v41_module_loading_args[@]+"${v41_module_loading_args[@]}"})
# RDMA_BOND_BALANCE (off, labels or probe): the coordinator's RoCE flow labels on a
# bonded port; see the generic launch below.
case "${RDMA_BOND_BALANCE:-off}" in
  off) ;;
  labels|probe) rdma_env_args+=(-e "CUTEAFD_RDMA_BOND_BALANCE=$RDMA_BOND_BALANCE") ;;
  *) release_die "RDMA_BOND_BALANCE must be off, labels or probe" ;;
esac
for rdma_env_name in CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES; do
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
      encoder_peers=()
      for vision_rank in "${spark_vision_ranks[@]}"; do encoder_peers+=("${lanes[$vision_rank]}:$encoder_port"); done
      encoder_peer="$(IFS=,; echo "${encoder_peers[*]}")"
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
local -a args=(--vision "$VISION" --audio "$AUDIO" serve-native "${vision_args[@]}" --snapshot "$(readlink -f "$hf_home/$snapshot_rel")" --native-lib /opt/cuteafd/lib/libcuteafd_native.so --peers "$peers" --rtx-gpus "$RELEASE_RTX_GPUS" --embedding-placement "$EMBEDDING" --listen "$ADDR" --prefill-batch-tokens "$PREFILL_BATCH_TOKENS" --concurrency "$CONCURRENCY" --prefix-cache-entries "$PREFIX_CACHE_ENTRIES" --max-context-tokens "$MAX_CONTEXT_TOKENS" --max-output-tokens "$MAX_OUTPUT_TOKENS" --mapped-scale-preload-gib "$(get MAPPED_SCALE_PRELOAD_GIB 0)")
case "$(get EXPERT_USAGE on)" in
  on) args+=(--expert-usage) ;;
  off) args+=(--expert-usage=false) ;;
  *) release_die "EXPERT_USAGE must be on or off" ;;
esac
release_kv_reserve_args v41_kv_args
args+=("${v41_kv_args[@]}")
# Loop abort (boot-time only): unset keeps the daemon's shipped defaults.
for pair in 'LOOP_ABORT_RUN:--loop-abort-run' 'LOOP_ABORT_COPIES:--loop-abort-copies' \
  'LOOP_FAST_SPAN:--loop-fast-span' 'LOOP_FAST_COPIES:--loop-fast-copies'; do
  key_name="${pair%%:*}"; flag="${pair#*:}"
  [[ -z "${!key_name:-}" ]] || args+=("$flag" "${!key_name}")
done
[[ -z "${PREFILL_PACKING:-}" ]] || args+=(--prefill-packing "$PREFILL_PACKING")
[[ -z "$v41_decode_lanes" ]] || args+=(--decode-lanes "$resolved_decode_lanes")
[[ -z "${V41_PREFILL_WAVES:-}" ]] || args+=(--prefill-waves "$V41_PREFILL_WAVES")
# DECODE_SHARE: as for every family (the engine's default 0.2; 0 prefills whole prompts).
[[ -z "$(get DECODE_SHARE)" ]] || args+=(--decode-share "$(get DECODE_SHARE)")
[[ -z "${V41_PREFILL_RESERVE_ROWS:-}" ]] || args+=(--prefill-reserve-rows "$V41_PREFILL_RESERVE_ROWS")
args+=("${memory_policy_args[@]}" --coordinator-weight-budget-gib "$v41_weight_budget_gib")
[[ -z "${SERVED_MODEL_ID:-}" ]] || args+=(--model-id "$SERVED_MODEL_ID")
args+=("${api_args[@]}")
args+=(--http-queue-depth "${HTTP_QUEUE_DEPTH:-$CONCURRENCY}" --http-queue-wait-ms "$HTTP_QUEUE_WAIT_MS")
[[ "$RTX_EXPERT_LAYERS" == auto ]] || args+=(--rtx-expert-layers "$RTX_EXPERT_LAYERS")
[[ "$HOST_CACHE_BYTES" == 0 ]] || args+=(--host-cache-bytes "$HOST_CACHE_BYTES")
[[ -z "$KV_POOL_SIZE" ]] || args+=(--kv-pool-size "$KV_POOL_SIZE")
if [[ -n "${POOL_TOKENS:-}" ]]; then
  planner_tokens="$POOL_TOKENS"; [[ "$planner_tokens" != auto ]] || planner_tokens=0
  args+=(--pool-tokens "$planner_tokens")
fi
[[ "$SPECULATOR" != dspark ]] || args+=(--dspark)
[[ -z "$draft_length" ]] || args+=(--draft-length "$draft_length")
[[ "$TP2_ATTENTION" != on ]] || args+=(--tp2-attention)
[[ "$TP2_QUERY_PROJECTION" != on ]] || args+=(--tp2-query-projection)
[[ "$TP2_OUTPUT_PROJECTION" != on ]] || args+=(--tp2-output-projection)
[[ "$TP2_DSPARK_EXPERTS" != on ]] || args+=(--tp2-dspark-experts)
[[ -z "$v41_l2_prefetch" ]] || args+=(--l2-prefetch "$v41_l2_prefetch")
[[ "$v41_engram_warm" != on ]] || args+=(--engram-warm)
release_serving_docker_args
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$coordinator" --restart no "${RELEASE_COORDINATOR_DEVICE_ARGS[@]}" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  -e "CUTEAFD_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=${RUST_LOG:-info}" \
  -e "CUTEAFD_V41_IMAGE_ADMISSIONS=${CUTEAFD_V41_IMAGE_ADMISSIONS:-2}" \
  "${rdma_env_args[@]}" "${bench_nonce_env_args[@]}" "${trace_args[@]}" "${graph_census_args[@]}" ${profile_args[@]+"${profile_args[@]}"} \
  "${shared_env_args[@]}" "${thermal_args[@]}" "${probe_args[@]}" \
  "${wip_mount_args[@]}" \
  "${api_mount_args[@]}" -e "CUTEAFD_IMAGE=$COORDINATOR_DOCKER_INFERENCE" -v "$bench_dir:/run/cuteafd/bench" -v "$fidelity_cache:/run/cuteafd/fidelity" -e CUTEAFD_FIDELITY_CACHE=/run/cuteafd/fidelity \
  ${profile_wrapper_args[@]+"${profile_wrapper_args[@]}"} \
  "${namespace_mount_args[@]}" "$COORDINATOR_DOCKER_INFERENCE" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd "${args[@]}" >/dev/null
}
deadline=$((SECONDS + ${CUTEAFD_RELEASE_READY_TIMEOUT_SECONDS:-900}))
echo "== starting native Spark experts =="
pids=()
for i in "${spark_worker_ranks[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  # ssh joins its arguments into one remote command line, which drops empty
  # arguments and shifts every later position; quote each one explicitly.
  worker_budget="$(release_spark_role_budget "$count_plan" "$i")" || exit 1
  remote_args=("$SPARK_EXPERT_DOCKER_INFERENCE" "$remote" "$i" "$worker_budget" "$EXPERT_PORT" "$snapshot_rel" "$fingerprint" "$spark_expert_world" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}" "${CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-}" "${RUST_LOG:-info}" "${WIP_LAYOUT_SLOT:-${wip_slot:-__none__}}")
  worker_encoder_rank=-1
  for vision_rank in "${spark_vision_ranks[@]}"; do
    [[ "$vision_rank" != "$i" ]] || worker_encoder_rank="$i"
  done
  remote_args+=("$worker_encoder_rank" "$encoder_port" "$encoder_hash" "$RELEASE_MODEL_REVISION")
  if release_spark_role_has_experts "$count_plan" "$i"; then remote_args+=(1); else remote_args+=(0); fi
  # Docker options for this worker follow a sentinel, after every positional argument:
  # PROFILE switches (absent unless set), the shared worker environment and a
  # PROFILE_SPARK rank's stop options; its profiler argv follows --profile-wrapper.
  release_profile_docker_args "${PROFILE:-}" "${PROFILE_DIR:-}" "$host" "${PROFILE_SPARK:-}"
  remote_args+=(--profile-args ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"} "${shared_worker_env_args[@]}")
  release_profile_spark_wrapper "${PROFILE_SPARK:-}" "${PROFILE_SPARK_RANKS:-}" "${PROFILE_DIR:-}" "$host" "$i" "$SPARK_COUNT" \
    "${PROFILE_SPARK_NCU_KERNEL:-}" "${PROFILE_SPARK_NCU_COUNT:-}" "${PROFILE_SPARK_NCU_SET:-}" \
    "${PROFILE_SPARK_NCU_SKIP:-}" "${PROFILE_SPARK_NSYS_GPU_METRICS:-}"
  if ((${#RELEASE_PROFILE_WRAPPER[@]})); then
    remote_args+=("${RELEASE_PROFILE_WRAPPER_DOCKER_ARGS[@]}" --profile-wrapper "${RELEASE_PROFILE_WRAPPER[@]}")
  fi
  # SPARK_MEMORY_CAP / SPARK_RESERVE: the worker's global memory flags, always
  # the last segment, behind their own sentinel.
  [[ -z "${spark_memory_args[*]-}" ]] || remote_args+=(--memory-args "${spark_memory_args[@]}")
  release_ssh "$host" "$(declare -f release_die release_hub_mounts release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args); export -f release_die release_hub_mounts release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args; HF_HOME=$(printf %q "$hf_home") CUTEAFD_SITE_HUB=$site_hub CUTEAFD_NAMESPACE_ROOT=$(printf %q "$namespace_root") bash -s -- $(printf '%q ' "${remote_args[@]}")" <<'REMOTE' &
set -euo pipefail
namespace_root="${CUTEAFD_NAMESPACE_ROOT:-/mnt/sparknest}"
image="$1"; name="$2"; rank="$3"; budget="$4"; port="$5"; snapshot_rel="$6"; fingerprint="$7"; world="$8"
# Optional RDMA tuning, forwarded only when the operator set it.
rdma_env="${9:-}"; execution_lanes="${10:-}"
# The operator's RUST_LOG reaches the worker too (a quoted heredoc does not
# see the caller's environment).
rust_log="${11:-info}"
wip_slot="${12:-__none__}"
wip_layout_slot="$wip_slot"
encoder_args=()
if [[ "$rank" == "${13:--1}" ]]; then
  encoder_flag=--encoder
  [[ "${17}" == 1 ]] || encoder_flag=--encoder-only
  encoder_args=("$encoder_flag" --encoder-listen "0.0.0.0:${14}" --encoder-plan-hash "${15}" --encoder-revision "${16}" --encoder-max-tokens 1024)
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
memory_args=()
for ((memory_at = 1; memory_at <= $#; memory_at++)); do
  [[ "${!memory_at}" != --memory-args ]] || { memory_args=("${@:memory_at+1}"); set -- "${@:1:memory_at-1}"; break; }
done
profile_args=() profile_wrapper=()
for ((profile_at = 1; profile_at <= $#; profile_at++)); do
  [[ "${!profile_at}" != --profile-args ]] || { profile_args=("${@:profile_at+1}"); break; }
done
for ((profile_at = 0; profile_at < ${#profile_args[@]}; profile_at++)); do
  if [[ "${profile_args[profile_at]}" == --profile-wrapper ]]; then
    profile_wrapper=("${profile_args[@]:profile_at+1}")
    profile_args=("${profile_args[@]:0:profile_at}")
    break
  fi
done
for ((profile_at = 0; profile_at + 1 < ${#profile_args[@]}; profile_at++)); do
  [[ "${profile_args[profile_at]}" != -v ]] || { mkdir -p "${profile_args[profile_at+1]%%:*}"; release_require_writable_state "${profile_args[profile_at+1]%%:*}"; }
done
rdma_args=()
[[ -z "$rdma_env" ]] || rdma_args+=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$rdma_env")
[[ -z "$execution_lanes" ]] || rdma_args+=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES=$execution_lanes")
hf_home="${HF_HOME:?missing HF_HOME}"
release_hub_mounts "$namespace_root" "$CUTEAFD_SITE_HUB" "$hf_home" "$snapshot_rel"
worker_snapshot="$RELEASE_HUB_SNAPSHOT"
worker_mounts=("${RELEASE_HUB_MOUNTS[@]}")
# A fixed INFO default makes the worker's structured startup evidence
# (`rank`/`world`/`role`/`intermediate`) observable; without it EnvFilter is
# ERROR and the readiness line never reaches the container log. This adds no
# positional argument, so the worker argument contract is unchanged.
release_serving_docker_args
cuteafd_cdi_gpu_args all || exit 2
expert_args=(--rank "$rank" --device-budget-bytes "$budget")
[[ "${17}" != 1 ]] || expert_args=(--rank "$rank" --world "$world" --device-budget-bytes "$budget" --listen "0.0.0.0:$port")
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$name" --restart no "${CUTEAFD_CDI_ARGS[@]}" --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e "CUTEAFD_RELEASE_CONFIG_SHA256=$fingerprint" -e "RUST_LOG=$rust_log" "${rdma_args[@]}" "${wip_args[@]}" ${profile_args[@]+"${profile_args[@]}"} "${worker_mounts[@]}" "$image" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd ${memory_args[@]+"${memory_args[@]}"} expertd-native --snapshot "$worker_snapshot" --native-lib /opt/cuteafd/lib/libcuteafd_native.so "${expert_args[@]}" "${encoder_args[@]}" >/dev/null
REMOTE
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in "${spark_worker_ranks[@]}"; do
  host="${hosts[$i]}"; remote="${spark_prefix}-${host}-${EXPERT_PORT}"
  ready_marker="$(release_spark_role_ready_marker "$count_plan" "$i")"
  # A connect-and-close probe is an invalid selection frame and kills the worker.
  until release_ssh "$host" "test \"\$(docker inspect -f '{{.State.Status}}' '$remote' 2>/dev/null)\" = running && docker logs '$remote' 2>&1 | grep -q '$ready_marker'" 2>/dev/null; do
    ((SECONDS < deadline)) || { release_ssh "$host" "docker logs --tail 100 '$remote'" >&2 || true; release_die "$host native expert did not become ready"; }
    sleep 1
  done
done

start_coordinator
api_url="$(release_probe_url "$ADDR")"
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
echo "  context/output: $MAX_CONTEXT_TOKENS/$MAX_OUTPUT_TOKENS; dSpark: $SPECULATOR"
}

# The family: run.sh passes --family; a direct call reads the checkpoint's own config.json.
core_view="$(get CORE_VIEW original)"
case "$core_view" in
  original|fp8-compact) selected_config=config.json ;;
  k8-compact) selected_config=config.k8.json ;;
  k6-compact) selected_config=config.k6.json ;;
  *) release_die "CORE_VIEW must be original, k8-compact, k6-compact or fp8-compact" ;;
esac
detected_family="$(release_config_family "$config" "$core_view")"
[[ -n "$family" ]] || family="$detected_family"
core_view_args=(--core-view "$core_view")
serve_core_view_args=()
case "$family" in
  glm5|glm5_flash|kimi_k3|step5|muse_spark) serve_core_view_args=("${core_view_args[@]}") ;;
  *) [[ "$core_view" == original ]] || release_die "CORE_VIEW=$core_view is unsupported for $family" ;;
esac

[[ "$family" == deepseek_v41 || "$family" == deepseek_v4 || "$(get TP2_DSPARK_EXPERTS off)" != on ]] ||
  release_die "TP2_DSPARK_EXPERTS requires V4/V4.1, two selected RTX GPUs and dSpark"
if [[ "$family" == deepseek_v41 ]]; then
  [[ "$detected_family" == deepseek_v41 ]] ||
    release_die "--family deepseek_v41 does not match the checkpoint ($detected_family)"
  attention_placement="$(get ATTENTION_PLACEMENT auto)"
  launch_deepseek_v41
  exit 0
fi
# Validate the name before it is used to identify allocations during admission.
instance="$(get INSTANCE)"
[[ -z "$instance" || "$instance" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] || { echo "INSTANCE must be [A-Za-z0-9_.-]" >&2; exit 2; }
coordinator_name="cuteafd-coordinator${instance:+-$instance}"
model="$(get MODEL_ID)"
revision="$(get MODEL_REVISION)"
hf_home="${HF_HOME:-$namespace_root/hf-home}"
hub="$(readlink -f "$hf_home/hub")"
root="$hub/models--${model//\//--}"
if [[ -z "$revision" ]]; then
  [[ -r "$root/refs/main" ]] || { echo "missing snapshot $model (no refs/main under $root)" >&2; exit 1; }
  revision="$(<"$root/refs/main")"
fi
snapshot="$(readlink -f "$root/snapshots/$revision")"
[[ -d "$root/snapshots/$revision" ]] || { echo "missing snapshot $model@$revision" >&2; exit 1; }
snapshot_rel="hub/models--${model//\//--}/snapshots/$revision"
release_hub_mounts "$namespace_root" "$site_hub" "$hf_home" "$snapshot_rel"
generic_worker_mounts=("${RELEASE_HUB_MOUNTS[@]}")
# Family, the serving command and the layers with routed experts.
described="$(python3 "$repo_root/scripts/lib/checkpoint-family.py" "$root/snapshots/$revision/$selected_config")" || exit 2
read -r detected model_type first_layer last_layer <<<"$described"
[[ -z "$family" || "$family" == "$detected" ]] ||
  { echo "--family $family does not match the checkpoint ($detected, model_type $model_type)" >&2; exit 2; }
family="$detected"
# The checkpoint's own family names its deployment plan; Mistral Large 4 runs
# the GLM 5.3 engine (serve-glm, its keys and lanes) as the engine's ml4 variant. Two GPUs:
# auto resolves to the ml42 page-partitioned context placement (one rank on one GPU runs the
# head split); heads and layers are explicit requests, and a missing context program falls
# back to whole-layer owners. The bundled mtp_dspark
# defaults on at its checkpoint-native depth; SPECULATOR=off disables it.
checkpoint_family="$family"
if [[ "$family" == mistral_large4 ]]; then
  family=glm5
  [[ -n "${cfg[SPECULATOR]+set}" ]] || cfg[SPECULATOR]=dspark
  case "$(get SPECULATOR dspark)" in
    off|dspark) ;;
    *) release_die "Mistral Large 4 drafts only with its bundled mtp_dspark (SPECULATOR=dspark or off)" ;;
  esac
  case "$(get ATTENTION_PLACEMENT auto)" in
    auto|layers|heads|context) ;;
    *) release_die "Mistral Large 4 attention placement is auto, layers, heads or context" ;;
  esac
fi
if [[ -n "$(get RTX_EXPERT_LAYERS)" ]]; then
  cfg[RTX_EXPERT_LAYERS]="$(release_onboard "$(get RTX_EXPERT_LAYERS)" "$model" "$revision")" || exit 2
fi
attention_placement="$(get ATTENTION_PLACEMENT auto)"
# Per-family acceptance lives in scripts/lib/attention-placement.sh, which the
# options page test drives for every family x value.
release_check_attention_placement "$attention_placement" "$family"
case "$family" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm5) serve=serve-glm ;;
  glm5_flash) serve=serve-glmf ;;
  mimo_v2) serve=serve-mimo ;;
  qwen4) serve=serve-qwen4 ;;
  qwen3_5) serve=serve-qwen35 ;;
  kimi_k3) serve=serve-kimi ;;
  step5) serve=serve-step5 ;;
  muse_spark) serve=serve-muse ;;
  *) echo "run-family.sh serves DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2, Qwen 3.8, Kimi K3, Step-5 and Muse Spark checkpoints, not $family (./run.sh serves DeepSeek V4.1)" >&2; exit 2 ;;
esac
# Kimi K3 and Step-5 serve through their own commands: automatic attention
# placement, text only (their bundled vision towers are not in scope) and GPU
# token embeddings. Kimi ships no MTP head; Step-5 defaults to native MTP
# depth 3 (SPECULATOR=off disables it).
text_family=0
[[ "$family" != kimi_k3 && "$family" != step5 ]] || text_family=1
# These commands own automatic context, pool and Spark budgets independently of media.
standalone_family=$text_family
[[ "$family" != muse_spark ]] || standalone_family=1
if ((text_family)); then
  case "$vision:$audio" in
    auto:auto|auto:off|off:auto|off:off) vision=off; audio=off ;;
    *) release_die "$family bundled media execution is not qualified; set VISION=off and AUDIO=off" ;;
  esac
fi
if [[ "$family" == qwen3_5 ]]; then
  case "$vision:$audio" in
    auto:auto|auto:off|off:auto|off:off) vision=off; audio=off ;;
    *) release_die "dense Qwen bundled media execution is not qualified" ;;
  esac
fi
# VISION and AUDIO default to auto for every family, as the release configs
# set them. Qualified MiMo, GLM Flash and Qwen towers are placed Spark-first by
# the encoder plan below; DeepSeek V4 and GLM 5.3 have no tower, so auto starts
# none and serves text only.
# Declared bundled audio is validated strictly by the encoder plan, not disabled here.
release_validate_verbs_device_map
audio="$(release_resolve_audio_mode "$audio" "$root/snapshots/$revision")"
# Auto/spark placement is resolved by the encoder plan below.
# An explicit count or host list bounds the available inventory.
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
  host_rows="$(
    # The shared resolver reads shell variables; generic launch config lives in cfg.
    for rank in {0..7}; do
      for field in HOST LANE_A LANE_B; do
        key="SPARK_${rank}_${field}"
        printf -v "$key" '%s' "$(get "$key")"
      done
    done
    release_spark_host_rows "$(get SPARK_HOSTS)" "$ranks"
  )" || exit 2
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
if [[ "$family" == qwen3_5 ]]; then
  [[ "$backend" != spark && ( "$count_explicit" == 0 || "$ranks" == 0 ) ]] ||
    release_die "dense Qwen has no Spark layout or dense-FFN offboarding"
  ranks=0
  backend=local
fi
if [[ "$family" == step5 ]]; then
  step5_kv_cache="$(get KV_CACHE fp8)"
  case "$step5_kv_cache" in bf16|fp8) ;; *) release_die "KV_CACHE must be bf16 or fp8" ;; esac
fi
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
# --wip SLOT serves a ./wip.sh slot: the development images run its artifacts, staged from
# the WIP containers into a release-shaped /opt/cuteafd layout per host (as ./run.sh --wip
# does for DeepSeek V4.1).
wip_layout="" wip_mount_args=() wip_worker_args=""
if [[ -n "$wip_slot" ]]; then
  [[ "$wip_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo "invalid WIP slot name: $wip_slot" >&2; exit 2; }
  WIP_INSTANCE="${WIP_INSTANCE:-$(get WIP_INSTANCE)}"
  # The WIP root the slot was staged under: the environment wins, else the
  # config (as V4.1's release_load_config reads it), else the default layout.
  WIP_ROOT="${WIP_ROOT:-$(get WIP_ROOT)}"
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
# Families with their own serve command: refuse a WIP slot whose binary
# predates it before planning or touching any service.
if ((standalone_family)) && [[ -n "$wip_slot" ]]; then
  docker run --rm --network none --mount "type=bind,src=$wip_layout/bin/cuteafd,dst=/opt/cuteafd-serve-probe,readonly" \
    --entrypoint /opt/cuteafd-serve-probe "$coordinator_image" "$serve" --help >/dev/null ||
    release_die "WIP slot $wip_slot has no '$serve' command; build the slot from a commit that serves $family"
fi

qwen_exl3=0
qwen_nvfp4=0
qwen_mtp=0
if [[ "$family" == qwen4 ]]; then
  kv_cache="$(get KV_CACHE fp8)"
  case "$kv_cache" in bf16|fp8) ;; *) release_die "KV_CACHE must be bf16 or fp8" ;; esac
  # NVFP4: a ModelOpt publication with 4-bit weight groups (nvidia/Qwen3.8-Flash-Next-NVFP4).
  qwen_features="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("quantization_config", {}); m=q.get("quant_method", q.get("method")); g=q.get("config_groups", {}).values(); print(int(m == "exl3"), int(m == "modelopt" and any(x.get("weights", {}).get("num_bits") == 4 for x in g)), c.get("text_config", c).get("mtp_num_hidden_layers", 0))' "$root/snapshots/$revision/$selected_config")"
  read -r qwen_exl3 qwen_nvfp4 qwen_mtp <<<"$qwen_features"
fi
# Docker options only the Spark expert workers take (their environment and mounts).
spark_worker_env=""
if [[ ( "$family" == mimo_v2 || "$family" == glm5 || "$family" == glm5_flash || "$family" == qwen4 || "$family" == deepseek_v41 ) && -n "$(get SPARK_GRAPH_RESERVE)" ]]; then
  spark_worker_env+=" -e $(printf '%q' "CUTEAFD_SPARK_GRAPH_RESERVE=$(get SPARK_GRAPH_RESERVE)")"
fi
# Options only the Spark expert workers take.
spark_worker_args=""
# Snapshot of a model id (and optional revision) inside the containers.
snapshot_of() {
  local id="$1" rev="$2" dir="$hub/models--${1//\//--}"
  if [[ -z "$rev" ]]; then
    [[ -r "$dir/refs/main" ]] || { echo "missing snapshot $id (no refs/main under $dir)" >&2; return 1; }
    rev="$(<"$dir/refs/main")"
  fi
  [[ -d "$dir/snapshots/$rev" ]] || { echo "missing snapshot $id@$rev" >&2; return 1; }
  readlink -f "$dir/snapshots/$rev"
}
spark_worker_args+=" ${serve_core_view_args[*]}"
exlr8_manifest="$(get EXLR8_MANIFEST)"
exlr8_prefill_activations="$(get EXLR8_PREFILL_ACTIVATIONS)"
if [[ -n "$exlr8_prefill_activations" ]]; then
  [[ "$exlr8_prefill_activations" == a16 || "$exlr8_prefill_activations" == a8 ]] || release_die "EXLR8_PREFILL_ACTIVATIONS must be a16 or a8"
  [[ -n "$exlr8_manifest" ]] || release_die "EXLR8_PREFILL_ACTIVATIONS requires EXLR8_MANIFEST"
fi
exlr8_plan_args=()
if [[ -n "$exlr8_manifest" ]]; then
  [[ "$exlr8_manifest" != *[!a-zA-Z0-9_-]* ]] || release_die "EXLR8_MANIFEST must be a published name, not a path"
  exlr8_gpu_count="$(get RTX_GPUS 1)"
  [[ ( "$exlr8_gpu_count" == 1 || "$exlr8_gpu_count" == 2 ) &&
     "$ranks" -ge 0 && "$ranks" -le 8 ]] || release_die "EXLR8 requires 1..2 RTX GPUs and 0..8 Spark ranks"
  [[ "$(get SPECULATOR dflash2)" == off || "$(get SPECULATOR dflash2)" == dflash2 ]] || release_die "EXLR8 supports SPECULATOR=off or dflash2"
  if [[ "$family" != glm5_flash ]]; then
    [[ "$attention_placement" == auto || "$attention_placement" == heads ||
       ( "$exlr8_gpu_count" == 2 && ( "$attention_placement" == context || "$attention_placement" == layers ) ) ]] ||
      release_die "GLM EXLR8 requires heads or two-GPU context/layers attention"
  fi
  exlr8_plan_args=(--exlr8-manifest "$exlr8_manifest")
  spark_worker_args+=" --exlr8-manifest $exlr8_manifest"
  if [[ -n "$exlr8_prefill_activations" ]]; then
    exlr8_plan_args+=(--exlr8-prefill-activations "$exlr8_prefill_activations")
    spark_worker_args+=" --exlr8-prefill-activations $exlr8_prefill_activations"
  fi
fi
# SPECULATOR picks the drafter. Unset, each family takes the speculator its
# release card ran (examples/configs/*.config; scripts/tests/test_example_configs.py
# keeps the two in step): GLM 5.3 DFlash2; GLM 5.3 Flash its measured best;
# MiMo V2.6 Flash/Pro MOPD the bundled DFlash; DeepSeek V4 dSpark when the
# checkpoint carries it; Qwen local EXL3/NVFP4 MTP3; otherwise off.
#   dflash2  GLM 5.x / GLM 5.3 Flash: the DFlash2 checkpoint SPECULATOR_MODEL_ID
#            (e.g. incoai/GLM-5.3-DFlash2, incoai/GLM-5.3-Flash-DFlash2);
#            MiMo V2.6 Flash/Pro: the snapshot's own dflash/ drafter unless
#            SPECULATOR_MODEL_ID names one (Pro needs SPARK_COUNT=6)
#   mtp      MiMo V2 Flash, Qwen 3.8, Step-5 (opt-in): the checkpoint's native MTP layers,
#            SPECULATOR_DEPTH drafts (qualified local Qwen default 3; otherwise 1)
#   dspark   DeepSeek V4 (its own drafter); GLM 5.3 Flash: the dSpark
#            checkpoint SPECULATOR_MODEL_ID (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
# Official Flash MOPD's bundled drafter defaults to single-copy FP8 (measured
# separately from its checkpoint BF16 target head/O); other MiMo defaults follow
# runtime weight policy. SPECULATOR_FP8=auto keeps drafter checkpoint format. GLM auto/unset
# drafts in single-copy FP8. on converts, off selects BF16.
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
  dsv4_dspark="$(python3 -c 'import json,sys; print(int(int(json.load(open(sys.argv[1])).get("dspark_block_size") or 0) > 0))' "$root/snapshots/$revision/$selected_config")"
fi
speculator="$(get SPECULATOR)"
# Qwen resolves auto only after expert admission, not from configured inventory.
if [[ "$family" == qwen4 && ( -z "$speculator" || "$speculator" == auto ) ]]; then speculator=auto; fi
default_drafter=""
[[ "$family" != muse_spark ]] || default_drafter=nvidia/Muse-Spark-Open-816B-A42B-DFlash2
if [[ -z "$speculator" ]]; then
  speculator=off
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
  if [[ $speculator == off && $mimo_bundled_dflash == 1 ]]; then
    speculator=dflash2
    echo "note: $model drafts with its bundled DFlash; SPECULATOR=off disables it" >&2
  fi
  if [[ $speculator == off && $dsv4_dspark == 1 ]]; then
    speculator=dspark
    echo "note: DeepSeek V4 drafts with its bundled dSpark; SPECULATOR=off disables it" >&2
  fi
  if [[ $speculator == off && $family == qwen3_5 ]] &&
      python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(c.get("text_config",c).get("hidden_size") != 5120)' "$root/snapshots/$revision/$selected_config"; then
    speculator=dspark default_drafter=RadixArk/Qwen3.8-27B-DSpark
  fi
  if [[ $speculator == off && $family == muse_spark ]]; then
    speculator=dflash2
    echo "note: Muse Spark drafts with DFlash2 ($default_drafter); SPECULATOR=off disables it" >&2
  fi
  if [[ $speculator == off && $family == step5 ]]; then
    speculator=mtp
    echo "note: Step-5 drafts with native MTP (default depth 3); SPECULATOR=off disables it" >&2
  fi

fi
case "$family:$speculator" in
  qwen4:mtp|qwen4:auto) ;;
  glm5:dspark) [[ "$checkpoint_family" == mistral_large4 ]] ||
    { echo "SPECULATOR=dspark does not apply to $family" >&2; exit 2; } ;; # The MTP layer's experts stay local even with Spark backbone experts.
  *:off|qwen3_5:dspark|qwen3_5:mtp|glm5:dflash2|glm5_flash:dflash2|glm5_flash:dspark|mimo_v2:dflash2|mimo_v2:mtp|kimi_k3:dflash2|step5:mtp|muse_spark:dflash2|deepseek_v4:dspark) ;;
  *) echo "SPECULATOR=$speculator does not apply to $family" >&2; exit 2 ;;
esac
draft_args=()
embedding="$(get EMBEDDING gpu)"
# Coordinator GPUs (the shared resolution V4.1 uses too): COORDINATOR_GPUS names
# one or two, primary first; unset takes the lowest-index GPU, plus the next for a
# two-GPU layout. RTX_GPUS and COORDINATOR_SPLIT say how many a launch asks for;
# COORDINATOR_SPLIT=off serves from the first alone.
rtx_gpus="$(get RTX_GPUS auto)"
case "$rtx_gpus" in auto|1|2) ;; *) echo "RTX_GPUS must be auto, 1 or 2" >&2; exit 2 ;; esac
split="$(get COORDINATOR_SPLIT auto)"
case "$split" in auto|heads|off) ;; *) echo "COORDINATOR_SPLIT must be auto, heads or off" >&2; exit 2 ;; esac
explicit_coordinator_gpus="$(get COORDINATOR_GPUS)"
coordinator_gpu_count=auto
if [[ "$split" != off ]]; then
  if [[ "$rtx_gpus" == 2 || "$split" == heads ]]; then coordinator_gpu_count=2
  elif [[ "$rtx_gpus" == 1 ]]; then coordinator_gpu_count=1
  fi
fi
release_resolve_coordinator_gpus "$explicit_coordinator_gpus" "$coordinator_gpu_count"
profile_bytes="$(release_cuda_total_bytes "$RELEASE_COORDINATOR_GPU_UUID")" || release_die "cannot query coordinator CUDA capacity"
profile="$(release_coordinator_profile "$profile_bytes" "$coordinator_budget")" || release_die "cannot resolve coordinator capacity profile"
read -r profile_cap default_concurrency <<<"$profile"
[[ "$family" != qwen3_5 ]] || default_concurrency=4
effective_concurrency="$(get CONCURRENCY "$default_concurrency")"
[[ "$effective_concurrency" =~ ^[0-9]+$ ]] && ((effective_concurrency >= 1 && effective_concurrency <= 64)) || release_die "CONCURRENCY must be an integer in 1..=64"
effective_prefix_entries="$(get PREFIX_CACHE_ENTRIES auto)"
[[ "$effective_prefix_entries" != auto ]] || effective_prefix_entries=$((effective_concurrency <= 16 ? 20 : (5 * effective_concurrency + 3) / 4))
if [[ "$family" == mimo_v2 ]]; then
  if ((profile_cap <= 32 << 30)); then
    [[ -n "${cfg[EMBEDDING]:-}" ]] || embedding=host
    echo "MiMo 32 GB profile: logical GPU $(release_gib "$profile_cap") GiB, embedding=$embedding (EMBEDDING overrides), int8 KV, checkpoint-full context unless MAX_CONTEXT_TOKENS overrides" >&2
  fi
fi
case "$embedding" in host|gpu) ;; *) echo "EMBEDDING must be host or gpu" >&2; exit 2 ;; esac
[[ "$family" != qwen3_5 || "$embedding" == gpu ]] || release_die "dense Qwen requires device token embedding"
((standalone_family == 0)) || [[ "$embedding" == gpu ]] || release_die "$family serves GPU token embeddings only (EMBEDDING=gpu)"
decode_lanes="$(get DECODE_LANES)"
if [[ -n "$decode_lanes" ]]; then
  [[ "$decode_lanes" =~ ^[12]$ ]] || release_die "DECODE_LANES must be 1 or 2"
fi
family_args=(--embedding-placement "$embedding" "${exlr8_plan_args[@]}" "${serve_core_view_args[@]}")
case "$(get EXPERT_USAGE on)" in
  on|off) ;;
  *) release_die "EXPERT_USAGE must be on or off" ;;
esac
if [[ $family == deepseek_v4 || $family == mimo_v2 || $family == glm5 || $family == kimi_k3 ]]; then
  case "$(get EXPERT_USAGE on)" in
    on) family_args+=(--expert-usage) ;;
    off) family_args+=(--expert-usage=false) ;;
    *) release_die "EXPERT_USAGE must be on or off" ;;
  esac
fi
tp2_dflash_default=off
if [[ ( "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5_flash || "$family" == muse_spark ) && ${#RELEASE_COORDINATOR_GPU_INDICES[@]} == 2 && "$split" != off && "$speculator" == dflash2 ]]; then
  tp2_dflash_default=on
  [[ "$family" != glm5_flash ]] || tp2_dflash_default=auto
fi
tp2_dflash="$(get TP2_DFLASH "$tp2_dflash_default")"
case "$tp2_dflash" in on|off) ;; auto) [[ "$family" == glm5_flash && -z "$(get TP2_DFLASH)" ]] || release_die "TP2_DFLASH must be on or off" ;; *) release_die "TP2_DFLASH must be on or off" ;; esac
[[ "$tp2_dflash" != on || ( ( "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5_flash || "$family" == muse_spark ) && ${#RELEASE_COORDINATOR_GPU_INDICES[@]} == 2 && "$speculator" == dflash2 ) ]] ||
  release_die "TP2_DFLASH requires MiMo/Kimi/GLM Flash/Muse, two selected RTX GPUs and DFlash2"
if [[ ( "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5_flash || "$family" == muse_spark ) && "$tp2_dflash" != auto ]]; then
  # Forward false as well: an explicit off must survive future default changes.
  tp2_dflash_bool="$([[ "$tp2_dflash" == on ]] && echo true || echo false)"
  family_args+=(--tp2-dflash "$tp2_dflash_bool")
fi
# Loop abort (boot-time only): unset keeps the daemon's shipped defaults.
for pair in 'LOOP_ABORT_RUN:--loop-abort-run' 'LOOP_ABORT_COPIES:--loop-abort-copies' \
  'LOOP_FAST_SPAN:--loop-fast-span' 'LOOP_FAST_COPIES:--loop-fast-copies'; do
  key_name="${pair%%:*}"; flag="${pair#*:}"
  value="$(get "$key_name")"
  [[ -z "$value" ]] || family_args+=("$flag" "$value")
done
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
    if ! release_path_within "$chat_template_from" "$namespace_root"; then
      if [[ "$chat_template_from" == */snapshots/* ]]; then
        template_repo="${chat_template_from%/snapshots/*}"
        chat_template_mounts=(-v "$template_repo:$template_repo:ro")
      else
        chat_template_mounts=(-v "$chat_template_from:$chat_template_from:ro")
      fi
    fi
  else
    [[ "$chat_template_from" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] &&
      release_path_has_no_dot_segment "$chat_template_from" || release_die "CHAT_TEMPLATE_FROM must be an existing snapshot or ORG/MODEL HF id"
    snapshot_of "$chat_template_from" "" >/dev/null || exit 2
  fi
  family_args+=(--chat-template-from "$chat_template_from")
fi
# Kimi K3 and Step-5 serve text only; their commands take no media options.
((text_family)) || family_args+=(--vision "$vision" --audio "$audio")
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
mtp_depth=1
[[ "$family" != step5 ]] || mtp_depth=3
[[ ( $qwen_exl3 != 1 && $qwen_nvfp4 != 1 ) || $qwen_mtp != 1 ]] || mtp_depth=3
case "$speculator" in
  off)
    [[ "$family" != step5 ]] || family_args+=(--mtp 0) ;;
  dflash2|dspark)
    drafter="$(get SPECULATOR_MODEL_ID)"
    [[ -n "$drafter" ]] || drafter="$default_drafter"
    [[ "$family:$speculator" != qwen3_5:dspark || -n "$drafter" ]] || drafter=RadixArk/Qwen3.8-27B-DSpark
    if [[ $speculator == dspark && ( $family == deepseek_v4 || $checkpoint_family == mistral_large4 ) ]]; then
      dspark_args=(--dspark)
    elif [[ -n "$drafter" ]]; then
      draft_revision="$(get SPECULATOR_MODEL_REVISION)"
      if [[ -z "$draft_revision" && $family == muse_spark && $drafter == nvidia/Muse-Spark-Open-816B-A42B-DFlash2 ]]; then
        draft_revision=22e2669e4c60689f9bff0749a34d9d8e59ce7cd6
      fi
      if [[ -z "$draft_revision" && $family:$speculator == qwen3_5:dspark && $drafter == RadixArk/Qwen3.8-27B-DSpark ]]; then
        draft_revision=85ef153be924f17ce4bf62726954eeaa4a73e854
      fi
      draft_snapshot="$(snapshot_of "$drafter" "$draft_revision")" ||
        { [[ -z "$default_drafter" ]] || echo "download it (hf download $drafter) or set SPECULATOR=off" >&2; exit 1; }
      draft_args=(--draft "$draft_snapshot")
      [[ "$family" != kimi_k3 ]] || draft_args+=(--draft-fp8 true)
      if [[ "$family" == muse_spark ]]; then
        muse_draft_kv="$(get DRAFT_KV_CACHE fp8)"
        case "$muse_draft_kv" in bf16|fp8) ;; *) release_die "DRAFT_KV_CACHE must be bf16 or fp8" ;; esac
        draft_args+=(--draft-kv-cache "$muse_draft_kv")
      fi
    elif [[ $family == mimo_v2 ]]; then
      draft_args=(--draft "$snapshot")
    else
      echo "SPECULATOR=$speculator needs SPECULATOR_MODEL_ID (a ${speculator/dflash2/DFlash2} checkpoint)" >&2; exit 2
    fi ;;
  mtp)
    family_args+=(--mtp "$(get SPECULATOR_DEPTH "$mtp_depth")") ;;
esac
draft_model_args=${#draft_args[@]}
# One fixed-length override for every family and speculator.
draft_length="$(get DRAFT_LENGTH)"
if [[ -n "$draft_length" ]]; then
  [[ "$draft_length" =~ ^[1-9][0-9]*$ ]] || release_die "DRAFT_LENGTH must be a positive integer"
  draft_args+=(--draft-length "$draft_length")
fi
# Prefix cache (MiMo, GLM 5.3, GLM 5.3 Flash, Qwen 3.8, DeepSeek V4): PREFIX_CACHE_ENTRIES
# snapshots per bank (prompts, turns; 0 = off), HOST_CACHE_BYTES of pinned host memory for
# retention (auto by default, e.g. 64GiB explicitly; 0 = off). POOL_TOKENS: paged KV tokens shared by
# live sequences and retained snapshots.
# KV reservation by decode chunks (every family): KV_DECODE_RESERVE_CHUNK tokens per top-up,
# KV_RESERVE_DRAIN_BUFFER (0 = the whole remaining output), KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS.
kv_reserve_args=()
[[ -z "$(get KV_DECODE_RESERVE_CHUNK)" ]] || kv_reserve_args+=(--kv-decode-reserve-chunk "$(get KV_DECODE_RESERVE_CHUNK)")
[[ -z "$(get KV_RESERVE_DRAIN_BUFFER)" ]] || kv_reserve_args+=(--kv-reserve-drain-buffer "$(get KV_RESERVE_DRAIN_BUFFER)")
[[ -z "$(get KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS)" ]] ||
  kv_reserve_args+=(--kv-early-termination-min-output-tokens "$(get KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS)")
case $family in
  mimo_v2|glm5|glm5_flash|qwen4|qwen3_5|deepseek_v4|kimi_k3|step5|muse_spark)
    family_args+=(--prefix-cache-entries "$effective_prefix_entries")
    [[ "$(get HOST_CACHE_BYTES auto)" == 0 ]] || family_args+=(--host-cache-bytes "$(get HOST_CACHE_BYTES auto)") ;;
esac
if [[ ( $family == mimo_v2 || $family == qwen4 || $family == glm5_flash ) && $vision != off ]]; then
  # Text-only frozen daemons can predate the optional media-cache flag.
  [[ -z "$(get MEDIA_CACHE_BYTES)" ]] || family_args+=(--media-cache-bytes "$(get MEDIA_CACHE_BYTES)")
fi
if [[ $family == mimo_v2 ]]; then
  [[ -z "$(get HTTP_QUEUE_DEPTH)" ]] || family_args+=(--http-queue-depth "$(get HTTP_QUEUE_DEPTH)")
  [[ -z "$(get HTTP_QUEUE_WAIT_MS)" ]] || family_args+=(--http-queue-wait-ms "$(get HTTP_QUEUE_WAIT_MS)")
  [[ "$(get MIMO_PREFIX_DRAFT off)" != on ]] || family_args+=(--mimo-prefix-draft)
  [[ "$(get MIMO_SNAPSHOT_WAIT off)" != on ]] || family_args+=(--mimo-snapshot-wait)
  if [[ "$(get MIMO_HOST_CACHE off)" == on ]]; then
    family_args+=(--mimo-host-cache)
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
  # dims) or bf16. SWA_KV_CACHE selects the sliding and native-MTP rings separately. Unset omits --kv-cache so the daemon default (int8) applies.
  mimo_kv="$(get KV_CACHE)"
  case "$mimo_kv" in
    "") ;;
    int8|bf16) family_args+=(--kv-cache "$mimo_kv") ;;
    *) release_die "KV_CACHE must be int8 or bf16" ;;
  esac
  mimo_swa_kv="$(get SWA_KV_CACHE int8)"
  case "$mimo_swa_kv" in
    bf16|int8|fp8) family_args+=(--swa-kv-cache "$mimo_swa_kv") ;;
    *) release_die "SWA_KV_CACHE must be bf16, int8 or fp8" ;;
  esac
  # Keep serving and admission on the same explicit graph policy.
  mimo_decode_graphs="$(get DECODE_GRAPHS on)"
  case "$mimo_decode_graphs" in
    on) family_args+=(--decode-graphs true) ;;
    off) family_args+=(--decode-graphs false) ;;
    *) echo "DECODE_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
if [[ "$family" == qwen3_5 ]]; then
  dense_qwen_kv_cache="$(get KV_CACHE fp8)"
  case "$dense_qwen_kv_cache" in bf16|fp8) ;; *) release_die "KV_CACHE must be bf16 or fp8" ;; esac
  family_args+=(--kv-cache "$dense_qwen_kv_cache")
  [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || family_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
  # Qwen templates can remove old thinking; retain the exact frontier before
  # the generation message, not an approximate reconstruction of that state.
  family_args+=(--prefix-point-boundaries "$(get PREFIX_POINT_BOUNDARIES 1)")
fi
prefill_packing="$(get PREFILL_PACKING)"
if [[ -n "$prefill_packing" ]]; then
  [[ "$family" == mimo_v2 || "$family" == glm5_flash || "$family" == step5 || "$family" == qwen4 || "$family" == qwen3_5 || "$family" == kimi_k3 || "$family" == deepseek_v4 || "$family" == glm5 || "$family" == deepseek_v41 || "$family" == muse_spark ]] || { echo "PREFILL_PACKING is unsupported for $family" >&2; exit 2; }
  case "$prefill_packing" in
    unrestricted|exact) family_args+=(--prefill-packing "$prefill_packing") ;;
    *) echo "PREFILL_PACKING must be unrestricted or exact" >&2; exit 2 ;;
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
  family_args+=(--kv-cache "$kv_cache")
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
if [[ $family =~ ^(glm5|qwen4|qwen3_5|deepseek_v4)$ ]]; then
  pool="$(get POOL_TOKENS auto)"
  if [[ "$pool" == auto ]]; then
    pool=0
  fi
  family_args+=(--pool-tokens "$pool")
fi
# Kimi K3 and Step-5: unset or auto POOL_TOKENS leaves the KV pool to the serve
# command's own automatic admission (no --pool-tokens). Kimi's RTX_EXPERT_LAYERS
# is auto (default: the planner picks the RTX-onboarded routed layers and marks)
# or a fixed count, which admission must then fit.
if [[ "$family" == step5 ]]; then
  step5_decode_graphs="$(get DECODE_GRAPHS on)"
  case "$step5_decode_graphs" in
    on) family_args+=(--decode-graphs true) ;;
    off) family_args+=(--decode-graphs false) ;;
    *) release_die "DECODE_GRAPHS must be on or off" ;;
  esac
  family_args+=(--kv-cache "$step5_kv_cache")
  # Pool-first EXL3 onboarding uses whole layers on one RTX or TP2 halves on
  # two. FP8 supports TP2 onboarding; NVFP4 is deprecated. Select auto, all or N.
  local_layers="$(get RTX_EXPERT_LAYERS auto)"
  [[ "$local_layers" == auto || "$local_layers" == all || "$local_layers" =~ ^[0-9]+$ ]] ||
    release_die "Step-5 RTX_EXPERT_LAYERS is auto, all or a fixed routed-layer count, not $local_layers"
  [[ "$local_layers" == auto ]] || family_args+=(--rtx-expert-layers "$local_layers")
fi
if ((standalone_family)); then
  pool="$(get POOL_TOKENS auto)"
  [[ "$pool" == auto ]] || family_args+=(--pool-tokens "$pool")
fi
# Muse Spark: RTX expert onboarding (normalized above; an image without its
# coordinator EXL3 package refuses rather than serving them from the Sparks),
# the partial-reply cutoff and wide format (the prefill unit shape comes from the
# deployment plan, as for every units family).
if [[ $family == muse_spark ]]; then
  muse_decode_graphs="$(get DECODE_GRAPHS on)"
  case "$muse_decode_graphs" in
    on) family_args+=(--decode-graphs true) ;;
    off) family_args+=(--decode-graphs false) ;;
    *) echo "DECODE_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
  local_layers="$(get RTX_EXPERT_LAYERS auto)"
  family_args+=(--rtx-expert-layers "$local_layers")
  [[ -z "$(get EXPERT_PARTIAL_BF16_MAX_ROWS)" ]] || family_args+=(--expert-partial-bf16-max-rows "$(get EXPERT_PARTIAL_BF16_MAX_ROWS)")
fi
if [[ $family == kimi_k3 ]]; then
  kv_cache="$(get KV_CACHE fp8)"
  case "$kv_cache" in bf16|fp8) ;; *) release_die "KV_CACHE must be bf16 or fp8" ;; esac
  family_args+=(--kv-cache "$kv_cache")
  local_layers="$(get RTX_EXPERT_LAYERS auto)"
  [[ "$local_layers" == auto || "$local_layers" == all || "$local_layers" =~ ^[0-9]+$ ]] ||
    release_die "Kimi K3 RTX_EXPERT_LAYERS is auto, all or a fixed routed-layer count, not $local_layers"
  family_args+=(--rtx-expert-layers "$local_layers")
  # Header-native EXL3 K6 core tensors are served natively (Kimi slots always
  # package the K6 programs).
fi
# DECODE_SHARE: the share of the time running requests keep while prompts
# prefill (the engine's default 0.2; 0 prefills whole prompts before the next
# step). Keys left unset pass nothing (images older than the options run).
[[ -z "$(get DECODE_SHARE)" ]] || family_args+=(--decode-share "$(get DECODE_SHARE)")
# RTX_EXPERT_LAYERS (DeepSeek V4, GLM 5.3 and MiMo): unset is uniformly pool-first
# auto; all pins every routed-expert layer of the target model and refuses (naming
# the count) when it does not fit. N fixes exactly N RTX-resident routed layers,
# then the KV pool fills the remaining budget up to its target (refused below its
# floor). 0 leaves the backbone experts on the Sparks. With two RTX every onboard
# mode uses TP2 halves; GPU1 whole-layer expert ranges are no longer supported.
if [[ $serve == serve-dsv4 || $serve == serve-glm || $serve == serve-mimo ]]; then
  local_layers="$(get RTX_EXPERT_LAYERS)"
  onboard_flag=--rtx-expert-layers
  if [[ $serve == serve-mimo ]]; then
    onboard_flag=--onboard
  fi
  case "$(get RTX_EXPERT_PEER off)" in
    on) echo "GPU1 whole-layer expert ranges were replaced by TP2 halves (v3 P4); remove RTX_EXPERT_PEER" >&2; exit 2 ;;
    off) ;;
    *) echo "RTX_EXPERT_PEER must be on or off" >&2; exit 2 ;;
  esac
  case "$local_layers" in
    "") ;;
    *) family_args+=("$onboard_flag" "$local_layers") ;;
  esac
fi
if [[ "$family" == glm5 || "$family" == glm5_flash || "$family" == deepseek_v41 || "$family" == mimo_v2 ]]; then
  draft_kv_cache="$(get DRAFT_KV_CACHE fp8)"
  case "$draft_kv_cache" in bf16|fp8) ;; *) release_die "DRAFT_KV_CACHE must be bf16 or fp8" ;; esac
  family_args+=(--draft-kv-cache "$draft_kv_cache")
fi
# GLM, GLM Flash, MiMo, Qwen: L2_PREFETCH (off, auto = 3/4 of the L2, or MiB;
# unset: auto for GLM 5.3 and GLM 5.3 Flash, off for MiMo and Qwen) pulls the
# next layer's weights into L2 during each one-lane decode step's Spark exchange; FP8_SCALES (amax, pow2, best) is the scale rule of the FP8
# copies made from BF16 weights at load.
if [[ $serve != serve-dsv4 ]]; then
  [[ -z "$(get L2_PREFETCH)" ]] || family_args+=(--l2-prefetch "$(get L2_PREFETCH)")
  [[ -z "$(get FP8_SCALES)" ]] || family_args+=(--fp8-scales "$(get FP8_SCALES)")
  if ((draft_model_args > 0)); then
    if [[ $family == mimo_v2 ]]; then
      [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || family_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
      [[ -z "$(get DRAFT_SEQUENCES)" ]] || family_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
      mimo_draft_default=""
      if [[ $mimo_flash_mopd == 1 && $speculator == dflash2 && -z "$drafter" ]]; then
        mimo_draft_default=on
      fi
      case "$(get SPECULATOR_FP8 "$mimo_draft_default")" in
        "") ;;
        auto) family_args+=(--draft-representation checkpoint) ;;
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "MiMo SPECULATOR_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
    elif [[ $family == glm5 || $family == glm5_flash ]]; then
      case "$(get SPECULATOR_FP8 auto)" in
        auto) ;; # The engine default: single-copy FP8 drafter weights.
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "SPECULATOR_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
      [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || family_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
      [[ -z "$(get DRAFT_SEQUENCES)" ]] || family_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    elif [[ "$(get SPECULATOR_FP8 on)" == off ]]; then
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
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
[[ -z "$wip_slot" ]] || spark_image="$(get SPARK_EXPERT_DOCKER_DEV cuteafd-spark-expert-dev)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
# Leave automatic weight/device budgets to the planner; an explicit
# SPARK_MEMORY_CAP applies to the full device layout (plan and serve receive it
# with the other memory policy flags) and is forwarded to every worker.
# Two coordinator GPUs (resolved above, primary first): families with a head
# split (MiMo V2 Flash/Pro, GLM 5.x, GLM 5.3 Flash, DeepSeek V4) split every layer's
# attention heads and dense / shared-expert MLPs over both by default (COORDINATOR_SPLIT=auto
# or heads), one hidden all-reduce per layer over peer memory; experts, router, head and
# drafter stay on the primary. COORDINATOR_SPLIT=off serves from the primary alone. Auto
# selection uses one GPU for checkpoints without a split and for splits that are opt-in
# (`split_opt_in`: measured not to pay at the reference layouts); an explicit split request
# (RTX_GPUS=2, COORDINATOR_GPUS=a,b or COORDINATOR_SPLIT=heads) fails before containers
# start when the checkpoint has none.
gpu="${RELEASE_COORDINATOR_GPU_INDICES[0]}"
second=""
((${#RELEASE_COORDINATOR_GPU_INDICES[@]} < 2)) || second="${RELEASE_COORDINATOR_GPU_INDICES[1]}"
explicit_split=0
if [[ "$rtx_gpus" == 2 || "$explicit_coordinator_gpus" == *,* || "$split" == heads ]]; then
  explicit_split=1
fi
[[ "$family" != qwen3_5 || "$explicit_split" == 0 ]] ||
  release_die "dense Qwen two-RTX layer ranges are not qualified"
split_hint=""
split_opt_in=""
case "$family:$model_type" in
  deepseek_v4:*|glm5:*|glm5_flash:*|mimo_v2:mimo_v2|mimo_v2:mimo_v2_flash|kimi_k3:*|step5:*|muse_spark:*) ;;
  qwen4:*)
    if [[ "$ranks" == 0 || "$explicit_split" == 1 ]]; then
      : # The planner checks expert ownership; inventory may contain only media.
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
coordinator_uuids=("$RELEASE_COORDINATOR_GPU_UUID")
head_split=0
if [[ -n "$second" && "$split" != off ]]; then
  if [[ -z "$split_hint" ]]; then
    if [[ "$family" == qwen4 ]]; then
      family_args+=(--rtx-expert-layers "$(get RTX_EXPERT_LAYERS all)")
      [[ "$attention_placement" != heads ]] || release_die "Qwen dual owns whole layers; head-split attention is not implemented"
      attention_placement=layers
    fi
    coordinator_uuids=("${RELEASE_COORDINATOR_GPU_UUIDS[@]:0:2}")
    head_split=1
    family_args+=(--device 0 --split-device 1)
  elif [[ "$explicit_split" != 1 ]]; then
    echo "note: $family ($model_type) has no head split; auto selected GPU $gpu alone" >&2
  fi
fi
[[ "$tp2_dflash" != on || "$head_split" == 1 ]] ||
  release_die "TP2_DFLASH requires an active two-GPU target split"
if [[ "$family" == qwen4 && "$attention_placement" == layers && "$head_split" != 1 ]]; then
  release_die "Qwen whole-layer attention requires a Spark-free coordinator pair"
fi
if [[ "$family" == deepseek_v4 ]]; then
  TP2_DSPARK_EXPERTS="$(get TP2_DSPARK_EXPERTS)"
  SPECULATOR="$speculator"
  release_resolve_tp2_dspark_experts "$family" "$((1 + head_split))"
  [[ "$TP2_DSPARK_EXPERTS" != on ]] || dspark_args+=(--tp2-dspark-experts)
fi
# GLM 5.3 Flash: BF16 target MLA/FFNs and KDA stay BF16 by default; native
# FP8/NVFP4 checkpoint weights retain their format. Explicit row128/channel
# KDA conversion on the two-GPU head split adds
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
  kda_fp8="$(key GLM5_FLASH_KDA_FP8 GLMF_KDA_FP8 auto)"
  case "$kda_fp8" in
    ""|auto) kda_fp8=off ;;
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
  # Continuous row units share one Spark link; every in-flight wave owns an intake plane.
  prefill_waves="$(get PREFILL_WAVES)"
  if [[ -n "$prefill_waves" ]]; then
    [[ "$prefill_waves" =~ ^[1-4]$ ]] || { echo "PREFILL_WAVES must be 1 to 4" >&2; exit 2; }
  fi
  unit_rows="$(get PREFILL_ROWS)"
  if [[ -n "$unit_rows" ]]; then
    if ! [[ "$unit_rows" =~ ^[1-9][0-9]*$ ]] || ((unit_rows > 4096 || unit_rows % 64 != 0)); then
      echo "PREFILL_ROWS must be a multiple of 64 up to 4096" >&2; exit 2
    fi
  fi
  # GLM5_FLASH_FP8_HEAD: auto preserves the checkpoint BF16 head on every topology.
  # Native heads stay unchanged. on selects E4M3; off keeps BF16; packed keeps exact 12-bit BF16
  # (mirroring V41_FP8_HEAD=packed): target logits exact, drafts through its coarse plane, about
  # three quarters of the BF16 bytes. The daemon checks packed split support after normalizing
  # native K6/EXLR8 heads, which ignore this conversion setting.
  fp8_head="$(key GLM5_FLASH_FP8_HEAD GLMF_FP8_HEAD auto)"
  case "$fp8_head" in
    ""|auto) fp8_head=auto ;;
    on|off|packed) ;;
    *) echo "GLM5_FLASH_FP8_HEAD must be auto, on, off or packed" >&2; exit 2 ;;
  esac
  case "$fp8_head" in
    on) family_args+=(--fp8-head true) ;;
    off) family_args+=(--fp8-head false) ;;
    auto|packed) family_args+=(--fp8-head "$fp8_head") ;;
  esac
  # GLM5_FLASH_DRAFT_LINEAR: w8a8 with an FP8 drafter; w8a16 runs the W8A16 GEMV in passes of
  # 64 rows), wide (the same bits in passes of 128 rows) or w8a8 (past one draft block, E4M3
  # activations per row and 128-wide K block on FP8 tensor cores). Drafts only.
  draft_linear_default=w8a16
  if [[ "$draft_model_args" != 0 && "$(get SPECULATOR_FP8 auto)" != off ]]; then draft_linear_default=w8a8; fi
  draft_linear="$(get GLM5_FLASH_DRAFT_LINEAR "$draft_linear_default")"
  case "$draft_linear" in
    ""|w8a16) ;;
    wide|w8a8) family_args+=(--draft-linear "$draft_linear") ;;
    *) echo "GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_TARGET_HEAD: GLM 5.3 Flash's own spelling of TARGET_HEAD (--target-head). auto
  # (default) forwards nothing: the shared setting applies (tensor cores at every row count over the
  # BF16 head; the FP8 and packed heads run their own GEMVs). exact (diagnostic) keeps the pedantic
  # GEMM; an explicit tensor with the FP8 or packed head is refused.
  target_head="$(get GLM5_FLASH_TARGET_HEAD auto)"
  case "$target_head" in
    ""|auto) ;;
    exact) family_args+=(--target-head exact) ;;
    tensor)
      if [[ $fp8_head != off ]]; then
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
  # BF16 or single-copy FP8 KDA projections on one GPU (no head split).
  kda_state="$(get GLM5_FLASH_KDA_STATE f32)"
  case "$kda_state" in
    ""|f32) ;;
    bf16|bf16-tile)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_KDA_STATE=$kda_state runs on one GPU, without a head split" >&2
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
  # arena (two staging marks with host retention, legacy counts with host off) or pool (units
  # of the KV pool itself, unit 0 reserved; no separate arena). Both use the shared host default.
  prefix_marks="$(get GLM5_FLASH_PREFIX_MARKS)"
  case "$prefix_marks" in
    "") ;;
    arena|pool) family_args+=(--prefix-marks "$prefix_marks") ;;
    *) echo "GLM5_FLASH_PREFIX_MARKS must be arena or pool" >&2; exit 2 ;;
  esac
  # PREFIX_CACHE_MARK_MIB: the device budget of the prefix mark arena, MiB (unset: serve-glmf's
  # 2048), as for MiMo; the host-off legacy arena retains its per-sequence floor.
  [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || family_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
  # GLM5_FLASH_REPLAY_RECORDS: auto shares when eligible; own keeps their
  # own 321 MB, 642 MB with GLM5_FLASH_DECODE_ROWS=128) or shared (the prefill lanes' scratch, which
  # no decode step reads). One GPU whose pool is sized from measured memory (an automatic pool with
  # Spark experts).
  requested_replay_records="$(get GLM5_FLASH_REPLAY_RECORDS auto)"
  case "$requested_replay_records" in
    ""|auto|own) ;;
    shared)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_REPLAY_RECORDS=shared keeps the records in one GPU's prefill scratch; serve it without a head split" >&2
        exit 2
      fi
      family_args+=(--replay-records shared) ;;
    *) echo "GLM5_FLASH_REPLAY_RECORDS must be own or shared" >&2; exit 2 ;;
  esac
  glmf_replay_pos=${#family_args[@]}
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
  # GLM5_FLASH_ROUTE_RING_CHECK=on (diagnostic) also copies Spark layers' router ids through the local
  # route ring and counts disagreements with the staged ids.
  case "$(get GLM5_FLASH_ROUTE_RING_CHECK off)" in
    off) ;;
    on) trace_args+=(-e "CUTEAFD_GLMF_ROUTE_RING_CHECK=1") ;;
    *) echo "GLM5_FLASH_ROUTE_RING_CHECK must be on or off" >&2; exit 2 ;;
  esac
fi
# INSTANCE names a launch that runs beside others on disjoint hardware
# (`cuteafd bench smoke` sets it): its coordinator container is
# cuteafd-coordinator-INSTANCE; empty keeps the one cuteafd-coordinator.
# Local-expert argv follows deployment admission, not the physical Spark count.
# Match persistent connections owned by the selected family's transport, not
# the Spark world size. Forward the resolved lane policy to the coordinator.
lane_env_args=()
case "$family" in
  deepseek_v4)
    [[ -z "$(get PREFILL_SPAN_ROWS)" ]] || lane_env_args+=(-e "CUTEAFD_PREFILL_SPAN_ROWS=$(get PREFILL_SPAN_ROWS)")
    ;;
  glm5)
    # Plan and serve resolve the same explicit target conversion policy.
    glm_bf16="$(get CUTEAFD_GLM_BF16 "${CUTEAFD_GLM_BF16:-native}")"
    case "$glm_bf16" in
      native|fp8) lane_env_args+=(-e "CUTEAFD_GLM_BF16=$glm_bf16") ;;
      *) release_die "CUTEAFD_GLM_BF16 must be native or fp8" ;;
    esac
    ;;
  glm5_flash|qwen4|qwen3_5|kimi_k3|step5|muse_spark|mimo_v2) ;;
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
  plan_cap="$profile_cap"
  if [[ "$family" == qwen4 ]] && ((!dry_run_total_capacity)); then
    total_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
    free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if ((restart)); then
      own_pids="$(docker top "$coordinator_name" -eo pid 2>/dev/null | python3 -c 'import sys; print(" ".join(x.strip() for x in sys.stdin if x.strip().isdigit()))' || true)"
      if [[ -n "$own_pids" ]]; then
        own_mib="$(nvidia-smi --id="$gpu" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits \
          | python3 -c 'import csv,sys;p=set(sys.argv[1].split());print(sum(int(r[1].strip()) for r in csv.reader(sys.stdin) if len(r)==2 and r[0].strip() in p and r[1].strip().isdigit()))' "$own_pids")"
        free_mib=$((free_mib + own_mib))
      fi
    fi
    # Occupancy stays wholly NVML-derived, including our restart PID credit.
    # Subtract that explicit used-memory term from the CUDA/logical capacity.
    plan_cap="$(python3 -c 'import sys;f,t,b=map(int,sys.argv[1:]);print(max(1,b-(t-f)*2**20))' "$free_mib" "$total_mib" "$profile_cap")"
  fi
  # Plan and serve admit resident weights against the same resolved budget.
  coordinator_weight_budget_gib="$(release_gib "$plan_cap")"
  plan_draft_args=(--concurrency "$effective_concurrency" --mapped-scale-preload-gib "$(get MAPPED_SCALE_PRELOAD_GIB 0)")
  if [[ ( "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5_flash || "$family" == muse_spark ) && "$tp2_dflash" != auto ]]; then
    plan_draft_args+=(--tp2-dflash "$tp2_dflash_bool")
  fi
  [[ -z "$decode_lanes" ]] || plan_draft_args+=(--decode-lanes "$decode_lanes")
  if [[ "$family:$speculator" == qwen3_5:dspark ]]; then
    plan_draft_args+=(--drafter-snapshot "$draft_snapshot")
  fi
  if [[ "$family:$speculator" == kimi_k3:dflash2 ]]; then
    plan_draft_args+=(--drafter-snapshot "$draft_snapshot" --draft-fp8 true)
  fi
  if [[ "$family:$speculator" == muse_spark:dflash2 ]]; then
    plan_draft_args+=(--drafter-snapshot "$draft_snapshot")
  fi
  # Read the effective serve argv, including family modes that enable auto.
  plan_host_cache=0
  for ((i=0; i<${#family_args[@]}; i++)); do
    [[ "${family_args[i]}" != --host-cache-bytes ]] || plan_host_cache="${family_args[i+1]}"
    [[ "${family_args[i]}" != --media-cache-bytes ]] || plan_draft_args+=(--media-cache-bytes "${family_args[i+1]}")
    if [[ "$family" == glm5_flash && ( "${family_args[i]}" == --draft-fp8 || "${family_args[i]}" == --draft-linear ) ]]; then
      plan_draft_args+=("${family_args[i]}" "${family_args[i+1]}")
    fi
  done
  plan_draft_args+=(--host-cache-bytes "$plan_host_cache")
  [[ "$(get FULL_PREFILL_LOGITS off)" != on ]] || plan_draft_args+=(--full-prefill-logits)
  case "$(get EXPERT_USAGE on)" in
    on) plan_draft_args+=(--expert-usage) ;;
    off) plan_draft_args+=(--expert-usage=false) ;;
  esac
  if [[ "$family" == qwen4 ]]; then
    plan_draft_args+=(--native-mtp-layers "$(get SPECULATOR_DEPTH "$mtp_depth")")
  fi
  if [[ "$family" == qwen3_5 ]]; then
    plan_draft_args+=(--kv-cache "$dense_qwen_kv_cache" --prefill-rows 4096
      --prefix-cache-entries "$effective_prefix_entries")
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
  fi
  if [[ "$family" == mimo_v2 ]]; then
    plan_draft_args+=(--mimo-swa-kv-cache "$mimo_swa_kv")
    plan_draft_args+=(--prefix-cache-entries "$effective_prefix_entries")
    plan_draft_args+=(--expert-input "${expert_input:-fp8}")
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "${local_layers:-}" ]] || plan_draft_args+=(--rtx-expert-layers "$local_layers")
    plan_draft_args+=(--mimo-decode-graphs "$mimo_decode_graphs")
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
    [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || plan_draft_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
    [[ -z "$(get DRAFT_SEQUENCES)" ]] || plan_draft_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    if [[ "$(get MIMO_PREFIX_DRAFT off)" == on ]]; then
      plan_draft_args+=(--mimo-prefix-draft)
    fi
    # Plan the native MTP depth serve-mimo receives (its --mtp, default 0).
    mimo_plan_mtp=0
    for ((i=0; i<${#family_args[@]}; i++)); do
      [[ "${family_args[i]}" != --mtp ]] || mimo_plan_mtp="${family_args[i+1]}"
    done
    plan_draft_args+=(--native-mtp-layers "$mimo_plan_mtp")
  fi
  # Rust resolves recurrent storage and prefix defaults from the same concurrency.
  if [[ "$family" == glm5_flash ]]; then
    plan_draft_args+=(--prefix-cache-entries "$effective_prefix_entries")
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
    # The plan resolves VISION=auto as serve-glmf would serve it: with the template source serving
    # gets, vision stays on; without one, a checkpoint template that cannot mark images plans text.
    [[ -z "$chat_template_from" || "$vision" == off ]] || plan_draft_args+=(--chat-template-from "$chat_template_from")
  fi
  if [[ "$family" == glm5_flash ]]; then
    case "$fp8_head" in on) plan_head=true ;; off) plan_head=false ;; auto|packed) plan_head="$fp8_head" ;; esac
    plan_draft_args+=(--kda-fp8 "$kda_fp8" --kda-state "${kda_state:-f32}" --fp8-head "$plan_head")
    [[ "${target_head:-auto}" == auto || -z "${target_head:-}" ]] || plan_draft_args+=(--target-head "$target_head")
    # Draft-head arithmetic borrows the same target weight, not a second resident head.
    plan_draft_args+=(--draft-head exact)
  fi
  [[ "$family" != glm5_flash || "${index_cache:-keys}" != compact ]] || plan_draft_args+=(--index-cache compact)
  # GLM 5.3 Flash: the server's prefix-mark store, so the plan reserves an arena only when serve-glmf allocates one.
  [[ "$family" != glm5_flash || -z "${prefix_marks:-}" ]] || plan_draft_args+=(--prefix-marks "$prefix_marks")
  # GLM 5.3 Flash: shared replay records (GLM5_FLASH_REPLAY_RECORDS above) live in the prefill
  # scratch, so the plan does not charge their own copy either.
  if [[ "$family" == glm5_flash ]]; then
    case "${requested_replay_records:-auto}" in
      shared) plan_draft_args+=(--replay-records shared) ;;
      ""|auto) plan_draft_args+=(--replay-records auto) ;;
    esac
  fi
  # GLM 5.3 Flash plans the decode rows serving takes (GLM5_FLASH_DECODE_ROWS above): 128 rows
  # charge their wider decode workspace, selector, replay records and expert intake, as serving
  # admits them, before an encoder placement is chosen.
  if [[ "$family" == glm5_flash && "${decode_rows:-64}" == 128 ]]; then
    plan_draft_args+=(--decode-rows 128)
  fi
  if [[ "$family" == kimi_k3 ]]; then
    plan_draft_args+=(--kv-cache "$kv_cache")
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
  fi
  [[ -z "$(get SPARK_PREFILL_REPLY_FORMAT)" ]] || plan_draft_args+=(--spark-prefill-reply-format "$(get SPARK_PREFILL_REPLY_FORMAT)")
  [[ -z "$(get SPARK_DECODE_REPLY_FORMAT)" ]] || plan_draft_args+=(--spark-decode-reply-format "$(get SPARK_DECODE_REPLY_FORMAT)")
  if [[ "$family" == muse_spark ]]; then
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
    # The serve command's partial-reply overrides and drafter rings, so plan == serve admission.
    [[ -z "$(get EXPERT_PARTIAL_BF16_MAX_ROWS)" ]] || plan_draft_args+=(--expert-partial-bf16-max-rows "$(get EXPERT_PARTIAL_BF16_MAX_ROWS)")
    [[ "$speculator" != dflash2 ]] || plan_draft_args+=(--draft-kv-cache "${muse_draft_kv:-fp8}")
  fi
  if [[ "$family" == glm5 || "$family" == glm5_flash || "$family" == deepseek_v41 || "$family" == mimo_v2 ]]; then
    plan_draft_args+=(--draft-kv-cache "$draft_kv_cache")
  fi
  if [[ "$family" == glm5 ]]; then
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
    [[ -z "$(get PREFILL_SPAN_ROWS)" ]] || plan_draft_args+=(--prefill-span-rows "$(get PREFILL_SPAN_ROWS)")
  fi
  if [[ "$family" == qwen4 ]]; then
    plan_draft_args+=(--kv-cache "$kv_cache")
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
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
  if [[ "$family" == glm5_flash ]]; then
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
  fi
  if [[ "$family" == deepseek_v4 ]]; then
    [[ -z "$(get PREFILL_ROWS)" ]] || plan_draft_args+=(--prefill-rows "$(get PREFILL_ROWS)")
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
  fi
  if [[ "$family" == step5 || "$family" == mimo_v2 ]]; then
    [[ -z "$(get PREFILL_WAVES)" ]] || plan_draft_args+=(--prefill-waves "$(get PREFILL_WAVES)")
  fi
  if [[ "$family" == step5 ]]; then
    plan_mtp_depth=0
    [[ "$speculator" != mtp ]] || plan_mtp_depth="$(get SPECULATOR_DEPTH "$mtp_depth")"
    plan_draft_args+=(--native-mtp-layers "$plan_mtp_depth" --kv-cache "$step5_kv_cache" --step5-decode-graphs "$step5_decode_graphs")
  fi
  plan_draft_args+=(--max-context-tokens "$(get MAX_CONTEXT_TOKENS auto)")
  for pair in 'CONCURRENCY:--concurrency' 'PREFIX_CACHE_ENTRIES:--prefix-cache-entries' \
    'PREFILL_BATCH_TOKENS:--prefill-rows' \
    'RTX_EXPERT_LAYERS:--rtx-expert-layers'; do
    key_name="${pair%%:*}"; flag="${pair#*:}"
    [[ "$family:$key_name" != mimo_v2:PREFILL_BATCH_TOKENS ]] || continue
    value="$(get "$key_name")"
    if [[ -n "$value" && "$value" != auto && " ${plan_draft_args[*]} " != *" $flag "* ]]; then
      plan_draft_args+=("$flag" "$value")
    fi
  done
  release_reject_unused_keys "$family" $(_family_allowed_keys "$family" "$detected_family")
  deployment_args=()
  deployment_peers=()
  for ((rank = 0; rank < available_sparks; rank++)); do
    deployment_peers+=("$(get "SPARK_${rank}_LANE_A"):$port")
  done
  ((${#deployment_peers[@]} == 0)) || deployment_args+=(--deployment-peers "$(IFS=,; printf '%s' "${deployment_peers[*]}")")
  [[ -z "$(get KV_POOL_SIZE)" ]] || deployment_args+=(--kv-pool-size "$(get KV_POOL_SIZE)")
  readiness_args=(--require-ready)
  [[ -z "$wip_slot" ]] || readiness_args=()
  plan_config_mounts=()
  if [[ "$family" == deepseek_v4 ]]; then
    # Pass the resolved value, including explicit off, to default-on admission.
    plan_config_text="$(printf '%s=%s\n' TP2_DSPARK_EXPERTS "$TP2_DSPARK_EXPERTS"
      for ((rank = 0; rank < available_sparks; rank++)); do
        printf '%s=%s\n' "SPARK_${rank}_HOST" "$(get "SPARK_${rank}_HOST")"
      done)"
    plan_config_dir="$HOME/.cache/cuteafd/run-config"
    mkdir -p "$plan_config_dir"
    plan_config="$plan_config_dir/plan-$(printf '%s\n' "$plan_config_text" | sha256sum | cut -c1-16).config"
    printf '%s\n' "$plan_config_text" >"$plan_config.tmp.$$" && mv "$plan_config.tmp.$$" "$plan_config"
    count_args+=(--config /run/cuteafd-plan.config)
    plan_config_mounts=(-v "$plan_config:/run/cuteafd-plan.config:ro")
  fi
  # Keep checkpoint paths unchanged inside the full namespace mount. The plan
  # admits under the same instrumentation the server will launch with, so a
  # profiled run (e.g. PROFILE=census) plans what that server actually charges.
  # Only the environment switches are forwarded: the plan writes no reports, so
  # it neither needs nor creates PROFILE_DIR (shared_profile_args owns that, and
  # it still runs after the Spark profile validation below).
  plan_profile_env=()
  if [[ "$family" == qwen3_5 ]]; then
    case "$(get DECODE_GRAPHS on)" in
      on) plan_profile_env+=(-e CUTEAFD_QWEN35_GRAPHS=1) ;;
      off) plan_profile_env+=(-e CUTEAFD_QWEN35_GRAPHS=0) ;;
      *) release_die "DECODE_GRAPHS must be on or off" ;;
    esac
  fi
  if [[ "$family" == mimo_v2 || "$family" == kimi_k3 || "$family" == glm5 || "$family" == glm5_flash || "$family" == qwen4 || "$family" == deepseek_v4 || "$family" == deepseek_v41 || "$family" == qwen3_5 || "$family" == step5 ]]; then
    for key in GPU_GRAPH_RESERVE GPU_GRAPH_RESERVE_EXTRA GPU_GRAPH_RESERVE_EXTRA_CAP SPARK_GRAPH_RESERVE; do
      [[ "$family" != kimi_k3 || "$key" != SPARK_GRAPH_RESERVE ]] || continue
      [[ -z "$(get "$key")" ]] || plan_profile_env+=(-e "CUTEAFD_$key=$(get "$key")")
    done
  fi
  release_profile_docker_args "$(get PROFILE)" "$(get PROFILE_DIR)" "$(hostname -s)" "$(get PROFILE_SPARK)"
  plan_profile_mount=0
  for arg in ${RELEASE_PROFILE_ARGS[@]+"${RELEASE_PROFILE_ARGS[@]}"}; do
    if ((plan_profile_mount)); then plan_profile_mount=0; continue; fi
    [[ "$arg" != -v ]] || { plan_profile_mount=1; continue; }
    plan_profile_env+=("$arg")
  done
  plan_json="$(docker run --rm --network none ${plan_profile_env[@]+"${plan_profile_env[@]}"} "${lane_env_args[@]}" -e "HF_HOME=$container_hf_home" "${generic_worker_mounts[@]}" "${chat_template_mounts[@]}" "${wip_mount_args[@]}" "${plan_config_mounts[@]}" \
    "$coordinator_image" cuteafd plan "$snapshot" --attention-placement "$attention_placement" --vision "$vision" --audio "$audio" --deployment "${readiness_args[@]}" \
    "${deployment_args[@]}" "${count_args[@]}" "${exlr8_plan_args[@]}" "${core_view_args[@]}" "${plan_memory_args[@]}" \
    --rtx "$plan_rtx" --gpu-memory-cap "$plan_cap" --coordinator-weight-budget-gib "$coordinator_weight_budget_gib" --pool-tokens "$plan_pool" --vision-replicas "$vision_replicas" "${plan_draft_args[@]}")"
  release_validate_deployment "$plan_json" "$checkpoint_family"
  RELEASE_RECLAIM_ARGS=()
  ((restart == 0)) || release_coordinator_reclaim_args "$coordinator_name"
  release_admit_coordinator_gpus "$(IFS=,; printf '%s' "${coordinator_uuids[*]}")" "$plan_json" \
    ${RELEASE_RECLAIM_ARGS[@]+"${RELEASE_RECLAIM_ARGS[@]}"}
  selected_ranks="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spark_ranks"])' <<<"$plan_json")"
  if ((count_explicit)) || [[ "$backend" == local ]]; then
    ((selected_ranks <= ranks)) || release_die "planner requested more Spark ranks than configured: $selected_ranks > $ranks"
    [[ "$selected_ranks" == "$ranks" ]] || echo "Spark roles use $selected_ranks of $ranks configured ranks"
  fi
  deployment_peer_csv="$(jq -er '.peers | join(",")' <<<"$plan_json")"
  if [[ "$speculator" != auto ]]; then
    [[ "$(jq -r '.speculator' <<<"$plan_json")" == "$speculator" ]] || release_die "deployment changed resolved speculator"
  fi
  # Media still reads the complete report's capability diagnostics. Launch roles,
  # admission policy and endpoints come from the typed deployment above.
  deployment_json="$plan_json"
  release_deployment_spark_roles "$deployment_json" "$ranks"
  if [[ "$family:$speculator" == qwen4:auto ]]; then
    speculator="$(jq -er '.speculator | select(. == "mtp" or . == "off")' <<<"$deployment_json")" ||
      release_die "Qwen deployment does not name a resolved speculator"
    [[ "$speculator" != mtp ]] || family_args+=(--mtp "$(get SPECULATOR_DEPTH "$mtp_depth")")
    echo "Qwen planner selected SPECULATOR=$speculator (admitted Spark expert world $spark_expert_world)"
  fi
  if [[ "$family" == glm5_flash ]]; then
    resolved_replay_records="$(jq -er '.replay_records | select(. == "own" or . == "shared")' <<<"$deployment_json")" ||
      release_die "GLM Flash deployment does not name resolved replay records"
    replay_eligible=0
    if [[ "$resolved_replay_records" == shared ]]; then
      replay_eligible=1; replay_reason="one GPU; automatic measured pool with Spark experts"
    elif [[ $head_split != 0 ]]; then replay_reason="head split keeps its own records"
    elif [[ $glmf_pool != 0 ]]; then replay_reason="fixed pool keeps its own records"
    else replay_reason="local experts keep their own records"; fi
    replay_records="$(release_glmf_auto GLM5_FLASH_REPLAY_RECORDS "${requested_replay_records:-auto}" \
      shared own "$replay_eligible" "$replay_reason")"
    if [[ "${requested_replay_records:-auto}" != shared && "$replay_records" == shared ]]; then
      family_args=("${family_args[@]:0:glmf_replay_pos}" --replay-records shared "${family_args[@]:glmf_replay_pos}")
    fi
  fi
  if [[ "$family" == qwen4 && "$head_split" == 1 && "$spark_expert_world" != 0 ]]; then
    release_die "Qwen dual requires a Spark-free local TP2 layout"
  fi
  if ((spark_expert_world == 0)); then
    case "$family" in
      glm5_flash|mimo_v2|qwen4) family_args+=(--local-experts) ;;
    esac
  fi
  plan_json="$(jq -c '.report' <<<"$deployment_json")"
  if [[ "$family" == glm5_flash && "$tp2_dflash" == auto ]]; then
    tp2_dflash_bool="$(jq -er '.tp2_dflash | if . == true then "true" elif . == false then "false" else error("missing drafter choice") end' <<<"$plan_json")"
    family_args+=(--tp2-dflash "$tp2_dflash_bool")
    echo "GLM Flash planner selected TP2_DFLASH=$tp2_dflash_bool"
  fi
  release_prefill_shape "$deployment_json" "$family"
  resolved_decode_lanes="$(jq -er '.decode_lanes | select(. == 1 or . == 2)' <<<"$deployment_json")" ||
    release_die "deployment does not name a supported decode lane count"
  if [[ -n "$decode_lanes" ]]; then
    case "$family" in
      deepseek_v41|mimo_v2) family_args+=(--decode-lanes "$resolved_decode_lanes") ;;
    esac
  fi
  case "$(family_prefill_mode "$family")" in
    units|units-span) family_args+=(--prefill-rows "$PREFILL_ROWS" --prefill-waves "$PREFILL_WAVES") ;;
  esac
  if [[ "$(family_prefill_mode "$family")" == units-span ]]; then
    family_args+=(--prefill-span-rows "$PREFILL_SPAN_ROWS")
  fi
  if ((spark_expert_world > 0)); then
    budget="$(release_spark_worker_budget "$deployment_json" "$spark_expert_world")" || exit 1
  fi
# Check the admitted count's images before --restart or weight loading. The worker's
# resident admission validates the sibling for its full 4096-row workspace even
# when only decode requests use BF16, so this preflight requires the same coverage.
if [[ -n "$expert_input" && "$expert_input" != fp8 ]]; then
  [[ "$spark_expert_world" != 0 ]] || { echo "EXPERT_INPUT=$expert_input requires Spark experts" >&2; exit 2; }
  store_dtype="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("quantization_config", {}).get("store_dtype", "fp8"))' "$root/snapshots/$revision/$selected_config")"
  case "$store_dtype" in
    fp8) expert_geometry=mimo ;;
    mxfp4)
      hidden="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("text_config", c).get("hidden_size", 0))' "$root/snapshots/$revision/$selected_config")"
      case "$hidden" in
        4096) expert_geometry=mimof ;;
        6144) expert_geometry=mimop ;;
        *) echo "EXPERT_INPUT=$expert_input has no MXFP4 MiMo package for hidden_size=$hidden" >&2; exit 2 ;;
      esac ;;
    *) echo "EXPERT_INPUT=$expert_input has no package for store_dtype=$store_dtype" >&2; exit 2 ;;
  esac
  printf -v preflight_command '%q ' docker run --rm -i --entrypoint python3 "$spark_image" - "$expert_geometry" "tp$spark_expert_world" 4096
  for rank in "${spark_expert_ranks[@]}"; do
    host="$(get "SPARK_${rank}_HOST")"
    if ! ssh "$host" "$preflight_command" < "$repo_root/scripts/launch/preflight-fp8-bf16.py"; then
      echo "$host cannot serve EXPERT_INPUT=$expert_input; build $expert_geometry:fp8 with CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$expert_geometry (or its WIP equivalent)" >&2
      exit 2
    fi
  done
fi
if { [[ ( "$family" == mimo_v2 || "$family" == qwen4 || "$family" == glm5_flash
          || "$family" == muse_spark || "$checkpoint_family" == mistral_large4 ) && "$vision" != off ]] &&
   python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("vision_config") else 1)' "$root/snapshots/$revision/$selected_config"; } ||
   [[ "$audio" != off ]]; then
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
    assert len(ranks)==len(set(ranks)) and all(r in json.loads(sys.argv[2]) for r in ranks)
    print("spark:"+str(k["rank"]),h,",".join(map(str,ranks)),cap)
elif kind=="rtx": print("rtx:"+str(k["gpu"]),h,"-",cap)
elif kind=="off": print("off",h,"-",cap)
else: raise ValueError("idle-host launch needs an explicit inventory")
' "$family" "$(printf '%s\n' "${spark_vision_ranks[@]}" | jq -sc '[.[] | select(. != null)]')" <<<"$plan_json")"
  read -r vision encoder_hash rank_csv encoder_max_tokens <<<"$selected"
  if [[ "$vision" == spark:* ]]; then
    IFS=, read -r -a encoder_ranks <<<"$rank_csv"
    for encoder_rank in "${encoder_ranks[@]}"; do
      vision_peers+=("$(get "SPARK_${encoder_rank}_LANE_A"):$encoder_port")
    done
    family_args+=(--vision-peers "$(IFS=,; printf '%s' "${vision_peers[*]}")" --encoder-plan-hash "$encoder_hash" --encoder-revision "$revision")
    if [[ "$family" == glm5_flash ]]; then
      # Internal plan-derived physical identities, never a user configuration key.
      family_args+=(--vision-peer-ranks "$(IFS=,; printf '%s' "${encoder_ranks[*]}")")
    fi
  fi
  if [[ "$audio" != off ]]; then
    selected_audio="$(python3 -c '
import json,sys
p=json.load(sys.stdin); e=p.get("audio_encoder")
k=e["kind"] if e else {"kind":"off"}; kind=k["kind"]; h=p["encoder_plan_hash"]
assert len(h)==64 and all(c in "0123456789abcdef" for c in h), "invalid audio plan hash"
if kind=="spark":
    ranks=[k["rank"]]+e["replicas"]
    assert len(ranks)==len(set(ranks)) and all(r in json.loads(sys.argv[2]) for r in ranks)
    print("spark:"+str(k["rank"]),h,",".join(map(str,ranks)))
elif kind=="rtx": print("rtx:"+str(k["gpu"]),h,"-")
elif kind=="off" and sys.argv[1]=="auto":
    if e and e.get("shortfall",0): print("audio auto disabled: "+e["reason"],file=sys.stderr)
    print("off",h,"-")
else: raise ValueError("enabled audio has no launchable admitted owner")
' "$audio" "$(printf '%s\n' "${spark_audio_ranks[@]}" | jq -sc '[.[] | select(. != null)]')" <<<"$plan_json")"
    read -r audio audio_encoder_hash audio_rank_csv <<<"$selected_audio"
    if [[ "$audio" == spark:* ]]; then
      audio_encoder_ranks=("${spark_audio_ranks[@]}")
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
[[ -z "$deployment_peer_csv" ]] || IFS=, read -r -a peers <<<"$deployment_peer_csv"
# Recipe registration is not proof that this seal/image contains its TP variants.
package_requirements="$(jq -c '.artifact_requirements.spark_packages // {}' <<<"$deployment_json")"
if [[ "$package_requirements" != '{}' ]]; then
  for rank in "${spark_expert_ranks[@]}"; do
    host="$(get "SPARK_${rank}_HOST")"
    identity="$checkpoint_family $host image $spark_image"
    if [[ -n "$wip_slot" ]]; then
      # Each Spark resolves its own home (~ expands there): the coordinator's HOME
      # need not be the Spark's, as on hosts whose HOME is moved off a small root.
      slot_root="${WIP_ROOT:-~/.cache/cuteafd/builds/wip${WIP_INSTANCE:+-$WIP_INSTANCE}}/slots/$wip_slot/spark-expert/workspace/.cuteafd-wip"
      printf -v package_command '%q ' python3 - "$package_requirements" "$identity slot $wip_slot ($slot_root)" "$slot_root"
    else
      printf -v package_command '%q ' docker run --rm --network none -i --entrypoint python3 "$spark_image" - "$package_requirements" "$identity"
    fi
    timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$package_command" < "$repo_root/scripts/launch/preflight-spark-packages.py" ||
      release_die "$host is missing deployment Spark packages (before startup)"
  done
fi
if ((dry_run)); then
  # No serving container, restart or remote staging below this point is allowed
  # during validation. Runtime-only proofs (including audio exports) stay pending.
  printf '%s\n' "$deployment_json"
  exit 0
fi
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
  cleanup_ranks=("${spark_worker_ranks[@]}")
  if ((restart_all)); then mapfile -t cleanup_ranks < <(seq 0 $((configured_ranks - 1))); fi
  for rank in "${cleanup_ranks[@]}"; do
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
if [[ -n "$wip_slot" ]]; then
  # The slot must exist everywhere before anything starts, and the development images must
  # carry the SparkInfer revision this checkout pins.
  pinned_sparkinfer="$(python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" --print-revision)"
  for rank in "${spark_worker_ranks[@]}"; do
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
worker_pids=()
for rank in "${spark_worker_ranks[@]}"; do
  host="$(get "SPARK_${rank}_HOST")"
  encoder_args=""
  encoder_flag=--encoder audio_encoder_flag=--audio-encoder
  worker_budget="$(release_spark_role_budget "$deployment_json" "$rank")" || exit 1
  expert_args="--rank $rank --device-budget-bytes $worker_budget"
  if release_spark_role_has_experts "$deployment_json" "$rank"; then
    expert_args="--rank $rank --world $spark_expert_world --device-budget-bytes $worker_budget $spark_worker_args --listen 0.0.0.0:$port"
  else
    encoder_flag=--encoder-only audio_encoder_flag=--audio-encoder-only
  fi
  for encoder_rank in "${encoder_ranks[@]}"; do
    if [[ "$rank" == "$encoder_rank" ]]; then
      encoder_args="$encoder_flag --encoder-listen 0.0.0.0:$encoder_port --encoder-plan-hash $encoder_hash --encoder-revision $revision --encoder-max-tokens $encoder_max_tokens"
    fi
  done
  for audio_rank in "${audio_encoder_ranks[@]}"; do
    if [[ "$rank" == "$audio_rank" ]]; then
      encoder_args+=" $audio_encoder_flag --audio-encoder-listen 0.0.0.0:$audio_encoder_port --audio-encoder-plan-hash $audio_encoder_hash --audio-encoder-revision $revision"
    fi
  done
  release_profile_docker_args "$(get PROFILE)" "$(get PROFILE_DIR)" "$host" "$(get PROFILE_SPARK)"
  spark_profile_wrapper_for "$host" "$rank" "$ranks"
  RELEASE_PROFILE_ARGS+=(${spark_profile_wrapper_args[@]+"${spark_profile_wrapper_args[@]}"})
  worker_profile="" worker_prep="" worker_wrapper="" worker_memory_args=""
  # SPARK_MEMORY_CAP / SPARK_RESERVE: the worker's own reserve components follow the policy too.
  [[ -z "${spark_memory_args[*]-}" ]] || worker_memory_args="$(printf '%q ' "${spark_memory_args[@]}")"
  ((${#RELEASE_PROFILE_ARGS[@]} == 0)) || worker_profile=" $(printf '%q ' "${RELEASE_PROFILE_ARGS[@]}")"
  ((${#spark_profile_wrapper[@]} == 0)) || worker_wrapper="$(printf '%q ' "${spark_profile_wrapper[@]}")"
  [[ -z "$(get PROFILE_DIR)" ]] || worker_prep="mkdir -p $(printf '%q' "$(get PROFILE_DIR)") && release_require_writable_state $(printf '%q' "$(get PROFILE_DIR)") && "
  ssh "$host" "$(declare -f release_die release_hub_mounts release_serving_docker_args release_require_writable_state cuteafd_cdi_require cuteafd_cdi_gpu_args); set -euo pipefail; export HF_HOME=$(printf %q "$hf_home"); release_hub_mounts $(printf '%q ' "$namespace_root" "$site_hub" "$hf_home" "$snapshot_rel"); release_serving_docker_args; cuteafd_cdi_gpu_args all || exit 2; ${worker_prep}docker run -d --name cuteafd-spark-expert-$host-$port --restart no \"\${CUTEAFD_CDI_ARGS[@]}\" --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \"\${RELEASE_SERVING_ARGS[@]}\" -e RUST_LOG=info $(printf '%q ' "${shared_worker_env_args[@]}")$spark_worker_env $wip_worker_args $device_map_env$worker_profile \
    \"\${RELEASE_HUB_MOUNTS[@]}\" '$spark_image' \
    ${worker_wrapper}cuteafd ${worker_memory_args}expertd-native --snapshot \"\$RELEASE_HUB_SNAPSHOT\" --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    $expert_args $encoder_args >/dev/null" &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do wait "$pid"; done
for rank in "${spark_worker_ranks[@]}"; do
  host="$(get "SPARK_${rank}_HOST")"
  # A placement worker bootstraps after media admission and waits for the
  # coordinator solve before expert weights load.
  ready_marker="$(release_spark_role_ready_marker "$deployment_json" "$rank")"
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
text=re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", sys.stdin.read())
matches=[backend for line in text.splitlines() if "audio encoder ready" in line
    for backend in re.findall(r"\bbackend=\"?([^\s\"]+)", line)]
assert matches and len(set(matches))==1, "missing or conflicting ready audio backend identity"
backend=matches[0]
assert re.fullmatch(r"(?:mimo_audio_fp32_v1/cuda[0-9]+/cufft[0-9]+/cublas[0-9.]+/cute_aot_sm121/export[0-9a-f]{64}|pcm-bf16-aot-v1)", backend), "invalid ready audio backend identity"
print(backend)')"
    [[ -z "$audio_backend" || "$audio_backend" == "$backend" ]] || release_die "audio replicas have different backend exports"
    audio_backend="$backend"
  done
  # RVQ backends identify architecture-specific exports; PCM has a fixed protocol identity.
  if [[ "$family" == mimo_v2 ]]; then
    family_args+=(--audio-encoder-backend "$audio_backend")
  else
    [[ "$audio_backend" == pcm-bf16-aot-v1 ]] || release_die "unexpected PCM audio backend: $audio_backend"
  fi
fi
peer_csv="$(IFS=,; echo "${peers[*]}")"
peer_args=()
[[ -z "$peer_csv" ]] || peer_args=(--peers "$peer_csv")
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
release_coordinator_device_args "${coordinator_uuids[@]}"
# Automatic context passes unchanged through planner and serve admission.
context_args=(--max-context-tokens "$(get MAX_CONTEXT_TOKENS auto)")
docker run -d "${RELEASE_SERVING_ARGS[@]}" --name "$coordinator_name" --restart no "${RELEASE_COORDINATOR_DEVICE_ARGS[@]}" --network host --ipc host \
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info "${shared_env_args[@]:0:4}" \
  "${bond_args[@]}" "${bench_nonce_env_args[@]}" "${tp2_env_args[@]}" "${graph_census_args[@]}" "${lane_env_args[@]}" "${thermal_args[@]}" \
  "${shared_env_args[@]:4}" -e "CUTEAFD_IMAGE=$coordinator_image" "${wip_mount_args[@]}" "${device_map_args[@]}" \
  "${generic_worker_mounts[@]}" -v "$bench_dir:/run/cuteafd/bench" -v "$fidelity_cache:/run/cuteafd/fidelity" -e CUTEAFD_FIDELITY_CACHE=/run/cuteafd/fidelity \
  "${api_mount_args[@]}" "${chat_template_mounts[@]}" "${trace_args[@]}" "${probe_args[@]}" ${profile_args[@]+"${profile_args[@]}"} ${profile_wrapper_args[@]+"${profile_wrapper_args[@]}"} "$coordinator_image" ${profile_wrapper[@]+"${profile_wrapper[@]}"} cuteafd "${memory_policy_args[@]}" $serve --snapshot "$snapshot" --coordinator-weight-budget-gib "$coordinator_weight_budget_gib" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so "${peer_args[@]}" --listen "$addr" \
  --max-sequences "$effective_concurrency" "${context_args[@]}" \
  --max-output-tokens "$(get MAX_OUTPUT_TOKENS 131072)" "${kv_reserve_args[@]}" "${dspark_args[@]}" \
  "${family_args[@]}" "${draft_args[@]}" "${served_args[@]}" >/dev/null
url="$(release_probe_url "$addr")"
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
