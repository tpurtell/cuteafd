#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
if [[ "${1:-}" == --agent ]]; then
  shift
  exec "$repo_root/scripts/launch/agent.sh" "${@:-start}"
fi

usage() {
  cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Starts the checkpoint named by MODEL_ID on the RTX coordinator and configured
Spark ranks. Every family (DeepSeek V4.1, the default, and DeepSeek V4, GLM
5.x, GLM 5.3 Flash, MiMo V2 and Qwen 3.8) is recognized from its config.json
and started by scripts/launch/run-family.sh, which reads the configuration
keys (see cuteafd.config, examples/configs/ and the family docs).
FULL_PREFILL_LOGITS=on (off by default) admits probe-only all-row prefill logits
at launch for these five families (--full-prefill-logits in the daemon). Fidelity
score_path=prefill requires this option; ordinary serving still uses last-row logits.
The CPU planner counts this workspace with plan --layout --full-prefill-logits.
Unspecified Spark count uses all configured hosts, or warns once and selects
its widest executable count. SPARK_COUNT or SPARK_HOSTS is exact: the planner
honors it or refuses with a reason and the counts that work, on every card size.
An optional expert-group topology is selected in the configuration with
SPARK_TP and SPARK_EP (all-or-none, native checkpoint only); at SPARK_COUNT=3
the explicit SPARK_TP=3 SPARK_EP=1 form is one unreplicated three-rank group.
See docs/tp-ep-configuration.md.
INSTANCE (configuration key, empty by default) names a launch that runs beside
others on disjoint hardware: its coordinator container is
cuteafd-coordinator-INSTANCE instead of the one shared cuteafd-coordinator, and
./stop.sh with the same configuration stops only that one. Accepts Docker's
container-name characters, [A-Za-z0-9_.-].

  --agent [start|stop|status]    independent experimental workspace sidecar
  --config FILE                 alternate complete configuration
  --set KEY=VALUE               override one configuration key for this launch
                                (repeatable; e.g. --set RTX_GPUS=2,
                                --set CONCURRENCY=8, --set DSPARK_DRAFT_LIMIT=7,
                                --set TP2_ATTENTION=on, --set ADDR=0.0.0.0:8001)
  --embedding-placement host|gpu  token embedding residency (EMBEDDING)
  --wip SLOT                     serve a ./wip.sh slot's artifacts in the dev images
  --restart                     replace this instance's deployment
  --all                         with --restart, sweep worker ports on selected hosts
  --dry-run                     validate a DeepSeek V4.1 launch without changing services

DeepSeek V4.1 keys (cuteafd.config documents each): RTX_GPUS, RTX_EXPERT_LAYERS,
CONCURRENCY, HTTP_QUEUE_DEPTH, HTTP_QUEUE_WAIT_MS, POOL_TOKENS, EMBEDDING,
KV_POOL_SIZE, HOST_CACHE_BYTES, MEMORY_RESERVATION, PREFIX_CACHE_ENTRIES,
MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, PREFILL_BATCH_TOKENS, DSPARK,
DSPARK_DRAFT_LIMIT (1..7), TP2_ATTENTION, TP2_QUERY_PROJECTION,
TP2_OUTPUT_PROJECTION, TP2_DSPARK_EXPERTS. The former command-line spellings
(--listen, --rtx-gpus, --concurrency, --kv-pool-size, --memory-reservation,
--prefix-cache-entries, --dspark-draft-limit, --tp2-*, ...) are these keys.

Optional RDMA tuning env values are forwarded to both roles only when set:
  CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP (local-ip=device,...),
  CUTEAFD_VERBS_APP_IB_PORT_NUM, CUTEAFD_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES.
This is how a multi-homed six-rank launch pins the rail per host.

Every remote step shares one SSH option set with the release build scripts:
  CUTEAFD_RELEASE_SSH_CONFIG       ssh config file to use (default empty: stock
                                  OpenSSH resolution; BatchMode is always forced so
                                  a poll or cleanup can never wait on a prompt).
                                  /dev/null discards a broken system include but
                                  also drops your ~/.ssh/config host aliases, so
                                  prefer a file containing 'Include ~/.ssh/config'.
EOF
}

config="$repo_root/cuteafd.config"
family_args=()
# The former DeepSeek V4.1 options, as the configuration keys they set.
legacy_key() {
  case "$1" in
    --listen) printf ADDR ;; --rtx-gpus) printf RTX_GPUS ;; --http-queue-depth) printf HTTP_QUEUE_DEPTH ;;
    --http-queue-wait-ms) printf HTTP_QUEUE_WAIT_MS ;; --rtx-expert-layers) printf RTX_EXPERT_LAYERS ;;
    --concurrency) printf CONCURRENCY ;; --host-cache-bytes) printf HOST_CACHE_BYTES ;;
    --pool-tokens) printf POOL_TOKENS ;; --kv-pool-size) printf KV_POOL_SIZE ;;
    --memory-reservation) printf MEMORY_RESERVATION ;; --prefix-cache-entries) printf PREFIX_CACHE_ENTRIES ;;
    --max-context-tokens) printf MAX_CONTEXT_TOKENS ;; --max-output-tokens) printf MAX_OUTPUT_TOKENS ;;
    --prefill-batch-tokens) printf PREFILL_BATCH_TOKENS ;; --dspark-draft-limit) printf DSPARK_DRAFT_LIMIT ;;
    *) return 1 ;;
  esac
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?$1 requires FILE}"; shift 2 ;;
    --set) family_args+=(--set "${2:?$1 requires KEY=VALUE}"); shift 2 ;;
    --embedding-placement) family_args+=(--embedding-placement "${2:?$1 requires host or gpu}"); shift 2 ;;
    --wip) family_args+=(--wip "${2:?$1 requires SLOT}"); shift 2 ;;
    --rotate-console-secret) exec "$repo_root/scripts/launch/console-secret.sh" rotate ;;
    --restart|--all|--dry-run) family_args+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    --dspark|--no-dspark|--tp2-*|--no-tp2-*)
      release_die "$1 was removed: set the configuration key instead (--set DSPARK=on|off, --set TP2_<OPTION>=on|off)" ;;
    *)
      legacy="$(legacy_key "$1")" || release_die "unknown run argument: $1"
      release_die "$1 was removed: use --set $legacy=VALUE (or $legacy in the configuration)" ;;
  esac
done

# Every family launches through scripts/launch/run-family.sh (same config file,
# same container names, so ./stop.sh stops it); the family comes from the
# checkpoint's config.json.
family="$(release_config_family "$config")"
exec "$repo_root/scripts/launch/run-family.sh" --config "$config" --family "$family" ${family_args[@]+"${family_args[@]}"}
