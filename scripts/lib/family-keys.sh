#!/usr/bin/env bash
# Per-family configuration-key registries: which KEY=VALUE launch keys each
# launch path actually reads. release/launch/run-family.sh sources this after
# scripts/lib/release-common.sh; scripts/tests/test_options_doc.py sources it
# directly to check docs/options.md against real accept/refuse behaviour.
# shellcheck shell=bash
# A configuration key this family's launch path never reads is a hard error.
# Unknown names already die at parse ("unknown configuration key"). The allowed
# set is static: the keys that path reads (get/key/cfg, or, for DeepSeek V4.1,
# the variables its loader and launch read). Empty values are not keys.
release_reject_unused_keys() {
  local family="$1"
  shift
  local -A allowed=()
  local key
  for key in "$@"; do allowed[$key]=1; done
  # Site build settings share the config file with launch; the launch path ignores them.
  for key in CUTEAFD_EXPORT_HOST WIP_BUILD_SEED; do allowed[$key]=1; done
  local -a unused=()
  for key in "${!cfg_file[@]}"; do
    [[ -n "${allowed[$key]:-}" ]] || unused+=("$key")
  done
  ((${#unused[@]} == 0)) || release_die "unused configuration key: $(printf '%s\n' "${unused[@]}" | sort | paste -sd' ' -)"
}
_spark_slot_keys() {
  local rank suffix
  for rank in 0 1 2 3 4 5 6 7; do
    for suffix in "$@"; do printf 'SPARK_%s_%s\n' "$rank" "$suffix"; done
  done
}
# Keys every non-V4.1 launch path reads, plus the keys only that family reads.
# WIP_INSTANCE and WIP_ROOT are the shared `--wip` slot staging root (the
# start_coordinator block below passes both to every rank's staging script).
_family_allowed_keys() {
  local checkpoint_family="${2:-$1}" registry_dir
  registry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  printf '%s\n' CORE_VIEW
  cat <<'EOF'
VISION
AUDIO
MEDIA_FEATURES_DIR
AUDIO_FEATURES_DIR
GPU_MEMORY_CAP
SPARK_MEMORY_CAP
GPU_RESERVE
SPARK_RESERVE
FAMILY_RESERVE
MEMORY_PADDING
GRAPH_DISABLE_LOW_MEM
GRAPH_NODE_PADDING
KV_TARGET_TOKENS
KV_MIN_TOKENS
KV_MAX_TOKENS
MAPPED_SCALE_PRELOAD_GIB
INSTANCE
MODEL_ID
MODEL_REVISION
SPARK_COUNT
SPARK_HOSTS
EXPERT_BACKEND
COORDINATOR_DOCKER_INFERENCE
COORDINATOR_DOCKER_DEV
WIP_INSTANCE
WIP_ROOT
SPECULATOR
DRAFT_LENGTH
EXPERT_USAGE
EMBEDDING
COORDINATOR_GPUS
CONCURRENCY
DECODE_LANES
DECODE_SHARE
SERVED_MODEL_ID
ADDR
RTX_GPUS
COORDINATOR_SPLIT
POOL_TOKENS
PREFIX_CACHE_ENTRIES
HOST_CACHE_BYTES
MAX_OUTPUT_TOKENS
KV_DECODE_RESERVE_CHUNK
KV_RESERVE_DRAIN_BUFFER
KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS
LOOP_ABORT_RUN
LOOP_ABORT_COPIES
LOOP_FAST_SPAN
LOOP_FAST_COPIES
THERMAL
API_KEY_FILE
ENABLE_BENCH
CONSOLE_ASSETS
USAGE
GATEWAY
EOF
  if [[ "$1" == qwen3_5 ]]; then
    cat <<'EOF'
DECODE_GRAPHS
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
GRAPH_TRACE
MAX_CONTEXT_TOKENS
PREFILL_PACKING
SPECULATOR_DEPTH
SPECULATOR_MODEL_ID
SPECULATOR_MODEL_REVISION
KV_CACHE
FULL_PREFILL_LOGITS
PROBE_DUMP_ROOT
PREFIX_CACHE_MARK_MIB
PREFIX_POINT_BOUNDARIES
SPECULATION_TRACE
PROFILE
PROFILE_DIR
PROFILE_NCU_KERNEL
PROFILE_NCU_COUNT
PROFILE_NCU_SET
PROFILE_NCU_SKIP
PROFILE_NSYS_GPU_METRICS
PROFILE_NSYS_CPU
PROFILE_NSYS_GRAPH_TRACE
EOF
    # Inventory hosts are accepted but never contacted by this RTX-only family.
    _spark_slot_keys HOST LANE_A
    return
  fi
  # Kimi K3, Step-5 and Muse Spark (WIP slots only): the shared serve contract
  # and Spark workers, no media or family modes yet; Step-5 takes an opt-in
  # MTP depth, Muse a DFlash2 drafter and its partial-reply format.
  if [[ "$1" == kimi_k3 || "$1" == step5 || "$1" == muse_spark ]]; then
    cat <<'EOF'
ATTENTION_PLACEMENT
SPARK_EXPERT_DOCKER_INFERENCE
SPARK_EXPERT_DOCKER_DEV
EXPERT_PORT
MAX_CONTEXT_TOKENS
RDMA_BOND_BALANCE
HOST_ISA
PROBE_DUMP_ROOT
TARGET_HEAD
SPARK_INTAKE
SPARK_PREFILL_REPLY_FORMAT
SPARK_DECODE_REPLY_FORMAT
HOST_WAIT
IDLE_PARK_SECONDS
CONSOLE_TEXT
FP8_EXPERT_PREFILL
ATTENTION_PLAN_ROWS
PROFILE
PROFILE_DIR
PROFILE_NCU_KERNEL
PROFILE_NCU_COUNT
PROFILE_NCU_SET
PROFILE_NCU_SKIP
PROFILE_NSYS_GPU_METRICS
PROFILE_NSYS_CPU
PROFILE_NSYS_GRAPH_TRACE
PROFILE_EXPERT_INPUT_SAMPLE
PROFILE_EXPERT_ROUTE_WEIGHTS
PROFILE_SPARK
PROFILE_SPARK_RANKS
PROFILE_SPARK_NCU_KERNEL
PROFILE_SPARK_NCU_COUNT
PROFILE_SPARK_NCU_SET
PROFILE_SPARK_NCU_SKIP
PROFILE_SPARK_NSYS_GPU_METRICS
EOF
    [[ "$1" != kimi_k3 ]] || printf '%s\n' TP2_DFLASH KV_CACHE RTX_EXPERT_LAYERS SPECULATOR_MODEL_ID SPECULATOR_MODEL_REVISION PREFILL_ROWS PREFILL_WAVES PREFILL_PACKING FULL_PREFILL_LOGITS GPU_GRAPH_RESERVE GPU_GRAPH_RESERVE_EXTRA GPU_GRAPH_RESERVE_EXTRA_CAP GRAPH_TRACE
    [[ "$1" != step5 ]] || printf '%s\n' DECODE_GRAPHS GPU_GRAPH_RESERVE GPU_GRAPH_RESERVE_EXTRA GPU_GRAPH_RESERVE_EXTRA_CAP GRAPH_TRACE SPECULATOR_DEPTH KV_CACHE RTX_EXPERT_LAYERS PREFILL_BATCH_TOKENS PREFILL_WAVES FULL_PREFILL_LOGITS PREFILL_PACKING
    [[ "$1" != muse_spark ]] || printf '%s\n' TP2_DFLASH RTX_EXPERT_LAYERS SPECULATOR_MODEL_ID SPECULATOR_MODEL_REVISION SPECULATION_TRACE DRAFT_KV_CACHE PREFILL_ROWS PREFILL_WAVES PREFILL_PACKING FULL_PREFILL_LOGITS EXPERT_PARTIAL_BF16_MAX_ROWS
    [[ "$1" != muse_spark ]] || printf '%s\n' VISION_REPLICAS DECODE_GRAPHS GPU_GRAPH_RESERVE GPU_GRAPH_RESERVE_EXTRA GPU_GRAPH_RESERVE_EXTRA_CAP GRAPH_TRACE
    _spark_slot_keys HOST LANE_A
    return
  fi
  [[ "$1" == mimo_v2 ]] || printf '%s\n' PREFILL_BATCH_TOKENS
  cat <<'EOF'
VISION_REPLICAS
RDMA_BOND_BALANCE
HOST_ISA
ATTENTION_PLACEMENT
CHAT_TEMPLATE_FROM
FULL_PREFILL_LOGITS
EXPERT_INPUT
SPARK_EXPERT_DOCKER_INFERENCE
SPARK_EXPERT_DOCKER_DEV
EXPERT_PORT
MAX_CONTEXT_TOKENS
RTX_EXPERT_LAYERS
EXLR8_MANIFEST
EXLR8_PREFILL_ACTIVATIONS
PROBE_DUMP_ROOT
DRAFT_SURVIVAL_LENGTHS
DRAFT_POLICY
DRAFT_ROW_COST
SPECULATION_TRACE
TARGET_HEAD
SPARK_INTAKE
SPARK_PREFILL_REPLY_FORMAT
SPARK_DECODE_REPLY_FORMAT
HOST_WAIT
IDLE_PARK_SECONDS
CONSOLE_TEXT
FP8_EXPERT_PREFILL
ATTENTION_PLAN_ROWS
PROFILE
PROFILE_DIR
PROFILE_NCU_KERNEL
PROFILE_NCU_COUNT
PROFILE_NCU_SET
PROFILE_NCU_SKIP
PROFILE_NSYS_GPU_METRICS
PROFILE_NSYS_CPU
PROFILE_NSYS_GRAPH_TRACE
PROFILE_EXPERT_INPUT_SAMPLE
PROFILE_EXPERT_ROUTE_WEIGHTS
PROFILE_SPARK
PROFILE_SPARK_RANKS
PROFILE_SPARK_NCU_KERNEL
PROFILE_SPARK_NCU_COUNT
PROFILE_SPARK_NCU_SET
PROFILE_SPARK_NCU_SKIP
PROFILE_SPARK_NSYS_GPU_METRICS
EOF
  _spark_slot_keys HOST LANE_A
  [[ "$1" != mimo_v2 && "$1" != kimi_k3 && "$1" != glm5_flash ]] || printf '%s\n' TP2_DFLASH
  case "$1" in
    deepseek_v4)
      cat <<'EOF'
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
GRAPH_TRACE
TP2_DSPARK_EXPERTS
PREFILL_PACKING
PREFILL_ROWS
PREFILL_WAVES
PREFILL_SPAN_ROWS
RTX_EXPERT_PEER
SPECULATOR_MODEL_ID
SPECULATOR_MODEL_REVISION
EOF
      ;;
    glm5)
      cat <<'EOF'
CUTEAFD_GLM_BF16
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
SPARK_GRAPH_RESERVE
GRAPH_TRACE
EXPERT_USAGE
PREFILL_PACKING
PREFILL_ROWS
PREFILL_WAVES
PREFILL_SPAN_ROWS
RTX_EXPERT_PEER
L2_PREFETCH
FP8_SCALES
SPECULATOR_FP8
DRAFT_CONTEXT_SLOTS
DRAFT_SEQUENCES
DRAFT_KV_CACHE
SPECULATOR_MODEL_ID
SPECULATOR_MODEL_REVISION
EOF
      ;;
    glm5_flash)
      cat <<'EOF'
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
SPARK_GRAPH_RESERVE
GRAPH_TRACE
L2_PREFETCH
FP8_SCALES
SPECULATOR_FP8
DRAFT_CONTEXT_SLOTS
DRAFT_SEQUENCES
DRAFT_KV_CACHE
SPECULATOR_MODEL_ID
SPECULATOR_MODEL_REVISION
MEDIA_CACHE_BYTES
PREFIX_CACHE_MARK_MIB
GLM5_FLASH_KDA_FP8
GLMF_KDA_FP8
GLM5_FLASH_KDA_SPLIT
GLM5_FLASH_INDEX_CACHE
PREFILL_WAVES
PREFILL_ROWS
GLM5_FLASH_FP8_HEAD
GLMF_FP8_HEAD
GLM5_FLASH_DRAFT_LINEAR
GLM5_FLASH_TARGET_HEAD
GLM5_FLASH_FP8_PREFILL
GLMF_FP8_PREFILL
GLM5_FLASH_EXL3_WORKER_PATH
GLM5_FLASH_EXL3_ROUTE_DUMP
GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS
GLM5_FLASH_KDA_STATE
GLM5_FLASH_EXL3_SCHEDULE
GLM5_FLASH_PREFIX_MARKS
GLM5_FLASH_REPLAY_RECORDS
GLM5_FLASH_DECODE_ROWS
GLM5_FLASH_PREFILL_BATCH
PREFILL_PACKING
GLM5_FLASH_ROUTE_RING_CHECK
EOF
      ;;
    mimo_v2)
      cat <<'EOF'
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
SPARK_GRAPH_RESERVE
GRAPH_TRACE
MIMO_WEIGHT_POLICY
MIMO_FP8_HEAD
MIMO_FP8_O_PROJ
SPECULATOR_MODEL_ID
SPECULATOR_MODEL_REVISION
SPECULATOR_DEPTH
MEDIA_CACHE_BYTES
HTTP_QUEUE_DEPTH
HTTP_QUEUE_WAIT_MS
MIMO_PREFIX_DRAFT
MIMO_SNAPSHOT_WAIT
PREFILL_PACKING
MIMO_HOST_CACHE
MIMO_PREFILL_CHUNK_S
PREFILL_ROWS
PREFILL_WAVES
PREFIX_CACHE_MARK_MIB
PREFIX_PARTIAL
KV_CACHE
SWA_KV_CACHE
DECODE_GRAPHS
RTX_EXPERT_PEER
L2_PREFETCH
FP8_SCALES
DRAFT_CONTEXT_SLOTS
DRAFT_SEQUENCES
DRAFT_KV_CACHE
SPECULATOR_FP8
EOF
      ;;
    qwen4)
      cat <<'EOF'
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
SPARK_GRAPH_RESERVE
GRAPH_TRACE
PREFILL_PACKING
KV_CACHE
QWEN_FP8_DECODE
QWEN_FP8_HEAD
PREFILL_ROWS
PREFILL_WAVES
SPECULATOR_DEPTH
SPECULATOR_FP8
MEDIA_CACHE_BYTES
L2_PREFETCH
FP8_SCALES
EOF
      ;;
  esac
}
# Each shared-executor port opts in here after its serve CLI accepts the shape.
family_prefill_mode() {
  case "$1" in
    step5|kimi_k3|muse_spark|qwen4|deepseek_v4|deepseek_v41) echo units ;;
    glm5|glm5_flash|mimo_v2) echo units-span ;;
    *) echo legacy ;;
  esac
}
release_prefill_shape() {
  local deployment="$1" family="$2" mode
  mode="$(family_prefill_mode "$family")"
  PREFILL_ROWS="$(jq -er '.prefill_unit_rows | select(type == "number" and . > 0)' <<<"$deployment")" ||
    release_die "the $family deployment plan names no prefill unit rows"
  PREFILL_WAVES="$(jq -er '.prefill_waves | select(type == "number" and . > 0)' <<<"$deployment")" ||
    release_die "the $family deployment plan names no prefill waves"
  if [[ "$mode" == units-span ]]; then
    PREFILL_SPAN_ROWS="$(jq -er '.prefill_span_rows | select(type == "number" and . > 0)' <<<"$deployment")" ||
      release_die "the $family deployment plan names no prefill span rows"
    echo "Resolved prefill: PREFILL_ROWS=$PREFILL_ROWS PREFILL_WAVES=$PREFILL_WAVES PREFILL_SPAN_ROWS=$PREFILL_SPAN_ROWS"
  elif [[ "$mode" == units ]]; then
    echo "Resolved prefill: PREFILL_ROWS=$PREFILL_ROWS PREFILL_WAVES=$PREFILL_WAVES"
  fi
}
# WIP bypasses artifact readiness, never the planner's deployment guards.
release_validate_deployment() {
  local plan="$1" family="$2"
  if ! jq -e --arg family "$family" --arg wip "$wip_slot" \
    '.version == 1 and .family == $family and .launch.family == $family and .spark_admission == "coordinator_authoritative" and (.executable == true or ($wip != "" and .wip_launchable == true))' <<<"$plan" >/dev/null; then
    jq -r '.report.hints[]?' <<<"$plan" >&2 || true
    release_die "invalid deployment specification"
  fi
}
# Physical ranks can own media without joining the expert transport world.
release_deployment_spark_roles() {
  local deployment="$1" inventory="$2"
  spark_expert_world="$(jq -er '.spark_expert_world | select(type == "number" and . >= 0 and . <= 8 and . == floor)' <<<"$deployment")" ||
    release_die "the deployment plan names no Spark expert world"
  jq -e --argjson inventory "$inventory" '
    .spark_ranks as $n | .spark_expert_world as $world |
    (.spark_roles | length) == $n and
    ([.spark_roles[].rank] | length == (unique | length)) and
    all(.spark_roles[];
      (.rank | type) == "number" and .rank == (.rank | floor) and
      .rank >= 0 and .rank < $inventory and
      (.experts or .encoder or .audio_encoder or .drafter)) and
    ([.spark_roles[] | select(.experts == true) | .rank] == [range($world)]) and
    all(.spark_roles[]; (.experts | type) == "boolean")' <<<"$deployment" >/dev/null ||
    release_die "the deployment plan has inconsistent Spark roles"
  mapfile -t spark_worker_ranks < <(jq -r '.spark_roles[] | select(.experts or .encoder or .audio_encoder or .drafter) | .rank' <<<"$deployment")
  mapfile -t spark_expert_ranks < <(jq -r '.spark_roles[] | select(.experts) | .rank' <<<"$deployment")
  mapfile -t spark_vision_ranks < <(jq -r '.spark_roles[] | select(.encoder) | .rank' <<<"$deployment")
  mapfile -t spark_audio_ranks < <(jq -r '.spark_roles[] | select(.audio_encoder) | .rank' <<<"$deployment")
  jq -e 'all(.spark_roles[]; .drafter == null)' <<<"$deployment" >/dev/null ||
    release_die "unsupported Spark drafter role: no standalone drafter worker is available"
  local rank
  for rank in "${spark_worker_ranks[@]}"; do
    [[ -n "$(get "SPARK_${rank}_HOST")" && -n "$(get "SPARK_${rank}_LANE_A")" ]] ||
      release_die "Spark role rank $rank has no configured host or lane"
    release_spark_role_budget "$deployment" "$rank" >/dev/null || return 1
  done
}
release_spark_role_has_experts() {
  jq -e --argjson rank "$2" '.spark_roles[] | select(.rank == $rank) | .experts == true' <<<"$1" >/dev/null
}
release_spark_role_ready_marker() {
  local deployment="$1" rank="$2"
  if release_spark_role_has_experts "$deployment" "$rank"; then
    echo 'worker bootstrap ready'
  elif jq -e --argjson rank "$rank" '.spark_roles[] | select(.rank == $rank) | .audio_encoder' <<<"$deployment" >/dev/null; then
    echo 'audio encoder ready'
  else
    echo 'vision encoder ready'
  fi
}
release_spark_worker_budget() {
  local deployment="$1" ranks="$2"
  jq -er --argjson ranks "$ranks" '
    [.spark_roles[] | select(.experts) | .rank] as $experts |
    [.memory_layout.devices[] | select(.kind == "spark") |
      select(.index as $index | any($experts[]; . == $index)) | .worker_budget_bytes] |
    select(length == $ranks and all(.[]; type == "number" and . > 0)) |
    unique | select(length == 1) | .[0]' <<<"$deployment" ||
    release_die "the deployment plan names no uniform Spark worker allocation ceiling"
}
release_spark_role_budget() {
  jq -er --argjson rank "$2" '
    [.memory_layout.devices[] | select(.kind == "spark" and .index == $rank) | .worker_budget_bytes] |
    select(length == 1 and all(.[]; type == "number" and . > 0)) | .[0]' <<<"$1" ||
    release_die "the deployment plan names no uniform Spark worker allocation ceiling for rank $2"
}
# DeepSeek V4.1 reads shell variables after release_load_config, not get().
_v41_allowed_keys() {
  cat <<'EOF'
EXPERT_USAGE
ADDR
API_KEY_FILE
ATTENTION_PLACEMENT
ATTENTION_PLAN_ROWS
AUDIO
MEDIA_FEATURES_DIR
AUDIO_FEATURES_DIR
CONCURRENCY
DECODE_LANES
DECODE_SHARE
CONSOLE_ASSETS
CONSOLE_TEXT
COORDINATOR_DOCKER_DEV
COORDINATOR_DOCKER_INFERENCE
COORDINATOR_GPUS
GPU_MEMORY_CAP
SPARK_MEMORY_CAP
GPU_RESERVE
SPARK_RESERVE
FAMILY_RESERVE
MEMORY_PADDING
GRAPH_DISABLE_LOW_MEM
GRAPH_NODE_PADDING
GPU_GRAPH_RESERVE
GPU_GRAPH_RESERVE_EXTRA
GPU_GRAPH_RESERVE_EXTRA_CAP
SPARK_GRAPH_RESERVE
GRAPH_TRACE
KV_TARGET_TOKENS
KV_MIN_TOKENS
KV_MAX_TOKENS
CUDA_MODULE_LOADING
DRAFT_POLICY
DRAFT_ROW_COST
DRAFT_SURVIVAL_LENGTHS
DRAFT_LENGTH
EMBEDDING
ENABLE_BENCH
ENGRAM_WARM
EXPERT_FORMAT
EXPERT_PORT
FP8_EXPERT_PREFILL
GATEWAY
HOST_CACHE_BYTES
HOST_ISA
HTTP_QUEUE_DEPTH
HTTP_QUEUE_WAIT_MS
INSTANCE
KV_POOL_SIZE
KV_POOL_TOKENS
L2_PREFETCH
MAPPED_SCALE_PRELOAD_GIB
MAX_CONTEXT_TOKENS
MAX_OUTPUT_TOKENS
KV_DECODE_RESERVE_CHUNK
KV_RESERVE_DRAIN_BUFFER
KV_EARLY_TERMINATION_MIN_OUTPUT_TOKENS
LOOP_ABORT_RUN
LOOP_ABORT_COPIES
LOOP_FAST_SPAN
LOOP_FAST_COPIES
MODEL_ID
MODEL_REVISION
MODEL_VARIANT
POOL_TOKENS
PREFILL_BATCH_TOKENS
PREFILL_PACKING
PREFILL_WAVES
PREFIX_CACHE_ENTRIES
PROBE_DUMP_ROOT
PROFILE
PROFILE_DIR
PROFILE_EXPERT_INPUT_SAMPLE
PROFILE_EXPERT_ROUTE_WEIGHTS
PROFILE_NCU_COUNT
PROFILE_NCU_KERNEL
PROFILE_NCU_SET
PROFILE_NCU_SKIP
PROFILE_NSYS_GPU_METRICS
PROFILE_NSYS_CPU
PROFILE_NSYS_GRAPH_TRACE
PROFILE_SPARK
PROFILE_SPARK_RANKS
PROFILE_SPARK_NCU_KERNEL
PROFILE_SPARK_NCU_COUNT
PROFILE_SPARK_NCU_SET
PROFILE_SPARK_NCU_SKIP
PROFILE_SPARK_NSYS_GPU_METRICS
RDMA_BOND_BALANCE
RTX_EXPERT_LAYERS
RTX_GPUS
SPARK_COUNT
SPARK_EXPERT_DOCKER_DEV
SPARK_EXPERT_DOCKER_INFERENCE
SPARK_HOSTS
SPARK_REDUCTION_MIN_ROWS
SPARKINFER_EXL3
SPECULATION_TRACE
SPECULATOR
SPARK_INTAKE
SPARK_PREFILL_REPLY_FORMAT
SPARK_DECODE_REPLY_FORMAT
HOST_WAIT
IDLE_PARK_SECONDS
TARGET_HEAD
THERMAL
TP2_ATTENTION
TP2_DSPARK_EXPERTS
TP2_OUTPUT_PROJECTION
TP2_QUERY_PROJECTION
USAGE
V41_FP8_HEAD
VISION
VISION_REPLICAS
WIP_INSTANCE
WIP_ROOT
EOF
  _spark_slot_keys HOST LANE_A LANE_B
}
