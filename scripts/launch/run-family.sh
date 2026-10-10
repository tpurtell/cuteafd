#!/usr/bin/env bash
# Launch a DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 or Qwen 3.8 Flash Next
# checkpoint (the family's serve command on one RTX, routed experts on the first
# SPARK_COUNT Sparks, or SPARK_HOSTS in explicit rank order) from the release images named in the config. ./run.sh
# starts this for every family but DeepSeek V4.1; the family comes from the
# snapshot's config.json (scripts/lib/checkpoint-family.py), or --family.
# Containers use run.sh's names, so ./stop.sh stops them.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
config="$repo_root/cuteafd.config"
restart=0
restart_all=0
family=""
embedding_override=""
table_override=""
wip_slot=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?--config requires FILE}"; shift 2 ;;
    --family) family="${2:?--family requires ID}"; shift 2 ;;
    --table-backend) table_override="${2:?--table-backend requires uring, mmap or mincore-routed}"; shift 2 ;;
    --embedding-placement) embedding_override="${2:?--embedding-placement requires host or gpu}"; shift 2 ;;
    --restart) restart=1; shift ;;
    --all) restart_all=1; shift ;;
    --wip) wip_slot="${2:?--wip requires SLOT}"; shift 2 ;;
    *) echo "usage: $0 [--config FILE] [--family ID] [--embedding-placement host|gpu] [--table-backend uring|mmap|mincore-routed] [--restart [--all]] [--wip SLOT]" >&2; exit 2 ;;
  esac
done
((restart_all == 0 || restart == 1)) || release_die "--all requires --restart"
# Plain KEY=VALUE lines; the launch reads only the keys below.
declare -A cfg
while IFS='=' read -r key value; do
  release_known_key "$key" || release_die "unknown configuration key: $key"
  cfg[$key]="$value"
done < <(grep -E '^[A-Z_0-9]+=' "$config")
[[ -z "$embedding_override" ]] || cfg[EMBEDDING]="$embedding_override"
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
table_backend="${table_override:-$(get TABLE_BACKEND "${CUTEAFD_TABLE_BACKEND:-mmap}")}"
release_validate_table_backend "$table_backend"
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
# Validate the name before it is used to identify allocations during admission.
instance="$(get INSTANCE)"
[[ -z "$instance" || "$instance" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] || { echo "INSTANCE must be [A-Za-z0-9_.-]" >&2; exit 2; }
coordinator_name="cuteafd-coordinator${instance:+-$instance}"
# key NEW OLD [DEFAULT]: a renamed key; the pre-rename spelling works for one
# release with a warning.
key() {
  if [[ -n "${cfg[$1]:-}" ]]; then printf '%s' "${cfg[$1]}"
  elif [[ -n "${cfg[$2]:-}" ]]; then echo "warning: config key $2 is deprecated; use $1" >&2; printf '%s' "${cfg[$2]}"
  else printf '%s' "${3:-}"; fi
}
model="$(get MODEL_ID)"
revision="$(get MODEL_REVISION)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
hub="$(readlink -f "$hf_home/hub")"
root="$hub/models--${model//\//--}"
[[ -n "$revision" ]] || revision="$(<"$root/refs/main")"
snapshot="/root/.cache/huggingface/hub/models--${model//\//--}/snapshots/$revision"
[[ -d "$root/snapshots/$revision" ]] || { echo "missing snapshot $model@$revision" >&2; exit 1; }
# Family, the serving command and the layers with routed experts.
described="$(python3 "$repo_root/scripts/lib/checkpoint-family.py" "$root/snapshots/$revision/config.json")" || exit 2
read -r detected model_type first_layer last_layer <<<"$described"
[[ -z "$family" || "$family" == "$detected" ]] ||
  { echo "--family $family does not match the checkpoint ($detected, model_type $model_type)" >&2; exit 2; }
family="$detected"
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
audio="$(release_resolve_audio_mode "$audio" "$root/snapshots/$revision")"
# Auto/spark placement is resolved by the encoder plan below.
# EXPERT_BACKEND=auto prefers qualified local experts when the planner admits
# their weights plus serving reservations on the selected GPU. SPARK_COUNT is
# the fallback topology; EXPERT_BACKEND=spark explicitly keeps it.
ranks="$(get SPARK_COUNT 4)"
configured_ranks="$ranks"
if [[ -n "$(get SPARK_HOSTS)" ]]; then
  host_rows="$(release_spark_host_rows "$(get SPARK_HOSTS)" "$ranks")" || exit 2
  while read -r rank host lane_a lane_b; do
    cfg[SPARK_${rank}_HOST]="$host"
    cfg[SPARK_${rank}_LANE_A]="$lane_a"
    cfg[SPARK_${rank}_LANE_B]="$lane_b"
  done <<<"$host_rows"
fi
backend="$(get EXPERT_BACKEND auto)"
case "$backend" in
  auto|spark) ;;
  local) ranks=0 ;;
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
  wip_coordinator_container="$(release_wip_container coordinator)"
  wip_spark_container="$(release_wip_container spark-expert)"
  coordinator_image="$(get COORDINATOR_DOCKER_DEV cuteafd-coordinator-dev)"
  release_ensure_dev_image "$coordinator_image"
  wip_layout="$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT"
  release_stage_wip_layout "$wip_coordinator_container" "$wip_slot" coordinator "$wip_layout"
  wip_mount_args=(-v "$wip_layout/bin:/opt/cuteafd/bin:ro" -v "$wip_layout/lib:/opt/cuteafd/lib:ro"
    -v "$wip_layout/share:/opt/cuteafd/share:ro"
    -v "$wip_layout/source:/source:ro" -e PYTHONPATH=/source/third_party/sparkinfer
    -e HOME=/tmp/cuteafd-home -e HF_HOME=/root/.cache/huggingface -e USER=tj -e LOGNAME=tj
    -e TORCH_EXTENSIONS_DIR=/tmp/cuteafd-home/torch-extensions
    -e XDG_CACHE_HOME=/tmp/cuteafd-home/.cache -e TRITON_CACHE_DIR=/tmp/cuteafd-home/triton
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/cuteafd-home/torchinductor
    -e "PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so
    --entrypoint /opt/cuteafd/share/release-entrypoint.sh)
  wip_worker_args="-v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/bin:/opt/cuteafd/bin:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/lib:/opt/cuteafd/lib:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/share:/opt/cuteafd/share:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$WIP_LAYOUT_SLOT/source:/source:ro \
    -e PYTHONPATH=/source/third_party/sparkinfer -e HOME=/tmp/cuteafd-home -e HF_HOME=/root/.cache/huggingface -e USER=tj -e LOGNAME=tj \
    -e TORCH_EXTENSIONS_DIR=/tmp/cuteafd-home/torch-extensions -e XDG_CACHE_HOME=/tmp/cuteafd-home/.cache \
    -e TRITON_CACHE_DIR=/tmp/cuteafd-home/triton -e TORCHINDUCTOR_CACHE_DIR=/tmp/cuteafd-home/torchinductor \
    -e PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so --entrypoint /opt/cuteafd/share/release-entrypoint.sh"
fi
qwen_exl3=0
qwen_nvfp4=0
qwen_mtp=0
if [[ "$family" == qwen4 ]]; then
  # NVFP4: a ModelOpt publication with 4-bit weight groups (nvidia/Qwen3.8-Flash-Next-NVFP4).
  qwen_features="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("quantization_config", {}); m=q.get("quant_method", q.get("method")); g=q.get("config_groups", {}).values(); print(int(m == "exl3"), int(m == "modelopt" and any(x.get("weights", {}).get("num_bits") == 4 for x in g)), c.get("text_config", c).get("mtp_num_hidden_layers", 0))' "$root/snapshots/$revision/config.json")"
  read -r qwen_exl3 qwen_nvfp4 qwen_mtp <<<"$qwen_features"
fi
if [[ "$qwen_exl3" == 1 && "$backend" == auto && "$ranks" != 0 ]]; then
  selected="$(get COORDINATOR_GPUS "$(get COORDINATOR_GPU 0)")"; selected="${selected%%,*}"
  free_mib="$(nvidia-smi --id="$selected" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
    if [[ "$restart" == 1 ]]; then
      # --restart will release this container's allocations after validation.
      # Credit only its host PIDs on this GPU, never another launch's memory.
      own_pids="$(docker top "$coordinator_name" -eo pid 2>/dev/null | tail -n +2 || true)"
      if [[ -n "$own_pids" ]]; then
        own_mib="$(nvidia-smi --id="$selected" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
          | python3 -c 'import csv,sys; p=set(sys.argv[1].split()); print(sum(int(r[1].strip()) for r in csv.reader(sys.stdin) if len(r)==2 and r[0].strip() in p and r[1].strip().isdigit()))' "$own_pids" || true)"
        [[ "$own_mib" =~ ^[0-9]+$ ]] && free_mib=$((free_mib + own_mib))
      fi
    fi
    free_gib="$(python3 -c 'import sys; print(int(sys.argv[1])/1024)' "$free_mib")"
    if [[ -n "$coordinator_budget" ]]; then
      total_mib="$(nvidia-smi --id="$selected" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
      # Credit this launch's restart above, but charge other physical usage
      # against the simulated smaller card, just as runtime admission does.
      if [[ "$total_mib" =~ ^[0-9]+$ ]]; then
        free_gib="$(python3 -c 'import sys; free,total,budget=map(float,sys.argv[1:]); print(max(0,min(free,budget-total+free)))' \
          "$free_gib" "$(python3 -c 'import sys; print(int(sys.argv[1])/1024)' "$total_mib")" "$coordinator_budget")"
      else
        free_gib=0 # No trustworthy sample: keep the Spark fallback.
      fi
    fi
    pool="$(get POOL_TOKENS 32768)"
    if [[ "$pool" =~ ^[1-9][0-9]*$ ]]; then
      # CPU-only preflight reads checkpoint headers in the selected serving image.
      # Older images that do not qualify auto placement keep the Spark fallback.
      preferred="$(docker run --rm --network none -v "$hub:/root/.cache/huggingface/hub:ro" "${wip_mount_args[@]}" \
        "$coordinator_image" cuteafd plan "$snapshot" --vision "$vision" --audio "$audio" --json --layout \
        --rtx 1 --coordinator-gpu-budget-gib "$free_gib" --coordinator-weight-budget-gib "$free_gib" --pool-tokens "$pool" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["spark_ranks"])' 2>/dev/null || true)"
      if [[ "$preferred" == 0 ]]; then
        echo "note: Qwen EXL3 auto selected resident local experts on GPU $selected; EXPERT_BACKEND=spark forces Spark ranks" >&2
        ranks=0
      fi
    fi
  fi
fi
layer_args="--first-layer $first_layer"
[[ "$last_layer" == -1 ]] || layer_args+=" --last-layer $last_layer"
# Docker options only the Spark expert workers take (their environment and mounts).
spark_worker_env=""
# Options only the Spark expert workers take.
spark_worker_args=""
# Snapshot of a model id (and optional revision) inside the containers.
snapshot_of() {
  local id="$1" rev="$2" dir="$hub/models--${1//\//--}"
  [[ -n "$rev" ]] || rev="$(<"$dir/refs/main")"
  [[ -d "$dir/snapshots/$rev" ]] || { echo "missing snapshot $id@$rev" >&2; return 1; }
  printf '%s' "/root/.cache/huggingface/hub/models--${id//\//--}/snapshots/$rev"
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
chat_template_from="$(get CHAT_TEMPLATE_FROM)"
if [[ -n "$chat_template_from" && "$vision" != off ]]; then
  [[ "$family" == glm5_flash ]] || release_die "CHAT_TEMPLATE_FROM currently applies only to GLM Flash vision"
  if [[ -d "$chat_template_from" ]]; then
    chat_template_from="$(readlink -f "$chat_template_from")"
    release_validate_path_setting CHAT_TEMPLATE_FROM "$chat_template_from"
    if release_path_within "$chat_template_from" "$hub"; then
      chat_template_from="/root/.cache/huggingface/hub${chat_template_from#"$hub"}"
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
  [[ "$(get MIMO_COPY_WINDOWS off)" != on ]] || family_args+=(--mimo-copy-windows)
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
# COPY_DRAFTS=off: decode without copy-window drafts (serve-glm, serve-glmf, serve-mimo, serve-qwen4).
if [[ "$(get COPY_DRAFTS on)" == off ]]; then
  [[ $serve != serve-dsv4 ]] || { echo "COPY_DRAFTS applies to GLM, MiMo and Qwen checkpoints" >&2; exit 2; }
  family_args+=(--no-copy-drafts)
fi
# DECODE_SHARE: the share of the time running requests keep while prompts
# prefill (the engine's default 0.2; 0 prefills whole prompts before the next
# step). Keys left unset pass nothing (images older than the options run).
[[ -z "$(get DECODE_SHARE)" ]] || family_args+=(--decode-share "$(get DECODE_SHARE)")
# DeepSeek V4 keeps complete expert layers local while memory permits. Honor
# an explicit limit; zero leaves the backbone experts on the Sparks.
if [[ $serve == serve-dsv4 ]]; then
  local_layers="$(get RTX_EXPERT_LAYERS auto)"
  case "$local_layers" in
    ""|auto) ;;
    *[!0-9]*) echo "RTX_EXPERT_LAYERS must be auto or a nonnegative integer" >&2; exit 2 ;;
    *) family_args+=(--local-expert-layers "$local_layers") ;;
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
# PROBE_DUMP_ROOT=/abs/host/dir: remote benchmark probes (POST /v1/bench/probe; `cuteafd bench
# fidelity run --dump-dir`, scripts/bench/glmf-teacher-kl.py) may stream full-vocabulary rows
# (dump_rows) into new leaves under it (CUTEAFD_PROBE_DUMP_ROOT, mounted at the same path); unset,
# remote probes cannot write rows.
probe_args=()
probe_root="$(get PROBE_DUMP_ROOT)"
if [[ -n "$probe_root" ]]; then
  [[ "$probe_root" == /* && -d "$probe_root" ]] || { echo "PROBE_DUMP_ROOT must be an existing absolute directory" >&2; exit 2; }
  probe_args=(-v "$probe_root:$probe_root" -e "CUTEAFD_PROBE_DUMP_ROOT=$probe_root")
fi
# SPECULATION_TRACE=/abs/host/file.jsonl: the per-cycle speculation trace
# (CUTEAFD_SPECULATION_TRACE, written by serve-glm and serve-qwen4; read by
# scripts/qualify/glm5/glm-draft-trace.py and qualify/qwen4/qwen4-draft-trace.py).
# COORDINATOR_TRACE is its pre-rename key; images older than the rename read
# the family variables, which are set too for one release.
trace_args=()
# Experimental shared policy changes stay off until emitted-throughput gates pass.
for policy_key in DRAFT_COST_BUCKETS DRAFT_CONFIDENCE COPY_DRAFT_POLICY; do
  case "$(get "$policy_key" off)" in
    on) trace_args+=(-e "CUTEAFD_$policy_key=1") ;;
    off) trace_args+=(-e "CUTEAFD_$policy_key=0") ;;
    *) echo "$policy_key must be on or off" >&2; exit 2 ;;
  esac
done
trace="$(key SPECULATION_TRACE COORDINATOR_TRACE)"
if [[ -n "$trace" ]]; then
  mkdir -p "$(dirname "$trace")"
  trace_args+=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_SPECULATION_TRACE=$trace"
    -e "CUTEAFD_GLM_TRACE=$trace" -e "CUTEAFD_QWEN4_TRACE=$trace")
fi
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
  qwen4:*) split_hint="add Qwen head-split GDN/GQA/shared-expert kernels and sharded recurrent/KV state" ;;
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
gpus="device=$gpu"
head_split=0
if [[ -n "$second" && "$split" != off ]]; then
  [[ "$second" =~ ^[0-9]+$ ]] || { echo "COORDINATOR_SPLIT_GPU must be a GPU index" >&2; exit 2; }
  [[ "$second" != "$gpu" ]] || { echo "the second coordinator GPU must differ from the first" >&2; exit 2; }
  if [[ -z "$split_hint" ]]; then
    for selected in "$gpu" "$second"; do
      grep -qx "$selected" <<<"$physical_gpus" ||
        { echo "two-GPU head split requires physical GPU $selected, but nvidia-smi did not report it" >&2; exit 2; }
    done
    lower=$((gpu < second ? gpu : second)) upper=$((gpu < second ? second : gpu))
    gpus="\"device=$lower,$upper\""
    head_split=1
    family_args+=(--device $((gpu == lower ? 0 : 1)) --split-device $((gpu == lower ? 1 : 0)))
  elif [[ "$explicit_split" != 1 ]]; then
    echo "note: $family ($model_type) has no head split; auto selected GPU $gpu alone" >&2
  fi
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
  # GLM5_FLASH_DRAFT_HEAD: tensor by default with a drafter; exact runs as the
  # target's own head, FP32 products and sums on CUDA cores past 24 rows) or tensor (from two draft
  # blocks of 8 rows, a BF16 tensor-core GEMM with FP32 accumulation that reads the head once).
  # Drafts only: the target verifies every proposal through its own head.
  draft_head_default=exact
  [[ ${#draft_args[@]} == 0 ]] || draft_head_default=tensor
  draft_head="$(get GLM5_FLASH_DRAFT_HEAD "$draft_head_default")"
  case "$draft_head" in
    ""|exact) ;;
    tensor) family_args+=(--draft-head tensor) ;;
    *) echo "GLM5_FLASH_DRAFT_HEAD must be exact or tensor" >&2; exit 2 ;;
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
  spec_tau="$(get GLM5_FLASH_SPEC_TAU)"
  if [[ -n "$spec_tau" ]]; then
    [[ "$spec_tau" =~ ^(0?[.][0-9]*[1-9][0-9]*|1([.]0*)?)$ ]] || { echo "GLM5_FLASH_SPEC_TAU must be in (0, 1]" >&2; exit 2; }
    family_args+=(--spec-tau "$spec_tau")
  fi
fi
# INSTANCE names a launch that runs beside others on disjoint hardware
# (`cuteafd bench smoke` sets it): its coordinator container is
# cuteafd-coordinator-INSTANCE; empty keeps the one cuteafd-coordinator.
# SPARK_COUNT=0: the routed experts run on the coordinator GPU (--local-experts;
# GLM 5.3 Flash, MiMo V2 and Qwen 3.8), the natural minimum for checkpoints
# that fit one RTX.
if [[ "$ranks" == 0 ]]; then
  case "$family" in
    glm5_flash|mimo_v2|qwen4) family_args+=(--local-experts) ;;
    *) echo "SPARK_COUNT=0 (local experts) serves GLM 5.3 Flash, MiMo V2 and Qwen 3.8, not $family" >&2; exit 2 ;;
  esac
fi
# Check every selected image before --restart or checkpoint reads. The worker's
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
# Resolve cold placement before starting or removing containers. Off skips the
# planner and tower startup entirely; text-only families keep their old path.
vision_peers=()
encoder_ranks=()
encoder_hash=""
encoder_max_tokens=4096
encoder_port=$((port + 1))
audio_encoder_port=$((port + 2))
audio_encoder_ranks=()
audio_peers=()
audio_encoder_hash=""
if { [[ ( "$family" == mimo_v2 || "$family" == qwen4 || "$family" == glm5_flash ) && "$vision" != off ]] &&
   python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("vision_config") else 1)' "$root/snapshots/$revision/config.json"; } ||
   [[ "$family" == mimo_v2 && "$audio" != off ]]; then
  plan_rtx=1; ((head_split == 0)) || plan_rtx=2
  plan_pool="$(get POOL_TOKENS auto)"; [[ "$plan_pool" != auto ]] || plan_pool=0
  plan_gib="${coordinator_budget:-95.5}"
  plan_draft_args=()
  if [[ "$family" == mimo_v2 ]]; then
    plan_draft_args+=(--concurrency "$(get CONCURRENCY "$default_concurrency")"
      --prefix-cache-entries "$(get PREFIX_CACHE_ENTRIES 20)")
    [[ -z "$(get PREFIX_CACHE_MARK_MIB)" ]] || plan_draft_args+=(--prefix-cache-mark-mib "$(get PREFIX_CACHE_MARK_MIB)")
    [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || plan_draft_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
    [[ -z "$(get DRAFT_SEQUENCES)" ]] || plan_draft_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    if [[ "$(get MIMO_PREFIX_DRAFT off)" == on ]]; then
      plan_draft_args+=(--mimo-prefix-draft --context-tokens "$(get MAX_CONTEXT_TOKENS 131072)")
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
  plan_json="$(docker run --rm --network none -v "$hub:/root/.cache/huggingface/hub:ro" "${wip_mount_args[@]}" \
    "$coordinator_image" cuteafd plan "$snapshot" --vision "$vision" --audio "$audio" --json --layout \
    --spark-ranks "$ranks" --spark-budget-gib "$(python3 -c 'import sys;print(int(sys.argv[1])/2**30)' "$budget")" \
    --rtx "$plan_rtx" --coordinator-gpu-budget-gib "$plan_gib" --pool-tokens "$plan_pool" --vision-replicas "$vision_replicas" "${plan_draft_args[@]}")"
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
# FP8_EXPERT_PREFILL: how FP8 expert packages run prefill row counts: auto
# (default: wire rows W8A8 with E4M3 x E4M3 gate/up, BF16 rows W8A16), w8a16
# (the former programs) or w8a8 (also quantizes the BF16 rows of experts on the
# coordinator GPU). Spark workers and the coordinator both read it.
fp8_prefill="$(get FP8_EXPERT_PREFILL auto)"
case "$fp8_prefill" in auto|w8a8|w8a16) ;; *) echo "FP8_EXPERT_PREFILL must be auto, w8a8 or w8a16" >&2; exit 2 ;; esac
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
    ssh "$host" bash -s -- "$wip_slot" "$spark_image" "$pinned_sparkinfer" "$wip_spark_container" "$WIP_LAYOUT_SLOT" <<'STAGE' ||
set -euo pipefail
slot="$1" image="$2" pinned="$3"
container="$4" layout_slot="$5"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  registry="${image%%/*}"
  [[ "$image" == */* && ( "$registry" == *.* || "$registry" == *:* || "$registry" == localhost ) ]] || { echo "missing local dev image $image" >&2; exit 1; }
  docker pull "$image"
fi
layout="$HOME/.cache/cuteafd/wip-run/$layout_slot" raw="$HOME/.cache/cuteafd/wip-run/$layout_slot.tmp/raw"
rm -rf "$layout.tmp" && mkdir -p "$raw" "$layout.tmp/bin" "$layout.tmp/lib" "$layout.tmp/share"
docker cp "$container:/wip/slots/$slot/spark-expert/workspace/.cuteafd-wip/." "$raw/"
docker cp "$container:/wip/slots/$slot/spark-expert/workspace/docker/release-entrypoint.sh" "$raw/"
mv "$raw/cuteafd" "$layout.tmp/bin/cuteafd"
mv "$raw/libcuteafd_native.so" "$layout.tmp/lib/"
[[ ! -d "$raw/exl3" ]] || mv "$raw/exl3" "$layout.tmp/lib/exl3"
[[ ! -d "$raw/fp8" ]] || mv "$raw/fp8" "$layout.tmp/lib/fp8"
mv "$raw/"* "$layout.tmp/share/"
mkdir -p "$layout.tmp/source/third_party" "$layout.tmp/source/scripts/build"
slot_source="$container:/wip/slots/$slot/spark-expert/workspace"
docker cp "$slot_source/third_party/sparkinfer" "$layout.tmp/source/third_party/"
docker cp "$slot_source/third_party/sparkinfer.lock.json" "$layout.tmp/source/third_party/"
docker cp "$slot_source/scripts/build/verify-sparkinfer-source.py" "$layout.tmp/source/scripts/build/"
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
  ssh "$host" "docker run -d --name cuteafd-spark-expert-$host-$port --restart no --gpus all --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill$spark_worker_env $wip_worker_args $device_map_env \
    -v \$(readlink -f \$HOME/.cache/huggingface/hub):/root/.cache/huggingface/hub:ro '$spark_image' \
    cuteafd expertd-native --snapshot '$snapshot' --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    --rank $rank --world $ranks --capacity 4096 --device-budget-bytes $budget $layer_args$spark_worker_args \
    --listen 0.0.0.0:$port $encoder_args >/dev/null" &
done
wait
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  # Expert readiness follows synchronous encoder startup in the same process.
  ready_deadline=$((SECONDS + 900))
  until ssh "$host" "docker logs cuteafd-spark-expert-$host-$port 2>&1 | grep -q 'worker ready'"; do
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
# The in-server benchmark keeps its history (SQLite) on the host; the image
# name labels its reports.
bench_dir="$HOME/.cache/cuteafd/bench"
mkdir -p "$bench_dir"
# SPARK_INTAKE: how routed partials reach the coordinator GPU (auto, gpu, pinned
# or host; see rust/crates/cuteafd-daemon/src/shared/spark_intake.rs).
intake="$(get SPARK_INTAKE auto)"
case "$intake" in auto|gpu|pinned|host) ;; *) echo "SPARK_INTAKE must be auto, gpu, pinned or host" >&2; exit 2 ;; esac
# CONSOLE_TEXT=on lets the live console at / stream generated token text (anyone who
# can reach the API port can then read every session's output).
family_args+=(--table-backend "$table_backend")
console_text="$(get CONSOLE_TEXT off)"
case "$console_text" in on|off) ;; *) echo "CONSOLE_TEXT must be on or off" >&2; exit 2 ;; esac
api_mount_args=()
API_KEY_FILE="$(get API_KEY_FILE "${API_KEY_FILE:-}")"
ENABLE_BENCH="$(get ENABLE_BENCH off)"
release_prepare_api_key "$ENABLE_BENCH" "${instance:-default}"
usage="$(get USAGE off)"
case "$usage" in on|off) ;; *) echo "USAGE must be on or off" >&2; exit 2 ;; esac
console_supported=0
if release_console_supported "$coordinator_image" "${wip_layout:+$wip_layout/bin/cuteafd}"; then
  console_supported=1
  release_prepare_console "${instance:-default}"
  api_mount_args+=(--mount "type=bind,src=$CONSOLE_SECRET_FILE,dst=/run/cuteafd-console-secret,readonly" -v "$USAGE_DIR:/root/.cache/cuteafd/usage")
  family_args+=(--console-secret-file /run/cuteafd-console-secret --usage-dir /root/.cache/cuteafd/usage --usage "$usage")
fi
if [[ -n "$API_KEY_FILE" ]]; then
  [[ -f "$API_KEY_FILE" && -r "$API_KEY_FILE" ]] || { echo "API_KEY_FILE must name a readable file" >&2; exit 2; }
  API_KEY_FILE="$(readlink -f "$API_KEY_FILE")"
  api_mount_args+=(--mount "type=bind,src=$API_KEY_FILE,dst=/run/cuteafd-api-key,readonly")
  family_args+=(--api-key-file /run/cuteafd-api-key)
fi
case "$ENABLE_BENCH" in
  on) family_args+=(--enable-bench) ;;
  off) ;;
  *) echo "ENABLE_BENCH must be on or off" >&2; exit 2 ;;
esac
table_env_args=()
[[ -z "${CUTEAFD_TABLE_ACCOUNTING:-}" ]] || table_env_args+=(-e "CUTEAFD_TABLE_ACCOUNTING=$CUTEAFD_TABLE_ACCOUNTING")
bench_nonce_env_args=()
[[ -z "${CUTEAFD_BENCH_NONCE_SEED:-}" ]] || bench_nonce_env_args+=(-e "CUTEAFD_BENCH_NONCE_SEED=$CUTEAFD_BENCH_NONCE_SEED")
docker run -d --name "$coordinator_name" --restart no --gpus "$gpus" --network host --ipc host \
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e "CUTEAFD_SPARK_INTAKE=$intake" \
  -e "CUTEAFD_CONSOLE_TEXT=$([[ $console_text == on ]] && echo true || echo false)" "${bond_args[@]}" "${table_env_args[@]}" "${bench_nonce_env_args[@]}" \
  -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill" -e "CUTEAFD_IMAGE=$coordinator_image" "${wip_mount_args[@]}" "${device_map_args[@]}" \
  -v "$hub:/root/.cache/huggingface/hub:ro" -v "$bench_dir:/root/.cache/cuteafd/bench" \
  "${api_mount_args[@]}" "${chat_template_mounts[@]}" "${trace_args[@]}" "${probe_args[@]}" "$coordinator_image" cuteafd "${coordinator_budget_args[@]}" $serve --snapshot "$snapshot" \
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
