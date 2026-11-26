#!/usr/bin/env bash
# ATTENTION_PLACEMENT validation shared by release/launch/run-family.sh.
# scripts/tests/test_options_doc.py drives it for every family x value cell of
# docs/options.md, so the documented accept/refuse table is the launcher's own.

# Dies (exit 2) when this family cannot run the requested explicit placement.
# `auto` is always accepted: the planner and server resolve it to the family's
# recorded default.
release_check_attention_placement() {
  local attention_placement="$1" family="$2"
  if [[ "$family" == deepseek_v41 ]]; then
    # V4.1 serves its established ranges until its runtime consumes solver placement.
    case "$attention_placement" in
      auto|heads) ;;
      context|layers) release_die "deepseek_v41 cannot run attention placement $attention_placement: runtime solver placement not implemented; existing ranges stay unchanged" ;;
      *) release_die "ATTENTION_PLACEMENT must be auto, context, layers or heads" ;;
    esac
    return 0
  fi
  # auto is the family's recorded default (GLM 5.3 and GLM 5.3 Flash: context where the image
  # carries its programs); the planner and the server resolve it. Explicit modes run only where an
  # executor exists.
  case "$attention_placement:$family" in
    auto:kimi_k3|heads:kimi_k3|context:kimi_k3|layers:kimi_k3) ;;
    # These executors have one concrete attention ownership mode.
    heads:qwen3_5|context:qwen3_5)
      release_die "dense Qwen $attention_placement attention is unbuilt: native half-head GDN/GQA and dense FFN exports, shared reductions, and context KV/softmax merge are needed (#2133); use layers" ;;
    auto:minimax_m3|heads:minimax_m3|layers:minimax_m3) ;;
    *:minimax_m3) release_die "MiniMax attention placement is auto or heads on one RTX and auto, layers or heads on two RTX" ;;
    auto:step5|heads:step5|auto:muse_spark|heads:muse_spark|auto:inkling2|heads:inkling2|auto:inkling2_mini|heads:inkling2_mini) ;;
    *:step5) release_die "Step-5 has no context/layers attention executor: SWA splits heads and CSA uses whole owners" ;;
    *:muse_spark) release_die "Muse has no context/layers attention executor: native attention splits heads" ;;
    *:inkling2) release_die "inkling2 supports heads attention only; context and layers are not implemented" ;;
    *:inkling2_mini) release_die "inkling2_mini attention placement is TP1-only (heads); context and layers are not implemented" ;;
    layers:hy4_flash) release_die "hy4_flash supports context (2-GPU default) or explicit heads attention; layers is not implemented" ;;
    layers:hy4) release_die "hy4 supports context (2-GPU default) or explicit heads attention; layers is not implemented" ;;
    auto:*|heads:*|context:glm5|context:glm5_flash|context:deepseek_v4|context:hy4_flash|context:hy4|layers:qwen4|layers:qwen3_5) ;;
    # K4/P7 qualified explicit layers; context remains the recorded auto default.
    # The planner still requires the selected image's executor and native programs.
    layers:glm5|layers:glm5_flash)
      echo "note: $family attention placement layers explicitly selected; auto remains context on two GPUs" >&2 ;;
    context:*|layers:*) release_die "$family cannot run attention placement $attention_placement: executor not implemented; only heads is qualified" ;;
    *) release_die "ATTENTION_PLACEMENT must be auto, context, layers or heads" ;;
  esac
  if [[ "$family" == qwen3_5 && "$attention_placement" != auto && "$attention_placement" != layers ]]; then
    release_die "dense Qwen $attention_placement attention is unbuilt: add native half-head GDN/GQA and dense FFN exports with shared reductions; context also needs partitioned KV/softmax merge (#2133); use layers"
  fi
}
