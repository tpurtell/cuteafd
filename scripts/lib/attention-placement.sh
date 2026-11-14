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
    # Step-5 and Muse Spark run only their recorded automatic placement.
    auto:minimax_m3|heads:minimax_m3) ;;
    *:minimax_m3) release_die "MiniMax WIP currently supports one-RTX whole attention (auto or heads)" ;;
    auto:step5|auto:muse_spark) ;;
    *:step5|*:muse_spark) release_die "$family supports only ATTENTION_PLACEMENT=auto" ;;
    auto:*|heads:*|context:glm5|context:glm5_flash|context:deepseek_v4|layers:qwen4) ;;
    # K4/P7 qualified explicit layers; context remains the recorded auto default.
    # The planner still requires the selected image's executor and native programs.
    layers:glm5|layers:glm5_flash)
      echo "note: $family attention placement layers explicitly selected; auto remains context on two GPUs" >&2 ;;
    context:*|layers:*) release_die "$family cannot run attention placement $attention_placement: executor not implemented; only heads is qualified" ;;
    *) release_die "ATTENTION_PLACEMENT must be auto, context, layers or heads" ;;
  esac
  # Dense Qwen is a single-device executor: auto resolves to heads, an explicit
  # mode is not qualified.
  [[ "$family" != qwen3_5 || "$attention_placement" == auto ]] ||
    release_die "dense Qwen currently supports only single-GPU automatic attention placement"
}
