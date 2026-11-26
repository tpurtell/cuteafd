# Hy4

The v7 Release-smoke checkpoint is `wrldsuksgo2mars/Hy4-EXL3-K4-MC`
(pinned revision `67346c4bdee2d8694606b4b4db9a96442e8667b7`) with EXL3 K4 MUL1
routed experts and four multi-core dense views (`original`, `k8-compact`,
`k6-compact` and load-time `fp8-compact`). The v7 main cards use automatic KV
sizing and `CORE_VIEW=k8-compact`.

## Architecture

78 layers, hidden size 6144, vocabulary 120832 (untied embedding and FP32
vocabulary head), 1,048,576-token maximum context:

- **Attention:** uniform gated DeepSeek Sparse Attention (DSA) MLA across all
  78 layers (64 query heads TP1 / 32 per rank TP2, `q_lora_rank=2048`,
  `kv_lora_rank=512`, `qk_nope=192`, `qk_rope=64`, `v=256`, F32 learnable
  sinks, elementwise sigmoid output gate, interleaved tail-64 RoPE).
- **IndexCache:** 32-head dim-128 per-token DSA indexer selecting top-2048
  tokens across 21 full-indexer layers (`[0, 1, 5, 9, ..., 77]`); the 57 shared
  layers reuse the preceding full layer's per-wave selection.
- **Residuals:** 4-stream identity hyper-connections (iHC, `H=6144`, FP32
  mixing, no sandwich norms) collapsed by `hc_head` before the final RMSNorm.
- **FFN / MoE:** layer 0 dense SwiGLU FFN (`I=18432`, unclamped); layers
  `1..=77` direct MoE (256 routed experts, top-8, sigmoid + bias router with
  `routed_scaling_factor=2.827`, `I=2048`, `swiglu_limit=10.0`, 1 shared expert,
  EXL3 K4 MUL1 storage; 347.40 GiB routed experts total). No LatentMoE, no
  Engram tables and no SWA layers.
- **Prefix cache:** pages-only exact prefix cache (`MarkStore::None`), with
  host tier support.
- **Bundled optional towers:** the checkpoint bundles a 5-layer DFly2 drafter
  (taps `[61, 65, 69, 73, 77]`, block size 7) and a 40-layer ViT vision tower
  (`vit.*`, 16 heads × 96). Current text serving runs with `SPECULATOR=off`,
  `VISION=off` and `AUDIO=off` while DFly2 and vision runtime wiring complete
  ([#2550](https://github.com/tpurtell/cuteafd/issues/2550)).

## Layouts and options

`cuteafd plan --spark-range` gives `N_min=4`, `N_max=8` declared
(`binding=[Size, Width, Cluster]`): **4–6 Sparks are executable today**
(`88.13` / `70.87` / `59.25` GiB routed experts per rank with zero forced
onboarding), while **7–8 Sparks have no executable package today**
(`executable today=false`). Three-Spark and zero-Spark layouts are not
executable in the release card set.

On two RTX GPUs, `ATTENTION_PLACEMENT=auto` (default) selects `context`
(ContextSplit: parity-owned DSA KV and IndexCache pages across both GPUs with no
KV replication). Explicit `ATTENTION_PLACEMENT=heads` opts into replicated-KV
head-split; single-GPU layouts keep whole layers, and `layers` is refused.

| Layout | Starting config | Planned pool / onboarding (`k8-compact`) |
| --- | --- | --- |
| 1 RTX PRO 6000 + 4 Sparks | [Min](../../examples/configs/hy4-exl3-k4-mc-min.config) | 1,261,120 tokens, 0/77 layers onboarded (88.13 GiB/Spark) |
| 2 RTX PRO 6000 + 6 Sparks | [Max (K8)](../../examples/configs/hy4-exl3-k4-mc-k8-max.config) | 2,149,760 tokens, 9/77 layers onboarded TP2 (52.33 GiB/Spark) |
| 2 RTX PRO 6000 + 6 Sparks (`original`) | [Max (original)](../../examples/configs/hy4-exl3-k4-mc-max.config) | 2,154,112 tokens, 4/77 layers onboarded TP2 (56.18 GiB/Spark) |
| 2 RTX PRO 6000 + 6 Sparks (`k6-compact`) | [Max (K6)](../../examples/configs/hy4-exl3-k4-mc-k6-max.config) | 2,174,720 tokens, 10/77 layers onboarded TP2 (51.56 GiB/Spark) |
| 2 RTX PRO 6000 + 6 Sparks (`fp8-compact`) | [Max (FP8)](../../examples/configs/hy4-exl3-k4-mc-fp8-max.config) | 2,161,024 tokens, 9/77 layers onboarded TP2 (52.33 GiB/Spark) |

The [MC comparison](../../scripts/bench/release-smoke-v7-mc-compact.json) runs
`original`, `k6-compact` and `fp8-compact` on the same two-RTX + six-Spark
layout as the K8 base. Cards and quality comparisons are pending Release smoke.

See [Hy4 Serving](../options.md#hy4-serving) for accepted launch keys, KV
defaults (`KV_CACHE=fp8`, `bf16` opt-in), `DECODE_ROWS` (fixed 64),
`EMBEDDING=gpu`, `EXPERT_USAGE=on`, `PREFILL_PACKING=unrestricted`, and
rank-local CUDA graph reserve controls.

## Changelog

| Release | Date | Changes | Basic eval |
| --- | --- | --- | --- |
