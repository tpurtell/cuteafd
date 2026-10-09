# DeepSeek V4.1 Flash

The regression anchor for CuteAFD: every phase and every shared-hot-path
change is checked against its parity numbers before merging.

## Supported checkpoints / quants

- `deepseek-ai/DeepSeek-V4.1-Flash` — official release: FP8 32x32-block
  coordinator weights (E4M3 + UE8M0 scales), native MXFP4 routed experts
  (E2M1 + UE8M0 per 32).
- EXL3 K2–K4 routed-expert quants of the same checkpoint.
- NVIDIA ModelOpt NVFP4 release: native W4A4 routed experts (E2M1 weights
  and activations with E4M3 group-16 scales). The coordinator sends BF16
  hidden rows; Sparks quantize each routed row to FP4 before the MMA.

## Engineering summary

- Attention: compressed MLA with a per-layer compression ratio schedule and
  CED (compressed encoder/decoder) KV sharing across index source layers
  `[2, 8, 14, 20]`.
- Engram memory-mapped embedding tables gate layers 1 and 14, read through
  the shared `MappedTable` path (page cache, bounded prefetch, pinned
  upload ring).
- Routed experts run on 2, 3, 4 or 6 Spark ranks over `expertd-native`
  (MXFP4, EXL3 K2–K4, or ModelOpt NVFP4); mHC hyper-connections mix the
  coordinator residual stream.
- Speculator: three-stage dSpark drafter (`markov_head`, `confidence_head`,
  `main_proj`) with adaptive width from a calibrated confidence and cost
  model.
- KV format: compressed MLA latent in FP8 32x32 blocks; a specialized
  exact prefix cache — radix banks keyed by token ids, shared
  FP4 pages, a copied SWA "front", pinned-host tier in `cuteafd-hostcache`.
- RTX/Spark layouts: natural minimum is 1 RTX + 4 Sparks; maximum is 2 RTX +
  4 Sparks. V4.1 keeps the coordinator layer-range split by default — the
  measured head-split hop cost on this fabric does not clear the bar its
  attention weights would need to win (see `PLAN.md` Phase 6).
- Vocabulary head: RC1 and RC2 default to `CUTEAFD_V41_FP8_HEAD=all`, one
  shared E4M3 copy for target and dSpark (FP32 scales per row and 128-wide K
  block), partitioned by vocabulary rows on dual RTX. BF16 is released after
  packing. `off` keeps BF16; `draft` retains both formats. Earlier release
  notes incorrectly described BF16 as the default. The matched two-RTX
  v0/RC1 ABAB recheck did not reproduce the historical C1 regression.
- Optional vision tower (MoonViT-style) when the checkpoint carries
  `vision_config`.

## Known limits

- The FP8 vocabulary head regressed short "hello" replies in its earlier
  comparison. The release C1 recheck covers the two-RTX code workload;
  it does not close the separate short-reply limitation.
- Prompt and turn-end prefix restores are byte-exact against their own
  snapshots. Cold prefill can differ because Spark FP32 atomic expert
  reductions are arrival-ordered at 256+ rows; batch-invariant prefill and
  verify remain deferred.
- Large index-selection and attention-query graph entries can be evicted
  between encoder and replay shapes, so warmed requests can recapture graphs.
- Device-driven exchange remains opt-in (`CUTEAFD_V41_DEVICE=1`); write mode
  (`CUTEAFD_SPARK_WRITE=1`) can stall on written flags and is unqualified.
- NVFP4 uses W4A4 for prefill, decode and speculative verification. FC1
  currently substitutes the layer's maximum static `input_scale` for each
  expert's calibration; FC2 retains per-expert scales. Checkpoint-exact FC1
  calibration, smaller wire rows and decode-kernel occupancy remain follow-ups.
- One-RTX startup still waits on slow Spark layer reads after the
  coordinator-first placement handoff; further load-speed work is open.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v1 | 2026-10-04 | Coordinator-first loading and smaller one-RTX workspaces; opt-in device exchange and default shared single-copy FP8 head; exact turn-end restore check; matched dual-RTX C1 requalification. | <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (max)"></a> |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
