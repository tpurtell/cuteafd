# DeepSeek V4.1 Flash

A core CuteAFD reference family. Releases are checked against this engine's own cards; correctness and memory safety gate, speed regressions are reported.

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
- RTX/Spark layouts: the v2 native MXFP4 minimum is 1 RTX + 3 Sparks;
  NVFP4 requires 4 Sparks with this launcher's kernel support. Maximum is 2 RTX +
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

<!-- release-v2-limits -->
- **rc2 32 GB startup failure:** `v41-flash-sim5090` fails before any baseline: a startup expert-session reconnect overlaps the old Spark endpoint and requests 419,495,936 B of mapped RDMA rings against the 302,120,960 B budget. The old session is not released first. Vision uses plain TCP and is not the third ring. The fix is targeted for rc3, not shipped or qualified in rc2; the separately labelled vision-off diagnostic reproduces the same 419,495,936 B versus 302,120,960 B failure, has no measured baseline, and is not a qualified replacement.
- Cold-prefill non-bit-reproducibility and prefill/decode stalls remain open; prefix snapshot restore exactness does not establish batch-invariant cold prefill.
- A historical small-card host-heap abort was not reproduced or root-caused. The hunt is closed for v2 unless it recurs; MALLOC_CHECK_=3 serving soaks remain required and no memory-safety qualification is claimed. The fixed graph bank addresses graph residency, not the heap cause.
- **Memory vs. kernel support:** NVFP4 fits by memory at 1x RTX + 3 Sparks (TP3), including the 31.8 GiB simulation. Kernel/launcher support requires TP4: the TP3 package is native-only and explicit TP3 rejects NVFP4 (`run.sh:263`). The NVFP4 reference cards therefore use four Sparks; the native MXFP4 minimum uses three.
- The official `deepseek-ai/DeepSeek-V4.1-Flash` reference (`deepseek_v41-v2_20261005`) applies to every V4.1 quant, including `nvidia/DeepSeek-V4.1-Flash-NVFP4`. rc2's bench client did not map the NVFP4 checkpoint ID to that publication, so these cards are not fidelity-qualified; the client-only lookup fix (`36993a2d`) is targeted for the next RC, not shipped in rc2. C1/C8/8K rates and cache results stay unchanged. An aggregate quality badge does not establish logit qualification, and the applicable gate remains top-1 >=94% / KL <=0.04.

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
| v2 | 2026-10-09 | Spark-rank-0 vision with independent image admission; 2M PRO/1M small-card KV defaults; measured startup/graph admission, host embedding and bounded small-card graphs; unified SM120 launch sizing. | <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-1rtx-3spark-v41-flash-min-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (1× RTX)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-2rtx-4spark-v41-flash-max-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (2× RTX)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-rc2-sim5090/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-rc2-sim5090/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (5090)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (1× RTX)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-09-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (2× RTX)"></a>  <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-v41-flash-rc2-sim5090/report.json">5090 startup FAIL: reconnect ring budget; no performance</a> · <a href="../../benchmarks/deepseek_v41/2026-10-09-smoke-v41-flash-rc2-sim5090/diagnostic-vision-off.json">diagnostic vision off: same FAIL, not qualified</a> |
| v1 | 2026-10-04 | Coordinator-first loading and smaller one-RTX workspaces; opt-in device exchange and default shared single-copy FP8 head; exact turn-end restore check; matched dual-RTX C1 requalification. | <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-1rtx-4spark-v41-flash-min/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-05-smoke-deepseek-v4-1-flash-2rtx-4spark-rc2/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-1rtx-4spark-v41-flash-nvfp4-min/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-04-smoke-deepseek-v4-1-flash-nvfp4-2rtx-4spark-v41-flash-nvfp4-max/card.svg" width="360" alt="DeepSeek-V4.1-Flash-NVFP4 (nvfp4-g16) (max)"></a> |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v41/2026-10-02-smoke-deepseek-v4-1-flash-2rtx-4spark/card.svg" width="360" alt="DeepSeek-V4.1-Flash (mxfp4-g32) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
