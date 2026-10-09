# MiMo V2 (V2.6 Flash / Pro)

Hybrid full / sliding-window GQA attention with learned sinks, sigmoid top-8
experts without a shared expert.

## Supported checkpoints / quants

- `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` — default Flash checkpoint, native
  MXFP4 routed experts (`mimof:fp8`, E2M1 + UE8M0 per 32), bundled DFlash
  drafter. Natural minimum: 1 RTX + 2 Sparks; maximum: 2 RTX + 4 Sparks.
- `XiaomiMiMo/MiMo-V2-Flash` — legacy FP8 128x128-block routed experts
  (`mimo:fp8`); still loads, with historical results retained.
- `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` — native MXFP4 routed experts
  (`mimop:fp8`, E2M1 + UE8M0 per 32). Supersedes `MiMo-V2.6-Pro-RL`
  (same architecture, config, tokenizer and tensor layout; Xiaomi's MOPD2
  pass fixes the RL release's tool-call repetition). RL still loads.

## Engineering summary

- Attention: GQA full attention on some layers, 128-token sliding-window
  GQA with a learned per-layer sink bias on the rest; sinks are taken on SWA
  layers only. NeoX RoPE on a partial head width; int8 full-attention KV
  (FP32 scale per 32 dims) in paged pools, BF16 SWA rings.
- No shared expert; sigmoid noaux_tc router. Flash MOPD uses BF16 router
  weights and an FP32 `e_score_correction_bias` (legacy Flash's router
  program uses FP32 hi/lo operands).
- Routed experts: Flash MOPD runs `mimof:fp8` (MXFP4, Spark TP2/TP4,
  coordinator TP1 local); legacy Flash runs `mimo:fp8` (E4M3 + FP32
  128x128 scales, Spark TP2/TP4/TP6); Pro runs `mimop:fp8` (MXFP4,
  Spark TP2/TP6, coordinator TP1 local).
- Speculator: native MTP layers (SWA attention, dense FFN, `eh_proj`
  fusion) or a DFlash external drafter — DFlash is the measured-best
  speculator for V2.6 Flash MOPD and Pro. Flash defaults to the checkpoint's
  own `dflash/`, with FP8 drafter weights; `SPECULATOR=off` disables it,
  `SPECULATOR_FP8=auto` preserves source BF16, and `off` explicitly selects BF16.
- Flash MOPD's constants have separate `mimof`/`mimof2` programs: norm
  epsilon 1e-6, full/SWA RoPE theta 1e7/1e4, target value scale 0.707,
  QK/V widths 192/128 and partial NeoX RoPE width 64. Shape-compatible
  arithmetic changes fail against the loaded program manifest before allocation.
- Flash MOPD and Pro's `qkv_proj` ship fused and TP-interleaved; the loader
  reads `metadata.tp_size` (4/8) and de-interleaves whole checkpoint shards.
  Each Q/K/V segment starts its own FP8 128x128 scale grid. Flash SWA has
  two contiguous 192-row key heads per checkpoint shard, not Pro's padded
  256-row key stride. Missing/invalid TP metadata is rejected.
- RTX/Spark layouts: head split (KV partitioned, one hidden all-reduce per
  layer) is the default on 2 RTX for both members of this family — the
  first family where splitting attention across two GPUs measured as a
  clear win, since its o_proj and attention weights are unusually large
  next to the dense path.
- Prefix cache: merged for both V2 Flash and V2.6 Pro.

## Flash MOPD qualification (2026-10-05)

Target precision stays at the checkpoint: BF16 head and o_proj, native FP8
block projections and MXFP4 experts. Only the bundled drafter converts to
single-copy FP8 by default; this is exactly the measured FP8 arm, not a
combined target head/O conversion. Explicit `MIMO_FP8_HEAD`/`MIMO_FP8_O_PROJ`
remain overrides, not newly qualified target defaults.

Conditions: 325 W RTX PRO 6000, rail A (live raptor 400 Gb/s and Spark ports
200 Gb/s), expert FP8 input, int8 full KV/BF16 SWA, graphs off, 16K context,
32K-token pool, prefill rows4096, four concurrent sequences. One warm launch
per arm, three interleaved C1/C4 all-code SSE waves (512-output limit), median
emitted tok/s; content, reasoning and tool deltas establish timing, not usage
or finish frames. The basic card uses a different 320-output code request.

| Layout | Drafter | C1 code | C4 aggregate | Warmed ~8K prefill |
| --- | --- | ---: | ---: | ---: |
| 1 RTX + 2 Sparks | BF16 | 75.78 | 118.81 | 5,444 |
| 1 RTX + 2 Sparks | **FP8 (default)** | **81.05** | **131.02** | 5,455 |
| 2 RTX + 4 Sparks | BF16 | 124.39 | 200.50 | 7,743 |
| 2 RTX + 4 Sparks | **FP8 (default)** | **127.49** | **212.04** | 8,399 |

tok/s. Drafter-only FP8 improves C1/C4 by 7.0%/10.3% at minimum and
2.5%/5.8% at maximum. No extra Flash TP6 performance layout: intermediate2048
splits exactly at TP4.

The official-weight eager reference widens MXFP4 exactly, dequantizes FP8
to BF16, retains FP32 correction biases and disables TF32. Its 512-position
NLL is 3.44858352. Native full-vocabulary smoke scores (513 logit rows):
minimum top-1 90.1%, NLL3.4570, KL0.04298; maximum 90.4%, NLL3.4883,
KL0.04759. API top-12-plus-tail coarse KL is separately 0.022/0.026,
with top-1 89.8%/88.7%. This passage is a smoke, **not** the redesigned
agentic fidelity set or a target-precision decision.

Native replay restores layers, logits, KV rows and the 25,559,040-byte mark
byte-exactly at position257 with 128-row chunks. Embedding/gather/greedy token
I/O and sampled distributions pass; not every stochastic host/device draw
matches. API prompt/turn restoration rows and the tested 128 greedy tokens
with DFlash on/off are identical on both layouts and drafter formats.

The reasoning-high money fixture at seed0 uses six turns and valid tools.
BF16 passes at 2048 tokens/turn; minimum FP8 initially hits that length limit
on turn2 and fails. One matched 4096-token/turn retry per arm succeeds for
both (six turns/eight tools/zero invalid). BF16 -> FP8: reasoning characters
8712 -> 9233 (+6.0%), output tokens3141 -> 3343 (+6.4%), session wall
51.14 -> 54.01 s, reported decode69.64 -> 68.14 tok/s. This single-session
retry establishes completion, not an agentic-speed win; the default follows
the paired code gains without a repeat failure or marked reasoning expansion.
Maximum fixtures already pass at2048 (eight/seven tools;121.70/128.00 decode
tok/s). Original cutoff evidence is retained, not overwritten.

Artifacts use Cargo/CMake Release-mode builds on task-specific dev-base
images, **not** a Release image/cut. Card report JSON records source commit,
fork pin, immutable image IDs and artifact SHA256s. Existing-family Pro
smoke passes (88.1% top-1, NLL3.3397, full KL0.04747); full-pin parity is
pending the coordinator's V4.1 merge gate.

The real launcher, with no speculator/precision overrides, selects bundled
FP8 DFlash and preserves checkpoint target formats on both layouts. Overlay
smoke passes fidelity, byte-exact prefix restoration and the tested 128-token
greedy draft-on/off comparison; template/batch checks remain Info. API readiness
is 51.05 s minimum / 42.35 s maximum. Published basic cards use their own
320-output C1 request (89.73 / 122.42 tok/s) and cold 8K prefill
(5,325 / 7,395 tok/s), not the warmed qualification table. Card report JSON
retains full native KL, the original 2048-token cutoff and matched 4096-token
completion evidence. Raw GPU runs pass; a container-file ownership error during
report annotation was recovered on CPU without repeating either launch.

## Legacy target precision (single residency)

Every weight has one resident format. Precision is chosen by measurement:
FP8 converts at load into the only copy where it is faster and the golden
stays within ~0.005 nat KL/NLL; drafters run FP8 whenever emitted tok/s is
higher (they cannot change the output). Measured 2026-10-03, natural minimum,
one warm launch per arm, `CONCURRENCY=4`, code tok/s (C4 aggregate), golden
512 tokens.

| Arm | C1 | C4 | 8K prefill | KL · top-1 · NLL |
| --- | ---: | ---: | ---: | --- |
| V2 Flash checkpoint | 69.4 | 108.5 | 5,864 | 0.103 · 82.6% · 4.313 |
| **V2 Flash FP8 head + o_proj (default)** | 72.6 | 125.1 | 5,548 | 0.092 · 83.2% · 4.308 |
| V2.6 Pro checkpoint | 55.1 | 71.5 | 2,352 | 0.026 · 86.7% · 3.379 |
| **V2.6 Pro FP8 head + o_proj + DFlash (default)** | 58.0 | 87.3 | 2,656 | 0.028 · 86.5% · 3.368 |

MiMo V2 Flash 1 RTX + 4 Sparks; V2.6 Pro 1 RTX + 6 Sparks. Coordinator VRAM: V2 Flash 16.4 → 14.2 GB; V2.6 Pro 46.0 → 36.4 GB (v0 dual copy 59.9 GB). `MIMO_WEIGHT_POLICY=checkpoint` keeps source formats; `MIMO_FP8_HEAD`/`MIMO_FP8_O_PROJ`/`SPECULATOR_FP8` override.

## Known limits

<!-- release-v2-limits -->
- **Memory vs. kernel support:** the planner reports 1,048,576 checkpoint context tokens and no compiled-index-extent requirement. Flash uses the memory-fitting two-Spark minimum; Pro needs six Sparks. Pro's 31.8 GiB no-fit is a memory rejection, whereas the PRO-card automatic-pool startup rejection below is runtime admission, not a missing TP kernel. The configured Pro minimum retains full context with a 1,048,576-token pool.
- **v2.0.0 blocker, Pro minimum at default settings:** automatic KV sizing fills the 97% PRO ceiling, then the 64 MiB Spark intake startup probe is charged on top. Admission needs 98,393,355,060 B against 98,327,870,832 B (65,484,228 B short). Vision/audio are already on Spark ranks 4/5. The configured retry uses a 1,048,576-token pool with full 1,048,576-token context; it does not qualify the broken automatic default. Fix automatic pool sizing after reserving max(intake probe, small-card floor) as a startup peak, so the probe always fits (`rust/crates/cuteafd-daemon/src/families/mimo_v2/admission.rs:460`).
- Historical text-only Flash MOPD bring-up cards do not qualify multimodal or 1M-context operation. This release smoke enables bundled vision/audio admission but measures text and quick quality, not a full multimodal/context campaign.
- `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` is not supported on a single 32 GB card with all six available Sparks; needs a larger coordinator card or a different checkpoint/placement. rc1 planner: rtx0 full memory layout needs 33511016304 bytes, budget 31031138713 bytes, shortfall 2479877591 bytes. See the linked planner-only 5090 cell; no performance was measured.

- V2.6 Pro's prefill is intake-bound on the Spark-to-coordinator exchange of
  partial rows at larger TP; Spark-side reduction of partials is a parked
  experiment (small gain, bandwidth-bound either way).
- V2.6 Pro's native checkpoint needs all six Sparks to fit its MXFP4 expert
  footprint; V2 Flash fits a smaller pool.
- Legacy V2 Flash's maximum layout prefills more slowly than its minimum;
  its smoke fidelity is near the threshold (KL about0.10, top-1 about82%).
  These historical results do not describe Flash MOPD. BF16 expert-input
  Spark packages remain opt-in (`EXPERT_INPUT=bf16`) and must be in the image.
- Flash MOPD tool calls parse and reasoning survives the round trip, but
  canonical re-render differs at token95 (Info), not an exact template claim.
  C1/C4 greedy outputs diverge (Info). Exact restore is replay at the same
  chunk boundary, not equivalence to a differently chunked straight prefill.
- Historical Flash MOPD bring-up qualification was text-only with bundled
  vision/audio towers skipped. Its measured16K/32K-pool configuration does
  not qualify the advertised1M context or real5090 performance; the v2 cards
  are distinct release-smoke measurements with current admission defaults.
- Exact Spark slices remove zero padding. The v1 scope's MXFP4 32-row
  down-projection tails are implemented on the unmerged `work/mxfp4-tails`
  branch; distributed-oracle and unchanged-NLL gates remain open. Larger
  padded slices can still cost memory and expert-wave time.
- Batch-invariant prefill and verify are deferred; a speculation-lossless
  smoke result permits proven numerical rounding, not a state mismatch.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v2 | 2026-10-09 | V2.6 Flash MOPD default; bundled vision and qualified MiMo audio auto admission with audio AOT in release images; full-context/small-card memory admission; Hugh-derived opt-in queue, warm snapshots and indexed copy windows. | Pending rc1 exports |
| v2-bringup | 2026-10-05 | Flash default moves to V2.6 Flash MOPD: distinct arithmetic programs, official TP4 QKV/MXFP4, TP2 minimum, checkpoint target formats and qualified FP8 bundled DFlash; text-only overlay smoke, not a release cut. | <a href="../../benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-1rtx-2spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-1rtx-2spark/card.svg" width="360" alt="MiMo-V2.6-Flash-MOPD (mxfp4-g32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-2rtx-4spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-05-smoke-mimo-v2-6-flash-mopd-2rtx-4spark/card.svg" width="360" alt="MiMo-V2.6-Flash-MOPD (mxfp4-g32) (max)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (max)"></a> |
| v1 | 2026-10-04 | V2.6 Pro MOPD checkpoint and DFlash; native A8 MXFP4 down projection and exact Spark slices; single-copy FP8 head/O/drafter; pipelined head-split prefill; Flash two-lane default. | <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-flash-1rtx-4spark-mimo-flash-min/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-flash-1rtx-4spark-mimo-flash-min/card.svg" width="360" alt="MiMo-V2-Flash (fp8-block128x128/f32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-flash-2rtx-4spark-mimo-flash-max/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-flash-2rtx-4spark-mimo-flash-max/card.svg" width="360" alt="MiMo-V2-Flash (fp8-block128x128/f32) (max)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark-mimo-pro-min/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark-mimo-pro-max/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (max)"></a> |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/mimo_v2/2026-10-02-smoke-mimo-v2-flash-1rtx-4spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-02-smoke-mimo-v2-flash-1rtx-4spark/card.svg" width="360" alt="MiMo-V2-Flash (fp8-block128x128/f32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-02-smoke-mimo-v2-flash-2rtx-4spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-02-smoke-mimo-v2-flash-2rtx-4spark/card.svg" width="360" alt="MiMo-V2-Flash (fp8-block128x128/f32) (max)"></a> |
| v0-mopd | 2026-10-04 | Model-affecting: V2.6 Pro checkpoint `MiMo-V2.6-Pro-RL` → `MiMo-V2.6-Pro-MOPD` (Xiaomi's MOPD2 pass over the RL weights fixes tool-call repetition; architecture, config, tokenizer, chat template and tensor layout unchanged, DFlash drafter retrained). Golden and bench fidelity reference regenerated from MOPD. | <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-1rtx-6spark/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (min)"></a> <a href="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark/report.svg"><img src="../../benchmarks/mimo_v2/2026-10-04-smoke-mimo-v2-6-pro-mopd-2rtx-6spark/card.svg" width="360" alt="MiMo-V2.6-Pro-MOPD (mxfp4-g32) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
