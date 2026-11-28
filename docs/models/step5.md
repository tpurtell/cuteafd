# Step-5

Automatic KV sizing uses the shared 2M-token target on every card, including
32 GB / RTX 5090. Placement and expert onboarding are fixed at that target;
the pool then grows into remaining admitted memory, within native encoding
limits. `KV_MAX_TOKENS` is unset by default and can explicitly cap growth.
See [memory policy](../options.md#memory-and-pools).

Step-5-Preview serves text from its standard Hugging Face checkpoint headers.
The v7 MC cards use K8 compact on one RTX PRO 6000 plus three Sparks and
two RTX plus six Sparks. V7 drops the original, K6 compact and FP8 compact comparison (TJ 2026-11-28); its last cards stay pinned in the [comparison matrix](../../scripts/bench/release-smoke-v7-mc-compact.json)'s retired list. Official FP8 and NVFP4 remain supported; NVFP4 experts
use the checkpoint's calibrated W4A4 math, including decode.

## EXL3 K4 MC

`wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC` at
`aedda90d25bd02bb6520bcaf88e06e6c9bc932c4` is quantized from the BF16
release. It replaces the standalone `Step-5-Preview-EXL3-K4-Core-K6`
release row; historical reports remain archived. Its K4 routed experts
(K6 for the last three routed layers, 88-90) run
on the Sparks from the shared `exl3-step5-k46` package, with Step's own
post-SiLU clamp at 7. The target attention q/k/v/o, shared and dense FFNs, and
lm_head have original BF16, native EXL3 K8 and K6 views on the RTX. Compact
views run through the shared dense trellis programs. Gate and up stay separate
projections, and the head-split
o_proj reduces its F32 partials before one output rotation. Head-wise
attention gates, the sparse indexer, routers, norms, embeddings and the vision
tower stay BF16/F32 exactly as shipped. The repaired original view restores
source BF16 non-routed matrices for all three MTP stages, including each
stage's own vocabulary head. Stored compact views retain their existing
K4 MTP matrices; routed expert encodes are unchanged.

Everything is read from `config.json` and the safetensors headers;
`quantize_config.json` is only cross-checked. `cuteafd plan` finds the
fit for each selected view. The card layouts are 1 RTX PRO 6000 + 3 Sparks
and 2 RTX PRO 6000 + 6 Sparks. Routed layers are onboarded to the RTX by default
(`RTX_EXPERT_LAYERS` unset is pool-first auto). After admission sizes the layout for the 2M KV target,
one RTX runs whole layers from `rtx-tp1`; two RTX split each onboarded layer
TP2 through the shared pair and `rtx-tp2`, with F32 partial exchange.
The automatic pool then grows into remaining memory; `KV_MAX_TOKENS` can cap it. The
prior K6 96 GiB header-fixture maximum onboards 24 layers; its minimum prioritizes
its roughly 1.81M-token pool and onboards none. These are planner results,
not hardware qualification. Start from
the MC [min](../../examples/configs/step5-exl3-k4-mc-min.config) and
[max](../../examples/configs/step5-exl3-k4-mc-max.config) configs, which pin the MC
publication, revision and `CORE_VIEW=original`. The old K6-core paths remain
retired historical recipes. The release smoke matrix uses the MC configs;
MC original min/max, all three max core views and all three min Standard
fidelity views are measured ([#1906](https://github.com/tpurtell/cuteafd/issues/1906)).

## Core views

The old `-HQA` checkpoint was withdrawn from Hugging Face. Use the published MC checkpoint for selectable core views.

The published MC follows the [MC contract](../mc-checkpoints.md):
`CORE_VIEW=original` (default) restores the BF16 target core and, in repaired
revisions, the BF16 MTP core (attention, FFN, EH and private vocabulary heads).
`k8-compact` and `k6-compact` retain their existing target encodes and K4 MTP
core. Routed expert encodes and non-core tensors stay unchanged. The selected
MTP matrices load natively into `step5/mtp/gpuN`, one independently retired
packed arena; admission charges its physical bytes and single allocation tail,
not a second payload copy. BF16 original MTP therefore costs more than K4.
Plain checkpoints refuse compact selectors. Release cards select `original`;
max-layout compact comparisons use the separate
`scripts/bench/release-smoke-v51-mc-core-comparison.json` matrix.

`fp8-compact` is an explicit, quality-affecting opt-in over exactly the
original view's selected q/k/v/o, dense/shared FFN, EH and vocabulary-head
matrices, including the three MTP stages' non-routed core. Native E4M3 values
and F32 block-128 scales occupy the existing target and independently retired
MTP arenas; no resident BF16 shadow or expert conversion is introduced.
Heads use the shared executor with private scratch. Unselected keep tensors
and routed EXL3 experts retain their checkpoint formats. Requires the v52e
coordinator projection exports; `original` remains the default.

Max-card qualification on 2026-11-16 passes generated Standard quality and
strict MTP-depth-3 losslessness (10/10 cases, token IDs and row hashes exact).
One warmed launch per view on 2 RTX PRO 6000 + 6 Sparks, 425 W/card,
checkpoint `aedda90d25bd02bb6520bcaf88e06e6c9bc932c4`:

| Metric | Original | FP8 compact |
| --- | ---: | ---: |
| C1 emitted tok/s | 59.87 | 68.98 |
| C16 emitted tok/s | 216.73 | 266.95 |
| 8K prefill tok/s | 2740 | 2562 |
| KV pool tokens | 2097152 | 2097152 |
| Onboarded RTX expert layers | 13 | 19 |
| Generated decode KL / top-1 | 0.01337 / 96.069% | 0.01597 / 95.166% |
| Generated prefill KL / top-1 | 0.01164 / 96.008% | 0.01429 / 95.811% |

The original arm uses engine `42a9b82b1`; FP8 replay uses `a0751bf38`, whose
only intervening runtime change fixes FP8-only scratch inventory. Original
execution is unchanged, and all 106 saved prompt hashes match the replay.
Decode has zero power-cap/thermal slowdown; GPU0/GPU1 busy C16 clocks are
2779/2845 MHz original and 2782/2841 FP8. Prefill incurs about 1-4 power-cap
GPU-seconds per point, no thermal slowdown, maximum 73 C and minimum T.Limit
headroom 24 C. Corrected CUDA peaks are GPU0 96.02 to 100.50 GB and GPU1
100.01 to 98.68 GB; additional onboarding uses the freed weight capacity.
Absolute quality bars pass; no paired non-inferiority claim is made. Reference
is still reference-only qualified, not engine calibrated. PR #2214 and tasks
`v5x-mc-fp8-step-max-fixed` / `v5x-mc-fp8-step-scratch-fixed` retain evidence.

Standard fidelity (2026-11-13) uses the published 32 decode + 32 prefill
windows rooted at `stepfun-ai/Step-5-Preview-BF16`, on the min card. Both paths use
bounded streamed prefill scoring (`full-prefill-logits=false`); every completed
path scored 16,384 positions with zero missing or non-finite rows. The
routed-expert 3-6 bpw bar is generated-row KL <= 0.05 and top-1 >= 0.93;
context-inclusive scores are informational, not the qualification policy.

| View / report | Path | All KL / top-1 | Generated KL / top-1 | All / generated NLL | Tier verdict |
| --- | --- | --- | --- | --- | --- |
| [original](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-fidelity-min-cand-r1/report.json) | decode | 0.01725 / 96.28% | 0.01267 / 95.92% | 0.66982 / 0.63749 | pass |
| [original](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-fidelity-min-cand-r1/report.json) | prefill | 0.02174 / 95.78% | 0.01120 / 96.15% | 0.86380 / 0.59045 | pass |
| [k8-compact](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-k8-compact-fidelity-min-cand-r1/report.json) | decode | 0.01550 / 96.41% | 0.01272 / 96.02% | 0.66744 / 0.63783 | pass |
| [k8-compact](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-k8-compact-fidelity-min-cand-r1/report.json) | prefill | 0.02240 / 95.79% | 0.01191 / 96.26% | 0.86576 / 0.59260 | pass |
| [k6-compact](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-k6-compact-fidelity-min-cand-r1/report.json) | decode | 0.01615 / 96.37% | 0.01311 / 95.97% | 0.66949 / 0.63949 | pass |
| [k6-compact](../../benchmarks/step5/2026-11-13-custom-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-k6-compact-fidelity-min-cand-r1/report.json) | prefill | 0.02423 / 95.70% | 0.01291 / 96.01% | 0.86608 / 0.59158 | pass |

One launch per view, default speculation, floating 850 W shared budget.
No software/hardware thermal slowdown occurred. GPU0 panel windows:

| View | Power-cap seconds | Busy SM clock | Max temperature | Min T.Limit headroom |
| --- | ---: | ---: | ---: | ---: |
| original | 9.0 s | 2670 MHz | 67 C | 28 C |
| k8-compact | 13.9 s | 2612 MHz | 68 C | 28 C |
| k6-compact | 15.9 s | 2613 MHz | 71 C | 28 C |

The published Step reference remains reference-only qualified, not
engine-calibrated; its reference manifest records a legacy Qwen text-checkpoint
label, while its root, revision, dataset and hashes identify Step-5 BF16.
All 32+32 windows completed; [#1899](https://github.com/tpurtell/cuteafd/issues/1899)
and [#1849](https://github.com/tpurtell/cuteafd/issues/1849) remain separate
engine work.

## MC original min card

The 2026-11-13 [Release smoke report](../../benchmarks/step5/2026-11-13-smoke-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-min-cand-r1/report.svg),
including quick fidelity and exact prefix-cache checks, passes quality.
C1 / C8 / 8K / 8K-with-tail: 43.03 / 96.70 /
2405 / 2259 tok/s. This is a single warmed
candidate run, not an A/B speed claim. Host page-reclaim caveats are retained
in the run evidence; they do not trigger a repeat campaign.

Baseline GPU0: 7.0 s power-cap, zero thermal slowdown,
2720 MHz busy SM clock, 68 C maximum temperature,
25 C minimum T.Limit headroom, floating 850 W shared budget.
The report completed before a cleanup timeout; later inspection confirmed the serving container absent ([#1948](https://github.com/tpurtell/cuteafd/issues/1948)).

## MC max core-view cards

Measured 2026-11-13 on two RTX PRO 6000 GPUs plus six Sparks, with a
2,097,152-token pool, default speculation and one warmed launch per view.
All six campaign jobs completed with exit 0 and passed quick fidelity and
speculation losslessness. These smoke checks complement, not replace, the
published Standard fidelity panels above.

| View / report | C1 | C8 | 8K | 8K with tail |
| --- | ---: | ---: | ---: | ---: |
| [original](../../benchmarks/step5/2026-11-13-smoke-step-5-preview-exl3-k4-mc-2rtx-6spark-step5-mc-original-max-cand-r1/report.svg) | 61.50 | 154.46 | 2635 | 2467 |
| [k8-compact](../../benchmarks/step5/2026-11-13-smoke-step-5-preview-exl3-k4-mc-2rtx-6spark-step5-mc-k8-compact-max-cand-r1/report.svg) | 67.25 | 153.20 | 2598 | 2579 |
| [k6-compact](../../benchmarks/step5/2026-11-13-smoke-step-5-preview-exl3-k4-mc-2rtx-6spark-step5-mc-k6-compact-max-cand-r1/report.svg) | 63.74 | 155.43 | 2666 | 2550 |

Rates are tok/s. Host page-reclaim and CPU/memory-pressure caveats remain
in the raw campaign evidence (`ab_verdict=rerun`); there is **no causal
core-view speed claim** and no extra comparison run. Original selects the
release-table cell; compact views remain supplementary reports.

No software/hardware thermal slowdown occurred in any recorded window.
Baseline windows below use GPU0 / GPU1 order; floating daemon budget 850 W.
Power-cap throttling reflects that budget, not thermal slowdown.

| View | Power-cap seconds | Busy SM clock (MHz) | Max temperature (C) | Min T.Limit headroom (C) |
| --- | ---: | ---: | ---: | ---: |
| original | 10.0 / 2.0 | 2733 / 2828 | 64 / 70 | 31 / 26 |
| k8-compact | 13.9 / 3.0 | 2693 / 2799 | 66 / 73 | 29 / 24 |
| k6-compact | 14.0 / 7.0 | 2691 / 2796 | 69 / 75 | 31 / 24 |

Prompt-end and turn-end restored rows and cold recomputes were byte-identical
for every view. The cache check remains informational: the short fresh
continuation path was not covered, so this is not a claim of that coverage.

## Native MTP

Native MTP is enabled by default at depth 3 for both layouts. The three
predictor stages are checkpoint layers 92-94; no separate drafter download is
needed. `SPECULATOR=off` disables MTP in the launcher; direct `serve-step5`
users can pass `--mtp 0`. Use `SPECULATOR=mtp` and `SPECULATOR_DEPTH=1`, `2`
or `3` to select another depth explicitly.

The planner defaults to the same three stages. `--native-mtp-layers` selects
the planned depth; `--no-speculator` or `--speculator off` excludes predictor
weights and state. The launcher forwards its effective depth to the planner,
including zero when disabled. Admission charges predictor weights, active KV
and hidden history, retained predictor marks, and the shared scratch arena
before sizing the target KV pool. Enabling MTP therefore reduces the available
target pool rather than allocating predictor state outside the budget.

Draft/verify/accept and rollback use the shared scheduler. Prompt-end and
turn-end snapshots retain all predictor stages and their canonical frontiers.
A drafting request reuses only predictor-complete ancestors; target-only
requests can still reuse target-only snapshots. Restoring a snapshot must
preserve target KV/CSA, predictor histories and cursors exactly.

## Launch

Start with `examples/configs/step5-fp8-max.config` or
`examples/configs/step5-nvfp4-min.config`. Replace the sample fabric endpoints
and select a sealed WIP slot containing `serve-step5`. Bundled PE vision defaults
to automatic joint admission (Spark-first with RTX fallback) whenever the KV
pool still meets `KV_MIN_TOKENS`; `VISION=off`
explicitly selects text-only operation. One or two coordinator GPUs execute
every supported expert format; actual memory admission determines fit, not
card name or a 32 GiB policy floor. NVFP4 remains supported wherever native
packages are exported, including TP6. Pool sizing is automatic with a 256K
minimum request context. Native whole-layer RTX expert onboarding on one GPU
is unimplemented (#2429); native experts stay on Sparks there. EXL3 supports
whole-layer onboarding and two GPUs support native TP2 onboarding.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v5.1 WIP | 2026-11-13 | MC original min/max Release smoke, max K8/K6 cards and published Standard fidelity (#1906). | [Original min](../../benchmarks/step5/2026-11-13-smoke-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-min-cand-r1/report.svg) |
| v6 | 2026-11-19 | Published benchmarks | <a href="../../benchmarks/step5/2026-11-19-smoke-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-min/report.svg"><img src="../../benchmarks/step5/2026-11-19-smoke-step-5-preview-exl3-k4-mc-1rtx-3spark-step5-mc-original-min/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-MC (exl3-k4+exl3-k6) (1× RTX)"></a> <a href="../../benchmarks/step5/2026-11-19-smoke-step-5-preview-exl3-k4-mc-2rtx-6spark-step5-mc-original-max/report.svg"><img src="../../benchmarks/step5/2026-11-19-smoke-step-5-preview-exl3-k4-mc-2rtx-6spark-step5-mc-original-max/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-MC (exl3-k4+exl3-k6) (2× RTX)"></a> |
| v5 | 2026-11-10 | Published benchmarks | <a href="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-1rtx-3spark-step5-exl3-k6core-min/report.svg"><img src="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-1rtx-3spark-step5-exl3-k6core-min/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-Core-K6 (exl3-k4+exl3-k6) (1× RTX)"></a> <a href="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-2rtx-6spark-step5-exl3-k6core-max/report.svg"><img src="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-2rtx-6spark-step5-exl3-k6core-max/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-Core-K6 (exl3-k4+exl3-k6) (2× RTX)"></a> |
