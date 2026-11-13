# Step-5

Step-5-Preview serves text from its standard Hugging Face checkpoint headers.
The MC release cards target one RTX PRO 6000 plus three Sparks and two RTX
plus six Sparks. Official FP8 and NVFP4 remain supported; NVFP4 experts
use the checkpoint's calibrated W4A4 math, including decode.

## EXL3 K4 MC

`wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC` at
`4483a4bd874e8c360aec486ea72bd31d1ad7a247` is quantized from the BF16
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
tower stay BF16/F32 exactly as shipped. The three MTP stages are K4
throughout, including each stage's own vocabulary head.

Everything is read from `config.json` and the safetensors headers;
`quantize_config.json` is only cross-checked. `cuteafd plan` finds the
fit for each selected view. The card layouts are 1 RTX PRO 6000 + 3 Sparks
and 2 RTX PRO 6000 + 6 Sparks. Routed layers are onboarded to the RTX by default
(`RTX_EXPERT_LAYERS` unset is pool-first auto). Once the 2M KV pool fits,
one RTX runs whole layers from `rtx-tp1`; two RTX split each onboarded layer
TP2 through the shared pair and `rtx-tp2`, with F32 partial exchange. The
prior K6 96 GiB header-fixture maximum onboards 24 layers; its minimum prioritizes
its roughly 1.81M-token pool and onboards none. These are planner results,
not hardware qualification. Start from
the MC [min](../../examples/configs/step5-exl3-k4-mc-min.config) and
[max](../../examples/configs/step5-exl3-k4-mc-max.config) configs, which pin the MC
publication, revision and `CORE_VIEW=original`. The old K6-core paths remain
retired historical recipes. The release smoke matrix uses the MC configs;
MC cards and Standard fidelity
are pending [#1906](https://github.com/tpurtell/cuteafd/issues/1906).

## Core views

The published MC follows the [MC contract](../mc-checkpoints.md):
`CORE_VIEW=original` (default) restores the BF16 target core; `k8-compact`
and `k6-compact` select complete native trellis cores. Routed experts,
non-core tensors and the independently owned K4 MTP stages stay unchanged.
Plain checkpoints refuse compact selectors. Release cards select `original`;
max-layout compact comparisons use the separate
`scripts/bench/release-smoke-v51-mc-core-comparison.json` matrix.

`fp8-compact` is **not supported** for this MC. Step-5's block-FP8 executor
supports O projections only; the MC's complete core also selects q/k/v,
dense/shared FFNs and the head, so role validation refuses it by tensor name.
The comparison is `na`; no partial conversion is presented as a whole-core view.

Standard fidelity uses the published 32+32 set rooted at
`stepfun-ai/Step-5-Preview-BF16`, on the min card, for each supported view.
KL, top-1, generated top-1 and NLL remain pending; the routed-expert 3-6 bpw
tier requires generated-row KL <= 0.05 and top-1 >= 0.93. Track full-length
window capacity under [#1899](https://github.com/tpurtell/cuteafd/issues/1899)
and trellis-core mount reads under [#1849](https://github.com/tpurtell/cuteafd/issues/1849);
neither is fixed by switching publication rows.

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
and select a sealed WIP slot containing `serve-step5`. Keep media disabled;
Step-5's bundled vision execution is not yet qualified. Pool sizing remains
automatic, and admission refuses configurations below the 256K context floor.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v5 | 2026-11-10 | Published benchmarks | <a href="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-1rtx-3spark-step5-exl3-k6core-min/report.svg"><img src="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-1rtx-3spark-step5-exl3-k6core-min/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-Core-K6 (exl3-k4+exl3-k6) (1× RTX)"></a> <a href="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-2rtx-6spark-step5-exl3-k6core-max/report.svg"><img src="../../benchmarks/step5/2026-11-10-smoke-step-5-preview-exl3-k4-core-k6-2rtx-6spark-step5-exl3-k6core-max/card.svg" width="360" alt="Step-5-Preview-EXL3-K4-Core-K6 (exl3-k4+exl3-k6) (2× RTX)"></a> |
