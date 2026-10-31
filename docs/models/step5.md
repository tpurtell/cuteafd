# Step-5

Step-5-Preview serves text from its standard Hugging Face checkpoint headers.
The release layouts are official FP8 on two RTX PRO 6000 GPUs plus six Sparks,
and official NVFP4 on one RTX PRO 6000 plus four Sparks. NVFP4 experts
use the checkpoint's calibrated W4A4 math, including decode.

## EXL3 K4 + Core K6

`wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6` is quantized from the BF16
release. Its K4 routed experts (K6 for the last three routed layers, 88-90) run
on the Sparks from the shared `exl3-step5-k46` package, with Step's own
post-SiLU clamp at 7. The target attention q/k/v/o, shared and dense FFNs, and
lm_head are EXL3 K6 and run natively on the RTX through the shared dense
trellis programs. Gate and up stay separate projections, and the head-split
o_proj reduces its F32 partials before one output rotation. Head-wise
attention gates, the sparse indexer, routers, norms, embeddings and the vision
tower stay BF16/F32 exactly as shipped. The three MTP stages are K4
throughout, including each stage's own vocabulary head.

Everything is read from `config.json` and the safetensors headers;
`quantize_config.json` is only cross-checked. `cuteafd plan` finds the
fit: the minimum is 1 RTX PRO 6000 + 3 Sparks, and the maximum is 2 RTX PRO
6000 + 6 Sparks. Routed layers are onboarded to the RTX by default
(`RTX_EXPERT_LAYERS` unset is pool-first auto). Once the 2M KV pool fits,
whole routed layers run on the entry RTX from the `rtx-tp1` package; on the
maximum card that is about 20 layers. Start from
`examples/configs/step5-exl3-k6core-{min,max}.config`.

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
