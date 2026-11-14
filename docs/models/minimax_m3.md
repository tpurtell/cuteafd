# MiniMax M3.1 Flash

Native MiniMax M3.1 Flash serves through the shared scheduler, continuous
prefill executor, Spark exchange, exact prefix cache and device token selector.
Qualified layouts are one RTX PRO 6000 with three Sparks or two RTX PRO 6000
with four Sparks, using the checkpoint's
MXFP8 dense core, NVFP4 W4A4 routed experts and FP8 MSA KV/index storage.

## Launch

Use `cuteafd serve --family minimax_m3` or `./run.sh` with a build containing
MiniMax exports. The existing [minimum config](../../examples/configs/minimax-m31-flash-nvfp4-min.config)
has the qualified geometry; its old WIP-only comments are historical. Set your
own endpoints and use images built from the qualified source, not the config's
older image tags. Shared settings are in [MiniMax Serving](../options.md#minimax-serving).
Keep `SPECULATOR=off`, `VISION=off` and `AUDIO=off`: bundled drafter/media
integration remains pending [#1856](https://github.com/tpurtell/cuteafd/issues/1856).

`ATTENTION_PLACEMENT=auto|heads` selects whole ownership on one RTX. On two RTX,
`auto|layers` selects contiguous attention with qualified FP32 split FFNs
([#2101](https://github.com/tpurtell/cuteafd/pull/2101)). Split sections remain eager,
without a graph-prewarm claim. Explicit two-RTX narrow heads remain
planning-only because their native kernels are unqualified (#1856). Context
placement is not implemented. Planning inventories are not serving qualification.

## Qualification

The [2026-11-14 trace-free basic card](../../benchmarks/minimax_m3/2026-11-14-custom-minimax-m3-1-flash-1rtx-3spark-minimax-flash-min-cand-r1/report.svg)
at `3a70a3768` used one RTX PRO 6000 + three Sparks, boot C16, automatic pool
and host prefix cache, and the default graph reserve. Generated aggregate
fidelity passes: KL 0.0265743, top-1 94.7977% (2076 generated rows). Context
and per-window scores are informational under the approved MiniMax scoring policy.
Prompt/turn restores (9743/9338 tokens) and cold recomputation are byte-identical.
Tool-call parsing/re-render and C1/C4 greedy identity also pass.

Basic C1/C8 are 18.42/70.37 tok/s; warmed C1/C16 sweep is 18.42/138.12 tok/s;
8K prefill is 2313 tok/s (2166 with tail). Ready time is 82.55 s; admitted
pool is 1,146,880 tokens. RTX ready and first-request reconciliation both differ
by +925,409 bytes, inside half of the 100 MiB release padding. The measured CUDA
baseline is one named resident owner; the 40 MiB graph arena has 44 MiB reserve
slack. No request errors or steady-state decode captures/evictions were observed.

Thermal conditions: basic card has 1.013 s power-cap slowdown and zero thermal
slowdown, busy SM mean 2828.6 MHz, max 62 C, minimum T.Limit headroom 32 C,
and reported cap 425-450 W. C1/C16 sweeps have zero power-cap/thermal slowdown,
busy SM means 2827/2815.4 MHz, max 62/60 C and headroom 31/35 C. Trace and
speculation instrumentation are disabled; these are not comparisons to another engine.

The canonical [Release smoke](../../benchmarks/minimax_m3/2026-11-14-smoke-minimax-m3-1-flash-1rtx-3spark-minimax-flash-min-cand-r1/report.svg)
on the same immutable seal supplies the matrix row: C1/C8/C16
18.23/70.25/101.08 tok/s and 8K prefill 2314 tok/s. It repeats the fidelity,
exact restore and tool checks successfully. Its basic thermal window records
2.003 s power-cap slowdown, zero thermal slowdown, busy SM mean 2829.6 MHz,
max 62 C and minimum T.Limit headroom 32 C (425-450 W reported cap).
The custom qualification sweep and canonical basic concurrency are different
workloads, not a speed comparison.

## Two-RTX Qualification

The [mixed-traffic card](../../benchmarks/minimax_m3/2026-11-14-custom-minimax-m3-1-flash-2rtx-4spark-minimax-flash-min-cand-r1/report.svg)
at `60831478b` passes generated fidelity (KL 0.0245256, top-1 95.2312%), exact
prompt/turn restores (9741/9340 tokens), cold recomputation, tool re-render,
greedy C1/C4 identity and 8K context-load C1/C16 traffic. Native decode-shaped
FP32 golden separately passes KL 0.0245271/top-1 94.9422%, zero missing/nonfinite.

Mixed-card C1/C8 are 18.58/83.31 tok/s, C1/C16 sweep 18.53/171.76 tok/s,
8K prefill 2210 tok/s, pool 2,097,152 tokens and ready time 69.21 s. RTX0/1
ready and first-request memory deltas are +8,260,411/+5,237,371 bytes, within
half of release padding, with graph slack zero. Both GPUs record zero
power-cap/thermal slowdown at 425 W each; basic busy SM means 2805.5/2851.2 MHz,
max temperatures 56/62 C and minimum T.Limit headroom 37/34 C.

The canonical [two-RTX Release smoke](../../benchmarks/minimax_m3/2026-11-14-smoke-minimax-m3-1-flash-2rtx-4spark-minimax-flash-min-cand-r1/report.svg)
on the same seal supplies the matrix row: C1/C8/C16 18.54/83.43/125.91 tok/s,
8K prefill 2227 tok/s (tail 2060), pool 2,097,152 and readiness 69.61 s.
Fidelity, exact restore, tool and greedy checks pass. Both GPUs record zero
power-cap/thermal slowdown, busy SM means 2804.1/2850.7 MHz, max56/61 C and
minimum T.Limit headroom37/33 C at reported420-425/425 W caps.

## Changelog

| Date | Change | Evidence |
| --- | --- | --- |
| 2026-11-14 | Qualify public one-RTX native serving, exact prefix restores, tool grammar and named baseline/graph memory owners. | Basic card at `3a70a3768`, generated aggregate fidelity pass; two-RTX and narrow heads remain gated by #1856. |
| 2026-11-14 | Qualify public two-RTX contiguous attention and FP32 split FFNs; eager split sections, native TP2 experts and exact shared prefix state. | Golden and mixed card at `60831478b`; narrow heads/media/drafts remain pending #1856. |
