# MiniMax M3.1 Flash

Native MiniMax M3.1 Flash serves through the shared scheduler, continuous
prefill executor, Spark exchange, exact prefix cache and device token selector.
The qualified layout is one RTX PRO 6000 with three Sparks, using the checkpoint's
MXFP8 dense core, NVFP4 W4A4 routed experts and FP8 MSA KV/index storage.

## Launch

Use `cuteafd serve --family minimax_m3` or `./run.sh` with a build containing
MiniMax exports. The existing [minimum config](../../examples/configs/minimax-m31-flash-nvfp4-min.config)
has the qualified geometry; its old WIP-only comments are historical. Set your
own endpoints and use images built from the qualified source, not the config's
older image tags. Shared settings are in [MiniMax Serving](../options.md#minimax-serving).
Keep `SPECULATOR=off`, `VISION=off` and `AUDIO=off`: bundled drafter/media
integration remains pending [#1856](https://github.com/tpurtell/cuteafd/issues/1856).

`ATTENTION_PLACEMENT=auto|heads` selects whole ownership on one RTX. Two-RTX
contiguous-attention/FP32 split FFN serving is tracked by
[#2101](https://github.com/tpurtell/cuteafd/pull/2101); it remains planning-only
until its serving card passes. Explicit two-RTX narrow heads also remain
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

## Changelog

| Date | Change | Evidence |
| --- | --- | --- |
| 2026-11-14 | Qualify public one-RTX native serving, exact prefix restores, tool grammar and named baseline/graph memory owners. | Basic card at `3a70a3768`, generated aggregate fidelity pass; two-RTX and narrow heads remain gated by #1856. |
