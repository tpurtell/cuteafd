# How releases are measured

## Cards

Each model and quant gets a basic profile on two reference layouts: the
natural minimum (1× RTX + the fewest Sparks it fits) and the maximum
(2× RTX + 4 or 6 Sparks). A third column covers a 32 GB card (RTX 5090).

Release smoke measures text C1 decode, warmed C8 aggregate code throughput,
cold ~8K prefill, quick fidelity against the family golden reference, exact
prefix-cache restore and lossless speculation. It is not a full multimodal,
full-context or C16 qualification. Cards are single launches, not A/B
measurements: compare cells across releases only with matched-prompt runs
(`scripts/bench/wip-cards.py --interleave --matched-prompts`, see
[WIP hardware cards](wip-cards.md)), because speculative decode speed depends
on the prompt. A release regression is C1 more than 3% slower in every
matched pair, or a median more than 5% slower.

## The 5090 column

Unless a cell is marked real, it is a memory-only simulation:
**simulated 5090: RTX PRO 6000 (188 SMs) capped at 31.8 GiB**. SM count
(188 vs 170), L2, clocks and power are not emulated, so speed is indicative and
likely optimistic. Real 5090 cards replace these cells when they arrive;
simulations stay in history. Planner-only no-fit cells link the planner's
reason and contain no performance.

## Context

Memory fit, compiled extent and the admitted pool are distinct. GLM, GLM Flash,
V4 and V4.1 compile a 1,048,576-token index extent; Qwen's is clamped to its
checkpoint's 262,144. The admitted pool can reduce effective context: cards
report it, and values below the 262,144-token agentic floor are listed as
findings, never hidden by enlarging pools.

## v2.0.0 matrix

26 natural-minimum/maximum cells: 23 pass every check, the two V4.1 NVFP4 cells
fail fidelity (not recommended) and the V4 Pro EXL3 minimum fails fidelity at
KL 0.0605. All three required spot checks pass, as do the GLM draft-kernel
selftests on SM120 and SM121. The simulated-5090 column has nine passing cells,
one V4.1 NVFP4 fidelity FAIL and four planner-only no-fit cells. The Qwen EXL3
1× RTX row also lists a separate no-Spark card with vision on RTX0 at a fixed
262,144-token pool. Vision state per cell is in the family pages.

rc3 (promoted unchanged to v2.0.0) changed the Spark transport every Spark
family shares, so rc2 and rc3 were compared on matched prompts, three
interleaved pairs per card: MiMo V2.6 Flash minimum, GLM 5.3 EXL3 maximum,
V4 Flash maximum and GLM Flash tr3 maximum. No card regressed: paired-median C1
changed +0.2%, +2.9%, +0.1% and −1.6%; decode with speculation off +0.2%,
+0.5%, 0.0% and −1.5%. Below the regression threshold, and not established
changes: C8 −3.9% on GLM 5.3 and −3.8% on GLM Flash tr3 (MiMo and V4 flat), and
V4 Flash 8K prefill −2.6%.

MiMo V2.6 Flash MOPD replaced the legacy Flash rows; its
[text-only bring-up qualification](models/mimo_v2.md#flash-mopd-qualification-2026-10-05)
keeps its original conditions.
