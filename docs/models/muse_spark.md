# Muse Spark

The v7 MC cards use K8 compact on 1 RTX + 4 Sparks and 2 RTX + 6 Sparks.
V7 drops the original, K6 compact and FP8 compact comparison (TJ 2026-11-28); its last cards stay pinned in the [comparison matrix](../../scripts/bench/release-smoke-v7-mc-compact.json)'s retired list.

Automatic KV sizing uses the shared 2M-token target on every card, including
32 GB / RTX 5090. Placement and expert onboarding are fixed at that target;
the pool then grows into remaining admitted memory, within native encoding
limits. `KV_MAX_TOKENS` is unset by default and can explicitly cap growth.
See [memory policy](../options.md#memory-and-pools).

Target KV storage accepts `KV_CACHE=bf16` (the current default) or experimental
`KV_CACHE=fp8`. FP8 stores native E4M3 K/V plus independent FP32 amax/448
scales for every token and KV head, for full and sliding attention. One aligned
record contains K, V, K scales and V scales: 2176 B/token/layer on one RTX or
1088 B/rank with two RTX, including scales and without padding. BF16 records
use 4096/2048 B respectively. Exact prefix marks retain the complete records;
BF16 and FP8 prefix namespaces never mix. `DRAFT_KV_CACHE` remains independent.
FP8 remains opt-in: the measured pool benefit does not offset its paired
quality cost, and the max-card C1 speed also falls below the default bar.

### FP8 KV qualification

Measured on dev `5e28a1d30`, fork `87962497`, K4 MC `CORE_VIEW=original`,
RTX PRO 6000, 2026-11-16. Quality used speculation off and the same sealed
WIP build for both arms. Standard max uses 2 RTX + 6 Sparks; Quick min uses
1 RTX + 4 Sparks. Results below score generated rows only; every arm passes
the absolute family bars (KL <= 0.05, top-1 >= 93%) with no missing or
non-finite rows.

| Quality path | Generated rows | BF16 KL | FP8 KL | KL change | BF16 top-1 | FP8 top-1 | Top-1 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard max decode | 10,629 | 0.009683 | 0.016891 | +0.007208 | 97.149% | 96.594% | -0.555 pp |
| Standard max prefill | 10,475 | 0.008539 | 0.014792 | +0.006253 | 97.652% | 96.936% | -0.716 pp |
| Quick min | 1,832 | 0.006365 | 0.012403 | +0.006038 | 97.271% | 97.162% | -0.109 pp |

Both Standard paths fail the paired default-quality limits (+0.003 KL and
-0.3 percentage points top-1). BF16 therefore stays the default.

The min-card Quick pair, with speculation off and 16 active slots, admits
510,272 BF16 versus 1,056,768 FP8 pool tokens (2.071x); served context rises
from 510,272 to the full 1,048,576. This is a speculation-off capacity
measurement, not a default-DFlash2 min-card speed result.

A separate quiet max-card pair uses default NVIDIA DFlash2 and one warmed
launch per arm, with one untimed batch at each concurrency. C1 emits
70.81 -> 65.08 tok/s (0.919x); C16 aggregate emits 253.03 -> 257.81 tok/s
(1.019x). Automatic pool-first admission grows the pool from 2,009,216 to
2,097,152 tokens and onboards 0 -> 9 RTX expert layers. No power-cap,
thermal or hardware slowdown is sampled in either timed window. GPU1 busy
SM clocks are 2854.7 -> 2854.7 MHz at C1 and 2847.3 -> 2845.2 MHz at C16;
maximum temperature is 61 C and minimum T.Limit headroom is 36 C. Powerd
limits are 425 W per card except FP8 C1's transient GPU0 435-445 W / GPU1
400-410 W distribution; neither card hits its cap. These are one-session
observations, not a repeat campaign. Incidental timings from the quality
launches are excluded because those runs experienced host page reclaim.

Receipts are retained under `~/.cache/cuteafd/builds/v5x-muse-kv-quality-max/`,
`v5x-muse-kv-quality-min/` and `v5x-muse-kv-perf-max/`. Min-card speed and 64K
prefill were deliberately not run after the paired quality default gate failed.

`meta-models/Muse-Spark-Open-816B-A42B` (`MuseSparkForConditionalGeneration`,
text `muse_spark_text`) serves text from an MC EXL3 publication: K4 routed
experts split per expert, the original BF16 core by default, and complete
K8/K6 MCG alternatives (attention q/k/v/o, the MoE's
`pre_expert_proj`/`post_expert_proj`, the LM head). The rest retains the
checkpoint's precision. The BF16 release fuses each layer's 256 experts into
one tensor and has no expert package; `cuteafd plan` names the publication to
serve instead.

Status: under qualification. The plan, the coordinator engine on one RTX or a
head-split pair, the Spark expert path, RTX onboarding, the prefix cache, the
chat encoding with constrained tool calls and both DFlash2 drafters are in
place. The starting configs also serve the checkpoint's bundled vision and
PCM audio towers by default; qualification and caveats are recorded below
and on #923.

## Core views

The old `-HQA` checkpoint was withdrawn from Hugging Face. Use the published MC checkpoint for selectable core views.

`wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC` provides the original BF16 core
and complete `k8-compact` / `k6-compact` cores at revision
`cc5dc2537083177e3f02ea0b2d9dfc179d40425c`. Select with `CORE_VIEW` (default
`original`) or `--core-view`; see the [MC contract](../mc-checkpoints.md).
The [min](../../examples/configs/muse-spark-exl3-k4-mc-min.config) and
[max](../../examples/configs/muse-spark-exl3-k4-mc-max.config) configs select
`CORE_VIEW=original`. The original path retains native attention, MoE
frontend/backend and vocabulary head weights. Compact widths have distinct
dispatch and exact memory charges. Their max-card comparison entries live in
`scripts/bench/release-smoke-v51-mc-core-comparison.json`; they do not add
release-table rows. MC original Release smoke passes on the min/max cards
(build `ac64e40ee`, 2026-11-14): C1 code 50.43/71.57 tok/s, 8K prefill
2662.70/3361.93 tok/s, generated KL 0.00636/0.00709 and top-1
97.27%/97.60%. Pools are 361,088/2,009,152 tokens (max context 1M on max).
No thermal slowdown occurred; power-cap slowdown was 43.93 s on min and
6.99/5.98 s on max's two GPUs during the baseline window (850 W shared budget).
Busy SM clocks were 2559 MHz on min and 2727/2832 MHz on max; peak temperatures
67 C and 64/70 C, minimum T.Limit headroom 23 C and 30/27 C. These are new
MC original measurements, not an identical-configuration A/B against Core-K6.
Prompt-end and turn-end restores compared byte-identically, as did cold
recompute; the smoke's short fresh-continuation shape remains explicitly
uncovered (informational), rather than claiming a complete prefix proof.
The standalone Core-K6
[min](../../examples/configs/muse-spark-exl3-min.config) and
[max](../../examples/configs/muse-spark-exl3-max.config) configs remain supported;
only their regular benchmark cards retire.
`fp8-compact` is not executable for this complete MC core yet: its selected
vocabulary head and pre/post-expert projections refuse by role before payload
reads (#1796). The existing attention-only executor cannot silently skip them.
Towers and routed experts are shared, not converted with the core.

### Max-card core comparison

Supplementary Release smoke, 2 RTX PRO 6000 + 6 Sparks, the same MC revision
and DFlash2 revision as above. Both cards reuse clean build `ac64e40ee` and
its matching launcher; one warmed launch per view, measured on different days
(original Nov 14, K8 Nov 15), not an interleaved A/B. Automatic placement stays
on: the smaller K8 core admits one RTX expert layer instead of zero and fills
the 2M-token target pool. Rates are emitted tok/s; card C16 uses distinct
prompts, not the separate code-decode sweep.

| Core view | Card C1 code | Card distinct C16 aggregate | Card 8K prefill | Generated KL / top-1 | KV pool tokens | RTX expert layers | Report |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `original` | 71.57 | 117.35 | 3361.93 | 0.007085 / 97.60% | 2,009,152 | 0 | [Release smoke](../../benchmarks/muse_spark/2026-11-14-smoke-muse-spark-open-exl3-k4-mc-2rtx-6spark-muse-spark-exl3-k4-mc-max-cand-r1/report.svg) |
| `k8-compact` | 75.96 | 119.02 | 3225.67 | 0.006341 / 97.60% | 2,097,152 | 1 | [Release smoke](../../benchmarks/muse_spark/2026-11-15-smoke-muse-spark-open-exl3-k4-mc-2rtx-6spark-muse-mc-k8-compact-max-k8-r1/report.svg) |

K8's **separate code-decode sweep** measures C1 61.00, C16 aggregate 257.88
and post-C16 C1 75.21 tok/s, with one untimed warm batch at each concurrency.
The original card has no equivalent sweep or post-C1 measurement. The 257.88
sweep result must not be compared with either card's distinct-C16 column.
Quick fidelity passes over 1,832 generated rows / 4,096 total rows, and lossless
DFlash2 checks pass. Prompt-end/turn-end and cold-recompute byte proofs pass;
the short fresh-continuation shape remains uncovered, as on original.

| Baseline-window telemetry (GPU0 / GPU1) | `original` | `k8-compact` |
| --- | --- | --- |
| End-window NVML free memory, MiB | 926.25 / 1951.25 | 1054.25 / 3287.25 |
| Power-cap slowdown, s | 6.985 / 5.984 | 6.002 / 8.987 |
| Software / hardware thermal slowdown, s | 0 / 0 on both GPUs | 0 / 0 on both GPUs |
| Busy SM clock, MHz | 2727 / 2832 | 2671 / 2815 |
| Peak temperature, C | 64 / 70 | 65 / 72 |
| Minimum T.Limit headroom, C | 30 / 27 | 30 / 25 |

Free memory is a point-in-time sample after the baseline, not minimum headroom
or an allocation ledger. Both runs used gpu-powerd's 850 W shared budget;
original limits ranged 400-450 W, K8 GPU0 425-435 W and GPU1 410-425 W.
The K8 sweep had zero power-cap or thermal slowdown; its GPU1 busy clock was
2841 MHz (GPU0 had no samples meeting the busy threshold).

Quiet-host preflight passed before enqueue and at fair-queue grant: no builds
or gates, CPU and memory PSI avg10 both zero. During the first timed K8 card
C1, memory PSI rose to 0.45 (prose C1 0.24), above the requested 0.1 quiet bar;
CPU PSI stayed zero and page-scan rate was zero. Card C16, 8K prefill and all
separate sweep windows recorded zero CPU/memory PSI. Treat the observed C1
+6.1%, distinct-C16 +1.4% and prefill -4.1% as one-session observations,
not fully quiet, statistically established speed changes. No rerun was taken.

## Architecture

Every one of the 62 layers is a mixture of experts; there is no dense FFN and
no shared expert.

- **Attention:** GQA, 128 query and 16 KV heads of 64. Three sliding layers
  (window 2048, RoPE theta 5e5) per full layer; the full layers (1, 5, ..., 61)
  are NoPE. Queries and keys are RMS-normalized per head without a weight
  (queries then scaled by `qk_scale_factor` 5.473), the attention output is
  normalized per head and gated by `sigmoid(gate_proj(x))` before `o_proj`.
- **Residuals:** learned residual gates replace both residual adds
  (`alpha * residual + beta * x`, temperature 0.3), with norms around each
  sublayer.
- **MoE:** the router reads the normalized residual (FP32 logits times
  `output_multiplier`, sigmoid, bias for selection only, top-8 of 256,
  renormalized). The experts run at a 4096-wide stream: `pre_expert_proj`
  (8192 to 4096) and a norm, the experts, then `post_expert_proj` back.
  **Each expert's output is RMS-normalized before its routing weight.**
- **Head:** untied, `20 * tanh(logits * 0.1768 / 20)`, vocabulary 202,048
  (the K6 head is stored at 202,112 rows and trimmed before the softcap).
- **Context:** 1M (`max_position_embeddings`).

Every `(1 + w)` in the centered and input-scaled norms is a BF16 add, as the
reference module computes it on the BF16 parameter.

## Experts: tensor-parallel partials, combined on the coordinator

Spark ranks hold every expert's slice of the intermediate (TP over the
shared EXL3 Spark worker, worlds 2..8 from the package range data). Because
each expert's output is normalized over its whole width before its routing
weight, the ranks cannot sum: each replies its **per-expert partials** for
every routed (row, slot), and the coordinator's combine
(`muse_tp_combine_*`) sums the ranks per route, applies the layer's
`post_expert_norm` and the routing weight, and reduces the eight routes.
The combine reads every rank's reply in place through an `i64` table of
region addresses (`rank_regions`).

With the phase keys omitted, replies are BF16 for narrow prefill segments
(up to `--expert-partial-bf16-max-rows`, 64 by default) and NVFP4 for wider
prefill segments; decode and verify stay BF16. The independent shared
`--spark-prefill-reply-format` and `--spark-decode-reply-format` selections
accept BF16, FP8 or NVFP4 and apply explicitly at every row count. Each
endpoint's frame is sized from the admitted worst reply across both phases.

**RTX onboarding** is on by default: an onboarded layer runs its experts on
the coordinator from the image's EXL3 package (`exl3-muse-k<tiers>/rtx-tp2`
on a pair, `rtx-tp1` on one GPU). On a pair each GPU holds half of every
onboarded expert's intermediate; both swap the expert-input halves, run the
shared EXL3 core with the partial-reply epilogue (all rows BF16), swap the
partials, and combine with ranks 2 (ranks 1 on one GPU). Automatic placement
onboards whatever memory allows after the KV pool target; an image without
the package refuses with a named reason rather than serving the layers from
the Sparks.

## Coordinator

A head-split RTX pair halves the attention heads (and the K/V records) per
GPU, splits `pre_expert_proj` over its outputs and `post_expert_proj` over
its inputs (single residency), and exchanges over peer access. One RTX runs
the whole layer. The embedding stays on the entry GPU (rank 1 gathers over
peer access); the head runs on the last rank.

The prefix cache restores exact frontiers (full-layer KV records and the
sliding rings), with the shared host tier; prefill runs as row units on the
shared prefill pipeline. Packed prefill is enabled by default (`unrestricted`);
`exact` falls back to singleton units for EXL3 when row invariance is not
available.

## Chat encoding

The checkpoint's own template renders the prompt (byte-identical to
transformers on the goldens). The model replies in recipient-addressed
messages: `to=self` is private reasoning, `to=commentary` and `to=user` are
visible content, any other recipient is a tool call in an
`<atem:function_calls>` block. `reasoning_effort` maps onto the template's
`low`/`medium`/`high`/`xhigh` (`minimal` to low, `max` to xhigh). Thinking
off pre-fills the final `to=user` recipient, skips private reasoning and
ignores reasoning effort; requests with tools are refused in this mode.
With thinking on, strict or required tool calls, tool-decoding assistance
and structured `response_format` run under a native grammar
(`muse_tool_schema`): reasoning first, then the calls or a `to=user` answer
in the requested format. Thinking-off structured output uses the ordinary
response grammar directly on the final answer.

## Layouts

Use `cuteafd plan --layout` with the actual checkpoint, bundled towers,
NVIDIA drafter and image's program manifest to resolve the KV pool and RTX
onboarding. The qualified cards are 1 RTX + 4 Sparks and 2 RTX + 6 Sparks;
the max card reaches the 2M pool target, while the min card's pool is
memory-limited. Header-only fixture estimates omit runtime reservations and
are not served-card capacities.

One RTX admits the pool below the 1M compiled context, so an automatic
context is clamped to it. KV is 4 KiB per token per full layer (BF16),
64 KiB per token over the 16 full layers; sliding layers keep a
per-sequence ring of the window plus one rollback step.

## Speculation

The launcher and planner default to NVIDIA DFlash2, pinned to
`22e2669e4c60689f9bff0749a34d9d8e59ce7cd6`. It is chosen on C4 emitted
throughput; baseline C1 ties RadixArk. Both pass the lossless set and exact
prefix-cache checks; NVIDIA is qualified on both the 1 RTX + 4 Spark and
2 RTX + 6 Spark cards. Set
`SPECULATOR=off` to disable drafting; an explicit model/revision overrides
the default.

| Drafter | Geometry | Drafter attention |
|---|---|---|
| `RadixArk/Muse-Spark-Open-DFlash2` (46c90f41) | 5 layers, 64 query / 8 KV heads of 128, taps 1, 16, 30, 44, 59 | shared d128 |
| `nvidia/Muse-Spark-Open-816B-A42B-DFlash2` (22e2669e) | 5 layers, 128 query / 16 KV heads of 64, taps 3, 19, 31, 47, 59 | native `_d64` |

Direct `serve-muse` calls select a drafter with `--draft SNAPSHOT`.
The drafter runs on the last coordinator GPU,
beside the target head. Neither carries a head: both read the target's exact
head path (K6 head, trim, softcap with the target's output multiplier) on
their own charged scratch, and a drafter trained against another softcap or
multiplier is refused. Taps are the target's hidden rows after the tapped
layers; a prefill call commits its last 2048 rows (the drafter's window) as
context.

## Media

The starting configs serve the checkpoint's bundled vision and PCM audio
towers (`VISION=auto`, `AUDIO=auto`); set either to `off` to drop it. Automatic
vision is preserved whenever the joint layout still meets `KV_MIN_TOKENS`. Both
prefer Sparks in Auto mode (min: vision on rank 0, 4,860,356,960 B; audio on
rank 1, 246,586,368 B). If no Spark admits a tower and its scratch plus
headroom, Auto falls back to an RTX with enough memory. Explicit `--vision rtx`
or `--audio rtx` requests coordinator placement directly. Qualification is on #923:

- **Vision:** byte-exact metadata and the FP32 yardstick (public RMS ratio
  0.990); prompt and turn prefix-cache restores exact.
- **Audio:** the native tower drives the same LM to the official BF16 tower's
  output within KL 0.006–0.010 and top-1 96.9–97.9% on silence, one sample,
  a 30 s signal and speech (512 generated rows each); restores exact. Its
  cumulative tower-level silence check stays open (RMS ratio 1.65 against
  the official BF16 error); it does not show in served output.
- **Latency (min, first use):** image 5.86 s, speech 11–15 ms.

## Numerics references

- `python/reference/families/muse_spark/model.py`: the official transformers
  5.22.1 modules, hash-pinned; shared with the fidelity golden.
- `rust/crates/cuteafd-loader/src/families/muse_spark/reference.rs`: the bit
  contracts native kernels gate against (rounding points, canonical FP32
  reductions, tolerances).
- `rust/crates/cuteafd-daemon/src/families/muse_spark/reference.rs`: the CPU
  oracle of the layer chain; on a tiny checkpoint it matches the official
  model to a max logit difference of 0.014 with every top-1 equal.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v5.1 dev | 2026-11-14 | MC original BF16 core, native pre/post-expert bridges and role-opened classification | [1 RTX + 4 Sparks](../../benchmarks/muse_spark/2026-11-14-smoke-muse-spark-open-exl3-k4-mc-1rtx-4spark-muse-spark-exl3-k4-mc-min-cand-r1/report.svg), [2 RTX + 6 Sparks](../../benchmarks/muse_spark/2026-11-14-smoke-muse-spark-open-exl3-k4-mc-2rtx-6spark-muse-spark-exl3-k4-mc-max-cand-r1/report.svg): Release smoke quality/shared-prefix panels pass; prompt/turn restores byte-identical, short fresh continuation uncovered |
| v6 | 2026-11-19 | Published benchmarks | <a href="../../benchmarks/muse_spark/2026-11-19-smoke-muse-spark-open-exl3-k4-mc-1rtx-4spark-muse-spark-exl3-k4-mc-min/report.svg"><img src="../../benchmarks/muse_spark/2026-11-19-smoke-muse-spark-open-exl3-k4-mc-1rtx-4spark-muse-spark-exl3-k4-mc-min/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-MC (exl3-k4) (1× RTX)"></a> <a href="../../benchmarks/muse_spark/2026-11-19-smoke-muse-spark-open-exl3-k4-mc-2rtx-6spark-muse-spark-exl3-k4-mc-max/report.svg"><img src="../../benchmarks/muse_spark/2026-11-19-smoke-muse-spark-open-exl3-k4-mc-2rtx-6spark-muse-spark-exl3-k4-mc-max/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-MC (exl3-k4) (2× RTX)"></a> |
| v5 | 2026-11-10 | Published benchmarks | <a href="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-1rtx-4spark-muse-spark-exl3-min/report.svg"><img src="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-1rtx-4spark-muse-spark-exl3-min/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-Core-K6 (exl3-k4) (1× RTX)"></a> <a href="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-2rtx-6spark-muse-spark-exl3-max/report.svg"><img src="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-2rtx-6spark-muse-spark-exl3-max/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-Core-K6 (exl3-k4) (2× RTX)"></a> |
