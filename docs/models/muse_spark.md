# Muse Spark

`meta-models/Muse-Spark-Open-816B-A42B` (`MuseSparkForConditionalGeneration`,
text `muse_spark_text`) serves text from an EXL3 publication: K4 routed
experts split per expert, a K6 MCG core (attention q/k/v/o, the MoE's
`pre_expert_proj`/`post_expert_proj`, the LM head) and the rest at the
checkpoint's precision. The BF16 release fuses each layer's 256 experts into
one tensor and has no expert package; `cuteafd plan` names the publication to
serve instead.

Status: under qualification. The plan, the coordinator engine on one RTX or a
head-split pair, the Spark expert path, RTX onboarding, the prefix cache, the
chat encoding with constrained tool calls and both DFlash2 drafters are in
place. The starting configs also serve the checkpoint's bundled vision and
PCM audio towers by default; qualification and caveats are recorded below
and on #923.

## Attention and core views

The `-HQA` checkpoints are withdrawn from Hugging Face. They're being replaced by multi-core (MC) checkpoints that bundle the original dense core with selectable K8 and K6 cores. Until those publish, serve the Core-K6 checkpoint above.

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
towers (`VISION=auto`, `AUDIO=auto`); set either to `off` to drop it. Both
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
| v5 | 2026-11-10 | Published benchmarks | <a href="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-1rtx-4spark-muse-spark-exl3-min/report.svg"><img src="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-1rtx-4spark-muse-spark-exl3-min/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-Core-K6 (exl3-k4) (1× RTX)"></a> <a href="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-2rtx-6spark-muse-spark-exl3-max/report.svg"><img src="../../benchmarks/muse_spark/2026-11-10-smoke-muse-spark-open-exl3-k4-core-k6-2rtx-6spark-muse-spark-exl3-max/card.svg" width="360" alt="Muse-Spark-Open-EXL3-K4-Core-K6 (exl3-k4) (2× RTX)"></a> |
