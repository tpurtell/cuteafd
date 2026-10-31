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
place; serving is text only until the bundled towers are bound (#936).

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

Replies are BF16 for a step's narrow segments (up to
`--expert-partial-bf16-max-rows`, 64 by default: decode and verify stay
lossless) and NVFP4 (or FP8, `--expert-partial-wide-format`) for wider
prefill segments. Each endpoint's frame is sized from the admitted worst
reply (NVFP4 91.5 MiB, FP8 117.8 MiB per frame).

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
shared prefill pipeline. Packed prefill is not supported yet (#923).

## Chat encoding

The checkpoint's own template renders the prompt (byte-identical to
transformers on the goldens). The model replies in recipient-addressed
messages: `to=self` is private reasoning, `to=commentary` and `to=user` are
visible content, any other recipient is a tool call in an
`<atem:function_calls>` block. `reasoning_effort` maps onto the template's
`low`/`medium`/`high`/`xhigh` (`minimal` to low, `max` to xhigh); turning
thinking off is refused, as the template always reasons. Strict or required
tool calls, tool-decoding assistance and structured `response_format` run
under a native grammar (`muse_tool_schema`): reasoning first, then the calls
or a `to=user` answer in the requested format.

## Layouts

`cuteafd plan --layout` on the header-only EXL3 K4 + Core-K6 fixture
(RTX PRO 6000, default Spark budget, no host prefix tier):

| Layout | KV pool | RTX expert layers |
|---|---|---|
| 2 RTX + 4..8 Sparks | 2.10M (the 2M target) | automatic (1 here; 3 with the host tier) |
| 2 RTX + 3 Sparks | refused | |
| 1 RTX + 4 Sparks | 814K | automatic (0) |
| 1 RTX + 6 Sparks | 805K | automatic (0) |
| 1 RTX + 8 Sparks | 796K | automatic (0) |
| any + 2 Sparks | refused | |

One RTX admits the pool below the 1M compiled context, so an automatic
context is clamped to it. KV is 4 KiB per token per full layer (BF16),
64 KiB per token over the 16 full layers; sliding layers keep a
per-sequence ring of the window plus one rollback step.

## Speculation

Two DFlash2 drafters; the default is the one that delivers the most tokens
per second (to be measured on the first hardware window).

| Drafter | Geometry | Drafter attention |
|---|---|---|
| `RadixArk/Muse-Spark-Open-DFlash2` (46c90f41) | 5 layers, 64 query / 8 KV heads of 128, taps 1, 16, 30, 44, 59 | shared d128 |
| `nvidia/Muse-Spark-Open-816B-A42B-DFlash2` (22e2669e) | 5 layers, 128 query / 16 KV heads of 64, taps 3, 19, 31, 47, 59 | native `_d64` |

`--draft SNAPSHOT` opts in. The drafter runs on the last coordinator GPU,
beside the target head. Neither carries a head: both read the target's exact
head path (K6 head, trim, softcap with the target's output multiplier) on
their own charged scratch, and a drafter trained against another softcap or
multiplier is refused. Taps are the target's hidden rows after the tapped
layers; a prefill call commits its last 2048 rows (the drafter's window) as
context.

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

| Date | Release | Change | Quality / performance |
|---|---|---|---|
