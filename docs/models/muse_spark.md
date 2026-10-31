# Muse Spark

`meta-models/Muse-Spark-Open-816B-A42B` (`MuseSparkForConditionalGeneration`,
text `muse_spark_text`) serves text from an EXL3 publication: K4 routed
experts split per expert, a K6 MCG core (attention q/k/v/o, the MoE's
`pre_expert_proj`/`post_expert_proj`, the LM head) and the rest at the
checkpoint's precision. The BF16 release fuses each layer's 256 experts into
one tensor and has no expert package; `cuteafd plan` names the publication to
serve instead.

Status: planned. The plan, Spark expert residency, expert-parallel EXL3
exports and the CPU reference are in place; the coordinator engine is in
progress. Example configs land with the first served build.

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
- **Head:** untied, `20 * tanh(logits * 0.1768 / 20)`, vocabulary 202,048.
- **Context:** 1M (`max_position_embeddings`).

Every `(1 + w)` in the centered and input-scaled norms is a BF16 add, as the
reference module computes it on the BF16 parameter.

## Experts are whole on each rank

Because each expert's output is normalized over its whole width before the
routing weight, the routed experts are never sliced by intermediate. Spark
ranks hold whole experts, partitioned by a shared owner table (expert
`e` of layer `l` on rank `(e + l) % N`). Every rank receives every row and
the global routes, runs only the routes it owns (the others map to -1 in its
expert map and are skipped), applies each route's norm and weight, and returns
one partial sum; the coordinator adds the ranks as it does for tensor
parallel. The exchange and the reply size are those of TP.

The residency follows the checkpoint: a routed layer that stores
`mlp.experts.post_expert_norm.weight` is served as whole experts, with the
norm weight resident beside its experts; other checkpoints keep the balanced
intermediate split. RTX-onboarded layers are whole on one GPU, alternating
between the two GPUs.

At decode a token's eight routes land unevenly on the ranks (two of eight on
the busiest of six at best). Popularity-balanced ownership and replication of
hot experts in Spark headroom are the levers; they are measured on real
routes from the fidelity golden before either is committed.

## Layouts

`cuteafd plan --layout` (header-only EXL3 K4 + Core-K6 fixture, 100 GiB Spark
budget, RTX PRO 6000):

| Layout | Experts per Spark | KV pool | RTX expert layers |
|---|---|---|---|
| 2 RTX + 6 Sparks | 61.3 GiB | 2.10M | automatic |
| 2 RTX + 5 Sparks | ~73 GiB | 2.10M | automatic |
| 2 RTX + 4 Sparks | 92.0 GiB | 2.10M | automatic |
| 1 RTX + 4..6 Sparks | | ~835K | automatic |

KV is 4 KiB per token per full layer (BF16), 64 KiB per token over the 16 full
layers; sliding layers keep a per-sequence ring of the window plus one prefill
unit.

## Speculation

Two DFlash2 drafters exist; the default is the one that delivers the most
tokens per second.

| Drafter | Geometry | Shared DFlash2 kernels |
|---|---|---|
| `RadixArk/Muse-Spark-Open-DFlash2` (46c90f41) | 5 layers, 64 query / 8 KV heads of 128, taps 1, 16, 30, 44, 59 | fits |
| `nvidia/Muse-Spark-Open-816B-A42B-DFlash2` (22e2669e) | 5 layers, 128 query / 16 KV heads of 64, taps 3, 19, 31, 47, 59 | needs a head_dim-64 drafter attention |

Both borrow the target's softcapped head for their candidate logits.

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
