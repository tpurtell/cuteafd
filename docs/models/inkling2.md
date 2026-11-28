# Inkling 2 Small

The v7 release keeps the EXL3 K4 MC cards, using K8 compact on both layouts.
V7 drops the original, K6 compact and FP8 compact comparison (TJ 2026-11-28); its last cards stay pinned in the [comparison matrix](../../scripts/bench/release-smoke-v7-mc-compact.json)'s retired list.
The native NVFP4 and MXFP8 cards are retired; their checkpoints and configs
remain supported.

Automatic KV sizing uses the shared 2M-token target on every card, including
32 GB / RTX 5090. Placement and expert onboarding are fixed at that target;
the pool then grows into remaining admitted memory, within native encoding
limits. `KV_MAX_TOKENS` is unset by default and can explicitly cap growth.
See [memory policy](../options.md#memory-and-pools).

Inkling 2 Small is being qualified as a first-class family on the shared engine
([#1930](https://github.com/tpurtell/cuteafd/issues/1930), v5.2). The initial
text-only public path is `cuteafd serve --family inkling2` or `./run.sh --wip
SLOT`, with one RTX and 2-8 Sparks. Admitted checkpoints and layouts report Serving without a repository, revision,
core-view or fidelity allowlist. Unsupported executor components and missing
native programs still refuse with their structural cause.
This page records checkpoint facts read from each checkpoint's own `config.json`
and safetensors headers. See [Inkling options](../options.md#inkling-serving)
for the accepted launch keys and current execution limits.

## Architecture

48 layers, hidden 4608, vocabulary 201024 stored (logits trimmed to 200058,
final softcap 30), untied embedding and unembedding. Attention is GQA 48/8 x 128
on every layer: 40 sliding-window-1024 layers and 8 global layers (5, 11, ...,
47), with a relative-position bias (`wr_du`, `rel_logits_proj`), per-token
per-head attention sinks (unscaled extra softmax-denominator logits), an output
gate, L2 clips on q, k and r, short causal K/V convolutions (static taps on local
layers; static plus per-token dynamic taps on global layers) and separate Q/K/V
input mixes on global layers. Residuals are softmax mixes over blocks of six
layers (F32 queries). Each layer also has `attn_sconv` and `mlp_sconv` short
convolutions on the hidden stream. The MoE is 384 routed experts, top-8, no
shared expert; `w13` is interleaved gate/up, the sigmoid uses the uncapped gate,
then tanh softcaps 4 (gate) and 25 (up) apply, and the route weight multiplies
the activated intermediate before the down projection's input quantization.

The bundled MTP drafter (`mtp.safetensors`) is a five-layer dSpark with block
size 8 over DFlare-mixed target features from layers 5, 11, 17, 23, 29, 35, 41
and the final norm, and an `rnn_hidden` Markov head. The bundled vision tower
is a 23-layer ViT with a 2x2 merge projector into the text hidden size.
`VISION=auto` admits the tower when the KV pool still meets `KV_MIN_TOKENS`;
explicit `VISION=off` leaves it unloaded.

## Checkpoints

| Checkpoint | Experts | Core |
|---|---|---|
| `thinkingmachines/Inkling-2-Small-MXFP8` @ `3febc62259ee92ea36e2fb6e8303371c4eb3922c` | MXFP8 on all 48 layers: E4M3 weights, UE8M0 scales per row and 32-wide K group, MXFP8 activations | BF16 |
| `thinkingmachines/Inkling-2-Small` @ `aac974122acd33ddb0d3e1f54433a9c5dd2249ce` | layers 0-39 NVFP4 (packed E2M1, E4M3 per-16 scales, F32 per-expert amax and outer denominator; per-token amax activations), layers 40-47 MXFP8 | BF16 |
| `wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC` | EXL3 K4 | MC views: `original` BF16, `k8-compact`, `k6-compact` (drafter included) |

The per-layer expert format is derived from each layer's tensor headers (the
presence of `w13_weight_amax` marks an NVFP4 layer), never from
`hf_quant_config.json`, which is only cross-checked. Expert math runs native:
MXFP8 W8A8, NVFP4 W4A4 and EXL3 trellis.

Serving status, layouts, configs and measurements land here stage by stage;
see the tracking issue for the current state.

## Changelog

| Release | Date | Changes | Cards |
| --- | --- | --- | --- |
| v6 | 2026-11-19 | Native NVFP4 and MXFP8, EXL3 multi-core views, bundled drafter, and dual-RTX serving. | [NVFP4 1 RTX](../../benchmarks/inkling2/2026-11-19-smoke-inkling-2-small-1rtx-4spark-inkling2-nvfp4-min/report.svg), [EXL3 1 RTX](../../benchmarks/inkling2/2026-11-19-smoke-inkling-2-small-exl3-k4-mc-1rtx-4spark-inkling2-exl3-min/report.svg), [EXL3 2 RTX](../../benchmarks/inkling2/2026-11-19-smoke-inkling-2-small-exl3-k4-mc-2rtx-6spark-inkling2-exl3-max/report.svg), [MXFP8 2 RTX](../../benchmarks/inkling2/2026-11-19-smoke-inkling-2-small-mxfp8-2rtx-4spark-inkling2-mxfp8-max/report.svg) |
